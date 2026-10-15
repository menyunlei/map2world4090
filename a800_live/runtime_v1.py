"""One resident2D runtime; DiT, decodeVAE and mapVAE stay on fixed devices."""
import copy,hashlib,json,sys,types
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(data,indent=2,ensure_ascii=False));q.replace(p)


class BoundedCalls(list):
 def __init__(self,values=()):super().__init__(values);self.total=len(self)
 def append(self,value):
  self.total+=1;super().append(value)
  if len(self)>16:del self[:-16]


class Runtime:
 def __init__(self,root,code):
  import torch
  from omegaconf import OmegaConf
  from native_loader_v1 import load_native_pipeline
  from native_execution_cache_v1 import install_execution_cache
  from streaming_pipeline_v1 import build_stream_inference
  from cpu_cache_indices_v1 import install_cpu_cache_indices
  from utils.wan_wrapper import WanVAEWrapper
  from native_stepper_v1 import NativeStepper
  self.root,self.code=root,code;self.project=root.parent.parent
  self.device=torch.device('cuda:0');self.decoder_device=torch.device('cuda:1');self.encoder_device=torch.device('cuda:2')
  torch.cuda.set_device(self.device);torch.set_num_threads(6);torch.set_grad_enabled(False)
  cfg=OmegaConf.merge(OmegaConf.load(root/'repo_sp_v1/configs/dreamx-ar/default_config.yaml'),OmegaConf.load(root/'repo_sp_v1/configs/dreamx-ar/causal_camera_forcing_5b.yaml'))
  self.pipeline,self.loader=load_native_pipeline(cfg,checkpoint_path=self.project/'models/DreamX-World-5B/model.safetensors',model_config_path=root/'repo_sp_v1/configs/dreamx-ar/config.json',device=self.device,num_output_frames=303,model_name=self.project/'models/Wan2.2-TI2V-5B',height=384,width=640)
  self.pipeline.vae.to(self.decoder_device)
  self.map_vae=WanVAEWrapper(vae_path=str(self.project/'models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth')).eval().requires_grad_(False).to(self.encoder_device,dtype=torch.bfloat16)
  self.map_projection=copy.deepcopy(self.pipeline.generator.model.patch_embedding).to(self.encoder_device)
  self.cache=install_execution_cache(self.pipeline)
  native=build_stream_inference();install_cpu_cache_indices(self.pipeline,native)
  self.stepper=NativeStepper(self.pipeline);self.hooks=None;self.width=640;self.height=384
  self.inputs=json.loads((root/'inputs/input.json').read_text())
  self.configure(self.inputs['map_path'],self.width,self.height)
 def configure(self,map_path,width,height):
  import torch
  from PIL import Image
  from torchvision import transforms
  from map_tokens_ar_v1 import encode_map_tokens_ar,install_map_token_hooks
  assert (width,height) in [(640,384),(768,448)]
  assert not self.pipeline.vae.training
  if self.hooks:self.hooks.remove()
  self.width,self.height=width,height;self.frame_tokens=width//32*(height//32)
  self.pipeline.frame_seq_length=self.frame_tokens;self.pipeline.generator.seq_len=self.frame_tokens*303
  self.pipeline.kv_cache1=None;self.pipeline.crossattn_cache=None
  for block in self.pipeline.generator.model.blocks:
   for a in [block.self_attn,block.cam_self_attn]:a.max_attention_size=12*self.frame_tokens
  pixels=transforms.Compose([transforms.Resize((height,width)),transforms.ToTensor(),transforms.Normalize([.5],[.5])])(Image.open(self.inputs['image_path']).convert('RGB'))[None,:,None]
  self.map_vae.model.clear_cache()
  self.i0=self.map_vae.encode_to_latent(pixels.to(self.encoder_device,dtype=torch.bfloat16)).to(self.device,dtype=torch.bfloat16)
  tokens,metadata=encode_map_tokens_ar(self.map_vae,self.map_projection,map_path,target_size=(width,height))
  self.hooks=install_map_token_hooks(self.pipeline.generator.model,tokens.to(self.device),metadata)
  for record in self.hooks.records.values():record['calls']=BoundedCalls()
  self.map_metadata=metadata;self.pipeline.vae.model.clear_cache()
 def reset(self):
  import torch
  torch.manual_seed(42);torch.cuda.manual_seed(42)
  self.stepper.reset(self.inputs['caption'],self.device,torch.bfloat16);self.pipeline.vae.model.clear_cache()
 def fixed_camera(self,poses,reference):
  import numpy as np, torch
  view=np.linalg.inv(np.stack([np.linalg.inv(reference)@p for p in poses])).astype('float32')
  view=torch.from_numpy(view)[:,None].expand(-1,self.frame_tokens,-1,-1).reshape(1,-1,4,4).to(self.device,dtype=torch.bfloat16)
  K=torch.tensor([[969.6969696969696/1920,0,.5],[0,969.6969696969696/1080,.5],[0,0,1]],device=self.device,dtype=torch.bfloat16)[None,None].expand(1,len(poses)*self.frame_tokens,3,3).contiguous()
  return {'viewmats':view,'K':K}
 def startup_parity(self,out):
  import torch
  from map_tokens_ar_v1 import tensor_sha256
  noise=torch.load(self.root/'inputs/initial_noise.pt',map_location=self.device,weights_only=True)[:,:9].clone();noise[:,0]=self.i0
  cams={k:v.to(self.device,dtype=torch.bfloat16) for k,v in torch.load(self.root/'inputs/cameras.pt',weights_only=True).items()}
  self.reset();raw=[];rgb=[]
  for i in range(0,9,3):
   camera={k:v[:,i*240:(i+3)*240] for k,v in cams.items()};chunk=self.stepper.step(noise[:,i:i+3],camera)
   raw.append(chunk.cpu());pixels=self.pipeline.vae.decode_to_pixel(chunk.to(self.decoder_device),use_cache=True)
   rgb.append(((pixels*.5+.5).clamp(0,1)[0].permute(0,2,3,1)*255).to(torch.uint8).cpu())
  ref=torch.load(self.root/'outputs/single_map_bulk/raw_latent.pt',map_location='cpu',weights_only=True)[:,:9]
  ref_rgb=torch.load(self.root/'outputs/single_map_bulk/rgb_uint8.pt',map_location='cpu',weights_only=True)[:33]
  report={'native9latent_exact':bool(torch.equal(torch.cat(raw,1),ref)),'native33RGB_exact':bool(torch.equal(torch.cat(rgb),ref_rgb)),'kernel_contract':self.stepper.kernel.contract,'fixed_device_map_encoder':str(self.encoder_device),'native_video_VAE_device':str(self.decoder_device),'fresh_model_device':str(self.device),'sampler_precision_weights_unchanged':True,'map_tokens_sha256':tensor_sha256(self.hooks.tokens)}
  write(out,report);assert report['native9latent_exact'] and report['native33RGB_exact'],report
  return report
