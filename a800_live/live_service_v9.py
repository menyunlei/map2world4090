"""Loopback-only live control/native RGB service. One resident two-A800 owner."""
import argparse
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
import types
from camera_motion_v2 import CameraMotionV2
from map_revision_v2 import initialize as initialize_map, request as request_map, apply_job as apply_map_job

TARGET_FPS=20
CONTROL_LEAD_SECONDS=1.15


class LiveStop(Exception): pass


def save(path, data):
    path = Path(path); temp = path.with_suffix(path.suffix + f'.{threading.get_ident()}.partial')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)); temp.replace(path)


class State:
    def __init__(self, root):
        self.root = root; self.inputs = json.loads((root/'inputs/input.json').read_text())
        self.condition = threading.Condition(); self.start_event = threading.Event()
        self.close_event = threading.Event(); self.stop_event = threading.Event()
        self.ready = False; self.running = False; self.closed = False; self.phase = '加载模型…'
        self.error = None; self.session = ''; self.keys = ''; self.seq = 0
        self.received_epoch = time.time(); self.last_touch = time.monotonic()
        self.latest = None; self.latest_meta = {}; self.path = [[0, 0]]; self.commands = []
        self.client_metrics = []; self.current_out = None; self.close_reason = None
        self.frames = deque(maxlen=32)
        initialize_map(self)

    def snapshot(self):
        with self.condition:
            return {'phase':self.phase,'ready':self.ready,'running':self.running,'closed':self.closed,
                'error':self.error,'session':self.session,'input_seq':self.seq,'input_keys':self.keys,
                'target_fps':TARGET_FPS,'native_width':640,'native_height':384,'max_session_seconds':1209/TARGET_FPS,
                'displayed_control_seq':self.latest_meta.get('control_seq'),
                'displayed_frame':self.latest_meta.get('frame_index'), 'path':self.path[-400:],
                'map_revision_id':self.map_revision_id,'map_sha256':self.map_sha256,
                'map_tokens_sha256':self.map_tokens_sha256,'map_changing':self.map_changing,'map_error':self.map_error,
                'map_source_size':list(self.map_size),'map_name':self.map_record.get('file_name','原始地图')}

    def post(self, path, data):
        if path=='/map':return request_map(self,data)
        with self.condition:
            if path == '/input':
                keys = ''.join(sorted(set(str(data.get('keys','')).lower()) & set('wasd')))
                self.seq += 1; self.keys = keys; self.received_epoch = time.time(); self.last_touch=time.monotonic()
                event={'seq':self.seq,'keys':keys,'received_epoch':self.received_epoch,
                       'client_epoch':data.get('client_epoch'),'session':self.session}
                self.commands.append(event)
                return event
            if path == '/start':
                if not self.ready or self.running: raise ValueError('Model not ready or session already running')
                self.session=f'live_{int(time.time()*1000)}'; self.running=True; self.keys=''
                self.phase='等待首个生成块…'; self.path=[[0,0]]; self.latest=None; self.latest_meta={}
                self.frames.clear(); self.condition.notify_all()
                self.stop_event.clear(); self.last_touch=time.monotonic(); self.start_event.set()
                return {'session':self.session,'seed':42,'explicit_new_I0_session':True,'map_revision_id':self.map_revision_id}
            if path == '/stop':
                self.keys='';self.stop_event.set();self.last_touch=time.monotonic();return {'stop_at_chunk_boundary':True}
            if path == '/close':
                if self.map_changing:raise ValueError('请等待地图编码完成后释放GPU')
                self.keys='';self.close_reason='user_close';self.close_event.set();self.stop_event.set();return {'close_at_chunk_boundary':True}
            if path == '/client-metric':
                event=dict(data,server_received_epoch=time.time());self.client_metrics.append(event)
                if self.current_out: save(self.current_out/'CLIENT_METRICS.json',self.client_metrics)
                return {'recorded':True}
            raise ValueError('Unknown endpoint')


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args): pass
        def respond(self, code, body=b'', content_type='application/json', headers=None):
            self.send_response(code);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            for key,value in (headers or {}).items():self.send_header(key,str(value))
            self.end_headers()
            if body:
                try:self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and origin not in ['http://127.0.0.1:18765','http://localhost:18765','http://127.0.0.1:18766','http://localhost:18766','http://127.0.0.1:18767','http://localhost:18767']:
                self.respond(403,b'{"error":"origin"}');return
            try:
                size=int(self.headers.get('Content-Length','0'))
                maximum=16*1024*1024 if self.path=='/map' else 4096
                if size<0 or size>maximum:
                    self.close_connection=True;self.respond(413,b'{"error":"body too large"}');return
                data=json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(data,dict):raise ValueError('JSON object required')
                result=state.post(self.path,data)
                self.respond(200,json.dumps(result).encode())
            except Exception as error:self.respond(409,json.dumps({'error':str(error)}).encode())
        def do_GET(self):
            from urllib.parse import urlparse,parse_qs
            request=urlparse(self.path)
            if request.path in ['/','/index.html']:
                self.respond(200,(state.root/'live_index_v15_buffered.html').read_bytes(),'text/html; charset=utf-8')
            elif request.path=='/map.png':
                with state.condition:payload=state.map_png
                self.respond(200,payload,'image/png')
            elif request.path=='/map-original.png':
                self.respond(200,state.original_map_png,'image/png')
            elif request.path=='/initial.png':
                self.respond(200,Path(state.inputs['image_path']).read_bytes(),'image/png')
            elif request.path in ['/maps','/maps/image','/maps/thumbnail']:
                import urllib.request
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18769'+self.path,timeout=8) as response:
                        self.respond(200,response.read(),response.headers.get('Content-Type','application/octet-stream'))
                except Exception:
                    self.respond(502,b'{"error":"map catalog unavailable"}')
            elif request.path=='/state':self.respond(200,json.dumps(state.snapshot()).encode())
            elif request.path=='/stream':
                q=parse_qs(request.query);after=int(q.get('after',['-1'])[0]);session=q.get('session',[''])[0]
                self.send_response(200)
                self.send_header('Content-Type','multipart/x-mixed-replace; boundary=dxframe')
                self.send_header('Cache-Control','no-store, no-transform')
                self.send_header('Connection','close');self.end_headers();self.close_connection=True
                self.connection.settimeout(3)
                try:
                    while True:
                        with state.condition:
                            state.condition.wait_for(lambda:state.closed or (state.frames and
                                (state.frames[-1][1]['frame_index']>after or state.session!=session)),timeout=1)
                            if state.closed:break
                            eligible=[(data,dict(meta)) for data,meta in state.frames
                                if meta['session']!=session or meta['frame_index']>after]
                        for data,meta in eligible:
                            # Copy under the condition, write outside: slow clients cannot stall the GPU.
                            headers=(f"--dxframe\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n"
                                f"X-Frame-Seq: {meta['frame_index']}\r\nX-Control-Seq: {meta['control_seq']}\r\n"
                                f"X-Session: {meta['session']}\r\n\r\n").encode('ascii')
                            self.wfile.write(headers+data+b'\r\n');self.wfile.flush()
                            session=meta['session'];after=meta['frame_index']
                except (BrokenPipeError,ConnectionResetError,TimeoutError,OSError):pass
            elif request.path=='/frame':
                q=parse_qs(request.query);after=int(q.get('after',['-1'])[0]);session=q.get('session',[''])[0]
                with state.condition:
                    state.condition.wait_for(lambda:state.closed or (state.latest is not None and
                        (state.latest_meta['frame_index']>after or state.session!=session)),timeout=1)
                    payload=state.latest;meta=dict(state.latest_meta)
                if payload is None or (meta.get('frame_index',-1)<=after and meta.get('session')==session):
                    self.respond(204);return
                self.respond(200,payload,'image/jpeg',{'X-Frame-Seq':meta['frame_index'],
                    'X-Control-Seq':meta['control_seq'],'X-Session':meta['session']})
            else:self.respond(404,b'{"error":"not found"}')
    return Handler


class Cameras:
    def __init__(self,state,device,playback):
        import numpy as np
        self.np=np;self.state=state;self.device=device;self.playback=playback
        self.position=np.zeros(3,dtype=np.float64);self.yaw=0.;self.previous=np.eye(4)
        self.previous_time=0.;self.records={};self.motion=CameraMotionV2()
    def __call__(self,start,count):
        import torch
        np=self.np
        # Bound future control lookahead. Keep only the compute/decode lead
        # needed for16fps, rather than generating a whole route in advance.
        first_pixel=0 if start==0 else 4*start-3
        while self.playback.get('start') is not None:
            desired=self.playback['start']+first_pixel/TARGET_FPS-CONTROL_LEAD_SECONDS
            if time.perf_counter()>=desired:break
            if self.state.stop_event.wait(min(.02,max(0,desired-time.perf_counter()))):raise LiveStop()
        if self.state.stop_event.is_set() or self.state.close_event.is_set():raise LiveStop()
        with self.state.condition:
            keys=self.state.keys;seq=self.state.seq;received=self.state.received_epoch
        reference=self.previous.copy();poses=[];global_poses=[]
        self.motion.command(keys)
        motion_anchors=[]
        for frame in range(start,start+count):
            pixel_time=0 if frame==0 else (1+4*(frame-1))/TARGET_FPS
            anchor=self.motion.advance_to(pixel_time)
            self.position=np.array(anchor['camera_position'],dtype=np.float64)
            self.yaw=anchor['camera_yaw_radians'];self.previous_time=pixel_time
            motion_anchors.append(anchor)
            c,s=math.cos(self.yaw),math.sin(self.yaw)
            pose=np.eye(4);pose[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]];pose[:3,3]=self.position
            global_poses.append(pose.copy());poses.append(np.linalg.inv(reference)@pose)
        self.previous=global_poses[-1]
        with self.state.condition:self.state.path.append([float(self.position[0]),float(self.position[2])])
        view=torch.from_numpy(np.linalg.inv(np.stack(poses)).astype(np.float32))
        view=view[:,None].expand(-1,240,-1,-1).reshape(1,-1,4,4).to(self.device,dtype=torch.bfloat16)
        k=torch.tensor([[969.6969696969696/1920,0,.5],[0,969.6969696969696/1080,.5],[0,0,1]],
                       device=self.device,dtype=torch.bfloat16)[None,None].expand(1,count*240,3,3).contiguous()
        self.records[start]={'latent_start':start,'latent_count':count,'control_seq':seq,'keys':keys,
            'input_received_epoch':received,'sampled_epoch':time.time(),
            'camera_position':self.position.tolist(),'camera_yaw_degrees':math.degrees(self.yaw),
            'global_c2w':np.stack(global_poses).tolist(),'native_pixel_camera_alignment':True,
            'motion_anchors':motion_anchors,'control_smoothing':self.motion.contract()}
        return {'viewmats':view,'K':k}


def gpu_owner(state):
    root=state.root;project=root.parent.parent;repo=root/'repo_sp_v1'
    sys.path[:0]=[str(root),str(repo)]
    package=types.ModuleType('pipeline');package.__path__=[str(repo/'pipeline')]
    package.__package__='pipeline';sys.modules['pipeline']=package
    import cv2
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from torchvision import transforms
    from torchvision.io import write_video
    from native_loader_v1 import load_native_pipeline
    from map_tokens_ar_v1 import encode_map_tokens_ar,install_map_token_hooks,tensor_sha256
    from live_camera_pipeline_v1 import build_live_inference
    torch.set_num_threads(8);torch.set_grad_enabled(False);torch.cuda.set_device(0)
    device=torch.device('cuda',0);decoder_device=torch.device('cuda',1)
    cfg=OmegaConf.merge(OmegaConf.load(repo/'configs/dreamx-ar/default_config.yaml'),
                        OmegaConf.load(repo/'configs/dreamx-ar/causal_camera_forcing_5b.yaml'))
    pipeline,loader=load_native_pipeline(cfg,checkpoint_path=project/'models/DreamX-World-5B/model.safetensors',
        model_config_path=repo/'configs/dreamx-ar/config.json',device=device,num_output_frames=303,
        model_name=project/'models/Wan2.2-TI2V-5B',height=384,width=640)
    save(root/'audit/LIVE_LOADER_SMOOTH_V9.json',loader)
    from native_execution_cache_v1 import install_execution_cache
    execution_cache=install_execution_cache(pipeline)
    save(root/'audit/LIVE_EXECUTION_CACHE_SMOOTH_V9.json',execution_cache.audit())
    pixels=transforms.Compose([transforms.Resize((384,640)),transforms.ToTensor(),
        transforms.Normalize([.5],[.5])])(Image.open(state.inputs['image_path']).convert('RGB'))[None,:,None]
    i0=pipeline.vae.encode_to_latent(pixels.to(device=device,dtype=torch.bfloat16)).to(torch.bfloat16)
    tokens,map_meta=encode_map_tokens_ar(pipeline.vae,pipeline.generator.model.patch_embedding,
                                        state.inputs['map_path'])
    hooks=install_map_token_hooks(pipeline.generator.model,tokens,map_meta)
    state.map_tokens_sha256=map_meta['tokens_sha256'];state.map_record['tokens_sha256']=map_meta['tokens_sha256']
    state.map_record['map_encoding_metadata']=map_meta
    function=build_live_inference(pipeline);live=types.MethodType(function,pipeline)
    save(root/'audit/LIVE_FUNCTION_CONTRACT_SMOOTH_V9.json',function.live_audit)
    pipeline.vae.to(decoder_device)
    base_noise=torch.load(root/'inputs/initial_noise.pt',map_location=device,weights_only=True)
    base_noise[:,0]=i0
    fixed_camera={key:value.to(device,dtype=torch.bfloat16) for key,value in
        torch.load(root/'inputs/cameras.pt',weights_only=True).items()}
    pipeline.vae.model.clear_cache();torch.manual_seed(42);torch.cuda.manual_seed(42)
    warm_chunks=[]
    def warm_callback(chunk,start,count):
        decoded=pipeline.vae.decode_to_pixel(chunk.to(decoder_device),use_cache=True)
        warm_chunks.append(((decoded*.5+.5).clamp(0,1)[0].permute(0,2,3,1)*255).to(torch.uint8).cpu())
    _,warm_latent=live(noise=base_noise[:,:9].clone(),text_prompts=[state.inputs['caption']],
                     y=None,y_camera=fixed_camera,return_latents=True,chunk_callback=warm_callback)
    ref=torch.load(root/'outputs/single_map_bulk/raw_latent.pt',weights_only=True)[:,:9]
    ref_rgb=torch.load(root/'outputs/single_map_bulk/rgb_uint8.pt',weights_only=True)[:33]
    warm_report={'raw_latent_exact':bool(torch.equal(warm_latent.cpu(),ref)),
        'all33_native_rgb_exact':bool(torch.equal(torch.cat(warm_chunks),ref_rgb)),
        'no_live_input_used_for_warmup':True,'live_camera_AST_native_math_preserved':True}
    save(root/'audit/LIVE_WARMUP_PARITY_SMOOTH_V9.json',warm_report)
    assert warm_report['raw_latent_exact'] and warm_report['all33_native_rgb_exact'],warm_report
    with state.condition:state.ready=True;state.phase='就绪：点击开始';state.condition.notify_all()
    print('LIVE READY; native33frame warmup raw/RGB exact',flush=True)
    while not state.close_event.is_set():
        if not state.start_event.wait(.5):
            if time.monotonic()-state.last_touch>28800:state.close_reason='idle_timeout_8h';state.close_event.set()
            continue
        state.start_event.clear()
        if state.map_job is not None:
            apply_map_job(state,pipeline,hooks,device,decoder_device,encode_map_tokens_ar);continue
        session=state.session
        out=root/'outputs/live_smooth_v9'/session;out.mkdir(parents=True)
        state.current_out=out;state.client_metrics=[]
        map_revision=dict(state.map_record);save(out/'MAP_REVISION.json',map_revision)
        generator=torch.Generator(device='cpu').manual_seed(42)
        noise=torch.randn((1,303,48,24,40),generator=generator,dtype=torch.bfloat16).to(device)
        noise[:,0]=i0
        pipeline.vae.model.clear_cache();torch.manual_seed(42);torch.cuda.manual_seed(42)
        decoded_queue=deque();decoded_condition=threading.Condition()
        jobs=queue.Queue(maxsize=2);errors=[];raw_chunks=[];rgb_chunks=[];chunk_records=[];presentations=[]
        done=threading.Event();playback={'start':None};request_epoch=time.time();request=time.perf_counter()
        cameras=Cameras(state,device,playback)
        decode_stream=torch.cuda.Stream(device=decoder_device)
        def decoder():
            torch.cuda.set_device(decoder_device)
            with torch.no_grad(),torch.cuda.stream(decode_stream):
                while True:
                    job=jobs.get()
                    if job is None:break
                    chunk,start,count,event=job
                    decode_began=time.time()
                    gpu_began=torch.cuda.Event(enable_timing=True);gpu_end=torch.cuda.Event(enable_timing=True)
                    try:
                        gpu_began.record(decode_stream)
                        decode_stream.wait_event(event);source=chunk
                        chunk=chunk.to(decoder_device,non_blocking=True);chunk.record_stream(decode_stream)
                        pixels=pipeline.vae.decode_to_pixel(chunk,use_cache=True)
                        rgb=((pixels*.5+.5).clamp(0,1)[0].permute(0,2,3,1)*255).to(torch.uint8).cpu()
                        gpu_end.record(decode_stream);gpu_end.synchronize()
                        decode_gpu_ms=gpu_began.elapsed_time(gpu_end)
                        raw_chunks.append(chunk.cpu());rgb_chunks.append(rgb)
                        record=dict(cameras.records[start],decode_start_epoch=decode_began,decode_gpu_ms=decode_gpu_ms,model_callback_epoch=callback_times[start],map_revision_id=map_revision['map_revision_id'],
                                    map_sha256=map_revision['map_sha256'],rgb_ready_epoch=time.time(),pixel_count=len(rgb),
                                    native_rgb_sha256=tensor_sha256(rgb))
                        chunk_records.append(record)
                        first=0 if start==0 else 4*start-3
                        encoded=[]
                        for i,frame in enumerate(rgb.numpy()):
                            ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,88])
                            assert ok
                            meta={'frame_index':first+i,'control_seq':record['control_seq'],'session':session,
                                  'input_received_epoch':record['input_received_epoch'],'rgb_ready_epoch':record['rgb_ready_epoch'],
                                  'map_revision_id':map_revision['map_revision_id'],'map_sha256':map_revision['map_sha256']}
                            encoded.append((jpeg.tobytes(),meta))
                        record['jpeg_ready_epoch']=time.time()
                        with decoded_condition:decoded_queue.extend(encoded);decoded_condition.notify_all()
                        save(out/'CHUNKS.json',chunk_records)
                    except BaseException as error:
                        errors.append(repr(error));state.stop_event.set();break
            done.set()
            with decoded_condition:decoded_condition.notify_all()
        def publish():
            deadline=None
            while True:
                with decoded_condition:
                    decoded_condition.wait_for(lambda:decoded_queue or done.is_set(),timeout=1)
                    if not decoded_queue:
                        if done.is_set():break
                        continue
                    jpeg,meta=decoded_queue.popleft()
                if deadline is None:
                    deadline=time.perf_counter()+.55;playback['start']=deadline
                gap=deadline-time.perf_counter()
                if gap>0:time.sleep(gap)
                published=time.time();meta['published_epoch']=published
                meta['input_to_publish_seconds']=published-meta['input_received_epoch']
                presentations.append(dict(meta))
                with state.condition:
                    state.latest=jpeg;state.latest_meta=meta;state.frames.append((jpeg,dict(meta)))
                    state.phase='实时生成中';state.condition.notify_all()
                deadline=max(deadline+1/TARGET_FPS,time.perf_counter())
            save(out/'PRESENTATIONS.json',presentations)
        decoding=threading.Thread(target=decoder,daemon=True);publishing=threading.Thread(target=publish,daemon=True)
        decoding.start();publishing.start()
        callback_times={}
        def callback(chunk,start,count):
            callback_times[start]=time.time()
            if errors:raise RuntimeError(errors)
            event=torch.cuda.Event();event.record(torch.cuda.current_stream(device));jobs.put((chunk,start,count,event))
        reason='native303_latent_limit'
        try:
            live(noise=noise,text_prompts=[state.inputs['caption']],y=None,y_camera=None,
                 return_latents=True,chunk_callback=callback,camera_callback=cameras)
        except LiveStop:reason='user_stop_at_native_chunk_boundary'
        except BaseException as error:errors.append(repr(error));reason='implementation_failure'
        jobs.put(None);decoding.join();publishing.join()
        with state.condition:state.running=False;state.ready=False;state.phase='保存原始结果…';state.condition.notify_all()
        if raw_chunks:
            latent=torch.cat(raw_chunks,dim=1);rgb=torch.cat(rgb_chunks,dim=0)
            torch.save(latent,out/'raw_latent.pt');torch.save(rgb,out/'rgb_uint8.pt')
            write_video(str(out/'native.mp4'),rgb,fps=TARGET_FPS)
            for frame in sorted(set([0,len(rgb)//4,len(rgb)//2,3*len(rgb)//4,len(rgb)-1])):
                Image.fromarray(rgb[frame].numpy()).save(out/f'native_frame_{frame:04d}.png')
        hooks.save_audit(out/'MAP_HOOK_AUDIT.json')
        save(out/'COMMANDS.json',[event for event in state.commands if event['session']==session])
        save(out/'CLIENT_METRICS.json',state.client_metrics)
        save(out/'COMPLETE.json',{'session':session,'reason':reason,'errors':errors,
            'native_latent_chunks':len(raw_chunks),'native_rgb_frames':sum(len(x) for x in rgb_chunks),
            'displayed_native_frames':len(presentations),'seconds_including_pacing_and_archive':time.perf_counter()-request,
            'camera_control_sampled_live_per_chunk':True,'precomputed_route_used':False,
            'native_sampler_and_codec_unchanged':True,'JPEG_transport_quality':88,
            'seed':42,'new_weights_or_training':False,'target_fps':TARGET_FPS,'native_width':640,'native_height':384,'control_lead_seconds':CONTROL_LEAD_SECONDS,'browser_metrics':state.client_metrics,
            'motion_smoothing':cameras.motion.contract(),'persistent_native_JPEG_transport':True,'execution_cache_audit':execution_cache.audit(),
            'map_revision':map_revision,'all30_map_prefixes_verified':hooks.audit()['all_block_cache_prefixes_verified']})
        if errors:
            with state.condition:state.error='; '.join(errors);state.phase='失败：未自动重试'
            state.close_event.set();break
        with state.condition:state.ready=True;state.phase='就绪：本段已保存';state.last_touch=time.monotonic();state.condition.notify_all()


def main():
    global TARGET_FPS
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=18768);p.add_argument('--fps',type=int,choices=[20,24],required=True);args=p.parse_args();TARGET_FPS=args.fps
    state=State(args.root.resolve());owner_path=state.root/'audit/LIVE_OWNER_SMOOTH_V9.json'
    assert not owner_path.exists(),'Do not duplicate live owner or reset history'
    save(owner_path,{'pid':os.getpid(),'epoch':time.time(),'port':args.port,'bind':'127.0.0.1',
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'idle_timeout_seconds':28800,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'automatic_retry':False})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(state));server.daemon_threads=True
    http_thread=threading.Thread(target=server.serve_forever,daemon=True);http_thread.start()
    try:gpu_owner(state)
    except BaseException as error:
        state.close_reason='implementation_failure'
        save(state.root/'audit/LIVE_FAILURE_SMOOTH_V9.json',{'epoch':time.time(),'error':repr(error),'automatic_retry':False})
        raise
    finally:
        with state.condition:state.closed=True;state.condition.notify_all()
        save(state.root/'audit/LIVE_EXIT_SMOOTH_V9.json',{'epoch':time.time(),'safe_chunk_boundary':True,
             'source_and_old_experiments_unchanged':True,'foreign_tasks_signalled':False,
             'close_reason':state.close_reason,'idle_timeout_seconds':28800})
        server.shutdown()


if __name__=='__main__':main()
