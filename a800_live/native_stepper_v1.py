"""Stateful bounded-memory native AR chunks; math extracted from native AST."""
import ast,copy,hashlib,inspect,textwrap


def build_step_kernel(native_class):
    source=textwrap.dedent(inspect.getsource(native_class.inference));tree=ast.parse(source)
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='(i, current_num_frames)']
    assert len(loops)==1
    body=loops[0].body
    denoise=[n for n in body if isinstance(n,ast.For) and ast.unparse(n.target)=='(index, current_timestep)']
    refresh=[n for n in body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='self.generator' and any(k.arg=='timestep' and ast.unparse(k.value)=='context_timestep' for k in n.value.keywords)]
    context=[n for n in body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='context_timestep']
    assert len(denoise)==len(refresh)==len(context)==1
    original=copy.deepcopy(denoise[0]);loop=copy.deepcopy(original)
    changes=[]
    for n in ast.walk(loop):
        if isinstance(n,ast.If) and ast.unparse(n.test)=='i == 0':n.test=ast.Name(id='first',ctx=ast.Load());changes.append(n)
    assert len(changes)==1
    reversed_loop=copy.deepcopy(loop)
    for n in ast.walk(reversed_loop):
        if isinstance(n,ast.If) and ast.unparse(n.test)=='first':n.test=ast.parse('i == 0',mode='eval').body
    assert ast.dump(reversed_loop,include_attributes=False)==ast.dump(original,include_attributes=False)
    fn=ast.parse('def native_chunk_step(self,noisy_input,conditional_dict,y_camera_latents,current_start_frame,first):\n    batch_size,current_num_frames=noisy_input.shape[:2]\n    noise=noisy_input\n    latents=noisy_input\n    y_latents=None\n    first_frame_mask_block=torch.ones_like(noisy_input)\n    if first:first_frame_mask_block[:,0]=0\n').body[0]
    fn.body += [loop,copy.deepcopy(context[0]),copy.deepcopy(refresh[0]),ast.parse('return denoised_pred').body[0]]
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));namespace=dict(native_class.inference.__globals__)
    exec(compile(module,'<native_stateful_chunk_v1>','exec'),namespace)
    f=namespace['native_chunk_step'];f.contract={'native_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'native_denoising_loop_restored_AST_equal':True,'native_context_refresh_AST_equal':True,'declared_change':'i==0 is first; caller supplies one native3-latent chunk and persistent KV','spatial_steps':4,'weights_changed':False,'frame_interpolation':False}
    return f


class NativeStepper:
    def __init__(self,pipeline):
        from pipeline.pipeline_causal_camera import CausalCameraInferencePipeline
        self.pipeline=pipeline;self.kernel=build_step_kernel(CausalCameraInferencePipeline);self.cursor=0;self.conditional=None
    def reset(self,caption,device,dtype):
        p=self.pipeline
        if p.kv_cache1 is None:p._initialize_kv_cache(1,dtype,device,303);p._initialize_crossattn_cache(1,dtype,device)
        for c in p.kv_cache1:
            for n in ['global_end_index','local_end_index']:c[n].zero_()
        for c in p.crossattn_cache:c['is_init']=False
        self.conditional=p.text_encoder(text_prompts=[caption]);self.cursor=0
    def step(self,noise,camera):
        assert noise.shape[1]==3
        result=self.kernel(self.pipeline,noise,self.conditional,camera,self.cursor,self.cursor==0)
        self.cursor+=3;return result
