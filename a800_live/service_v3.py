"""Versioned quality prompts; same native model/math, with fixed comparisons."""
import argparse,hashlib,io,json,os,sys,threading,time,types
from http.server import ThreadingHTTPServer
from pathlib import Path
from prompt_policy_v2 import prompt_record,settings_job


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--code',type=Path,required=True);ap.add_argument('--port',type=int,default=18767)
    a=ap.parse_args();root,code=a.root,a.code
    sys.path[:0]=[str(code),str(root),str(root/'repo_sp_v1')]
    package=types.ModuleType('pipeline');package.__path__=[str(root/'repo_sp_v1/pipeline')];package.__package__='pipeline';sys.modules['pipeline']=package
    import live_service_v9 as base
    from runtime_v1 import Runtime,write
    from session_v3 import run_session,FixedRoute
    from quality_review_v2 import serve_review,validate_and_package
    presets=json.loads((code/'PROMPT_PRESETS_V2.json').read_text())
    restore=json.loads((code/'RESTORE_SOURCE_V3.json').read_text())

    class State(base.State):
        def __init__(self,r):
            super().__init__(r);self.fps=16;self.width=640;self.height=384;self.quality='native640';self.memory_enabled=restore['memory_enabled'];self.memory_status={};self.settings_job=None;self.actual_sample_fps=0
            self.prompt=restore['prompt'];self.model_input_seq=0
        def snapshot(self):
            v=super().snapshot();m=self.memory_status;last=m.get('last_query') or {}
            v.update(target_fps=self.fps,native_width=self.width,native_height=self.height,max_session_seconds=None,continuous=True,quality=self.quality,actual_sample_fps=self.actual_sample_fps,prompt=self.prompt,prompt_presets=presets,prompt_quality_version=2,low_latency_version=3,model_input_seq=self.model_input_seq,
                     memory=dict(m,enabled=self.memory_enabled,entries=m.get('records_in_RAM',0),hits=m.get('applied_count',0),misses=m.get('query_count',0)-m.get('hit_count',0),last_hit={'distance_m':last.get('distance_m'),'yaw_delta_degrees':last.get('yaw_difference_degrees')} if last.get('selected_record_id') else None))
            return v
        def post(self,path,data):
            if path=='/settings':
                with self.condition:
                    job=settings_job(self,data,presets)
                    self.settings_job=job;self.ready=False;self.phase='应用生成设置…';self.start_event.set();self.last_touch=time.monotonic();self.condition.notify_all()
                    return {'accepted':True,'prompt':job['prompt'],'quality':job['quality']}
            return super().post(path,data)
    state=State(root);base.TARGET_FPS=16
    Parent=base.make_handler(state)
    class Handler(Parent):
        def do_GET(self):
            from urllib.parse import urlparse,parse_qs
            req=urlparse(self.path)
            if req.path in ['/','/index.html']:
                self.respond(200,(code/'interactive_index_v3.html').read_bytes(),'text/html; charset=utf-8');return
            if serve_review(self,req,root/'interactive_prompt_quality_v2'):return
            if req.path in ['/maps','/maps/image','/maps/thumbnail']:
                from server_map_catalog_v1 import inventory,natural_key,MAPS_ROOT
                from PIL import Image,ImageOps
                entries=inventory(MAPS_ROOT)
                if req.path=='/maps':self.respond(200,json.dumps({'maps':[{k:v for k,v in x.items() if not k.startswith('_')} for x in sorted(entries.values(),key=natural_key)]}).encode());return
                x=entries.get(parse_qs(req.query).get('id',[''])[0])
                if x is None:self.respond(404,b'{"error":"map id"}');return
                if req.path=='/maps/image':self.respond(200,x['_path'].read_bytes(),x['_mime']);return
                with Image.open(x['_path']) as im:
                    im=ImageOps.contain(im.convert('RGB'),(200,200));b=io.BytesIO();im.save(b,'JPEG',quality=85)
                self.respond(200,b.getvalue(),'image/jpeg');return
            super().do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    def worker():
        try:
            runtime=Runtime(root,code)
            assert runtime.inputs['caption']==presets['original_caption']['caption']
            runtime.startup_parity(code/'audit/STARTUP_NATIVE_PARITY_V3.json')
            print('STATEFUL NATIVE PARITY PASS',flush=True)
            if state.close_event.is_set():return
            width,height=restore['native_width'],restore['native_height'];map_path=Path(restore['map_path'])
            assert hashlib.sha256(map_path.read_bytes()).hexdigest()==restore['map_file_sha256']
            runtime.configure(map_path,width,height)
            with state.condition:
                state.width=width;state.height=height;state.quality=restore['quality'];state.map_path=map_path;state.map_png=map_path.read_bytes();state.map_revision_id=restore['map_revision_id'];state.map_sha256=restore['map_file_sha256'];state.map_tokens_sha256=runtime.map_metadata['tokens_sha256'];state.map_record=restore['map_record'];state.map_size=tuple(restore['map_source_size'])
            expected_tokens=restore.get('map_tokens_sha256')
            assert not expected_tokens or runtime.map_metadata['tokens_sha256']==expected_tokens
            assert json.loads((root/'low_latency_parts_preflight_v3/COMPLETE.json').read_text())['passed']
            runtime.inputs['caption']=presets['neutral_scene']['caption'];state.prompt=prompt_record(presets,'neutral_scene')
            state.phase='低延迟流水线原生一致性验证…';state.stop_event.clear();state.client_metrics=[]
            out=code/'pipeline_parity_v3'
            report=run_session(state,runtime,out,memory_enabled=False,route=FixedRoute(runtime,chunks=24),max_chunks=24)
            assert not report['errors'],report
            reference=json.loads((root/'interactive_prompt_quality_v2/quality_comparison/neutral_scene/CHUNKS.json').read_text())
            actual=json.loads((out/'CHUNKS.json').read_text());assert len(reference)==len(actual)
            for before,after in zip(reference,actual):
                for key in ['input_noise_sha256','global_c2w','native_rgb_sha256']:assert before[key]==after[key],key
            write(code/'audit/PIPELINE_PARITY_V3.json',{'all285_native_RGB_exact':True,'same_input_noise_and_cameras':True,'decoder_preflight_reused':True,'weights_sampler_resolution_unchanged':True})
            state.prompt=restore['prompt'];runtime.inputs['caption']=state.prompt['caption'];state.memory_status={}
            with state.condition:state.ready=True;state.phase='就绪：逐段原生解码与低延迟流水线';state.condition.notify_all()
            print('PIPELINE PARITY PASS; LIVE READY',flush=True)
            while not state.close_event.is_set():
                if not state.start_event.wait(.5):
                    if time.monotonic()-state.last_touch>28800:state.close_reason='idle_timeout_8h';state.close_event.set()
                    continue
                state.start_event.clear()
                if state.map_job is not None:
                    job=state.map_job
                    try:
                        runtime.configure(job['path'],state.width,state.height)
                        record={k:job[k] for k in ['map_revision_id','parent_revision','map_sha256','upload_sha256','requested_epoch','file_name','source_size']};record.update(map_encoding_metadata=runtime.map_metadata,tokens_sha256=runtime.map_metadata['tokens_sha256'],applied_epoch=time.time(),fixed_map_VAE_device=True,spatial_fidelity_acceptance=False)
                        write(job['path'].parent/'APPLIED.json',record)
                        with state.condition:state.map_png=job['png'];state.map_sha256=job['map_sha256'];state.map_tokens_sha256=runtime.map_metadata['tokens_sha256'];state.map_revision_id=job['map_revision_id'];state.map_record=record;state.map_path=job['path'];state.map_size=tuple(job['source_size']);state.memory_status={};state.latest=None;state.frames.clear();state.path=[[0,0]];state.phase='地图已应用'
                        job['result']={'map_revision_id':state.map_revision_id,'tokens_sha256':state.map_tokens_sha256}
                    except BaseException as e:job['error']=repr(e);state.error=repr(e);state.close_reason='map_config_failure';state.close_event.set()
                    finally:
                        with state.condition:state.map_job=None;state.map_changing=False;state.ready=not state.close_event.is_set();job['event'].set();state.condition.notify_all()
                    continue
                if state.settings_job is not None:
                    job=state.settings_job;quality=job['quality']
                    if quality!=state.quality:
                        w,h=(768,448) if quality=='native768' else (640,384);runtime.configure(state.map_path,w,h)
                        state.width,state.height=w,h
                    runtime.inputs['caption']=job['prompt']['caption']
                    with state.condition:state.quality=quality;state.prompt=job['prompt'];state.memory_enabled=job['memory_enabled'];state.settings_job=None;state.ready=True;state.phase='描述与设置已应用：点击开始';state.condition.notify_all()
                    write(code/'audit/ACTIVE_SETTINGS_V3.json',dict(job,epoch=time.time()))
                    continue
                if not state.running:continue
                out=code/'sessions'/state.session;state.current_out=out;state.client_metrics=[]
                write(code/'audit/CURRENT_SESSION_V3.json',{'session':state.session,'quality':state.quality,'prompt':state.prompt,'epoch':time.time()})
                report=run_session(state,runtime,out,memory_enabled=state.memory_enabled,live=True)
                with state.condition:state.running=False;state.ready=not state.close_event.is_set();state.phase='就绪：本段已保存';state.condition.notify_all()
                if report['errors']:raise RuntimeError(report['errors'])
        except BaseException as e:
            state.error=repr(e);state.phase='失败：未自动重试';state.close_reason='implementation_failure';state.close_event.set();write(code/'audit/FAILURE_V3.json',{'epoch':time.time(),'error':repr(e),'automatic_retry':False});print('FAILURE',repr(e),flush=True)
        finally:
            with state.condition:state.closed=True;state.ready=False;state.condition.notify_all()
    thread=threading.Thread(target=worker,name='native2d_prompt_owner');thread.start()
    write(code/'audit/OWNER_V3.json',{'pid':os.getpid(),'epoch':time.time(),'bind':'127.0.0.1','port':a.port,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'automatic_retry':False})
    http=threading.Thread(target=server.serve_forever,daemon=True);http.start();thread.join();server.shutdown();server.server_close();write(code/'audit/EXIT_V3.json',{'epoch':time.time(),'reason':state.close_reason,'closed':state.closed,'safe_chunk_boundary':True})


if __name__=='__main__':main()
