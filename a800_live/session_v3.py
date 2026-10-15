"""Fresh native chunk production, bounded RAM, and lossless raw chunk archive."""
import hashlib,json,math,queue,threading,time
from collections import deque
from pathlib import Path
from runtime_v1 import write
from slowdown_profile_v1 import ChunkTrace
from camera_motion_v2 import CameraMotionV2


class LiveCamera:
 def __init__(self,state,runtime,playback):
  import numpy as np
  self.state,self.runtime,self.playback=state,runtime,playback;self.motion=CameraMotionV2();self.previous=np.eye(4);self.records={}
 def __call__(self,start):
  import numpy as np
  first=0 if start==0 else 4*start-3
  while self.playback['start'] is not None:
   desired=self.playback['start']+first/self.state.fps-.85
   if time.perf_counter()>=desired:break
   if self.state.stop_event.wait(min(.02,max(0,desired-time.perf_counter()))):return None
  if self.state.stop_event.is_set() or self.state.close_event.is_set():return None
  with self.state.condition:keys,seq,received=self.state.keys,self.state.seq,self.state.received_epoch
  with self.state.condition:self.state.model_input_seq=seq
  self.motion.command(keys);poses=[];anchors=[]
  for f in range(start,start+3):
   t=0 if f==0 else (1+4*(f-1))/self.state.fps;a=self.motion.advance_to(t);anchors.append(a)
   c,s=math.cos(a['camera_yaw_radians']),math.sin(a['camera_yaw_radians']);p=np.eye(4);p[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]];p[:3,3]=a['camera_position'];poses.append(p)
  camera=self.runtime.fixed_camera(poses,self.previous);self.previous=poses[-1]
  record={'latent_start':start,'control_seq':seq,'keys':keys,'input_received_epoch':received,'sampled_epoch':time.time(),'camera_position':anchors[-1]['camera_position'],'camera_yaw_degrees':anchors[-1]['camera_yaw_degrees'],'model_time_seconds':anchors[-1]['camera_time_seconds'],'global_c2w':[p.tolist() for p in poses]}
  with self.state.condition:self.state.path.append([float(poses[-1][0,3]),float(poses[-1][2,3])]);self.state.path=self.state.path[-600:]
  return camera,record


class FixedRoute:
 def __init__(self,runtime,chunks=24):
  import numpy as np
  self.runtime=runtime;self.previous=np.eye(4);self.chunks=chunks
 def __call__(self,start):
  import numpy as np
  n=start//3
  if n>=self.chunks:return None
  poses=[]
  for f in range(start,start+3):
   t=0 if f==0 else (1+4*(f-1))/16
   if t<=6:z=2*t;yaw=0
   elif t<=7:z=12;yaw=math.radians(45)*(t-6)
   elif t<=8:z=12;yaw=math.radians(45)*(8-t)
   elif t<=14:z=12-2*(t-8);yaw=0
   else:z=0;yaw=0
   c,s=math.cos(yaw),math.sin(yaw);p=np.eye(4);p[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]];p[:3,3]=[0,0,z];poses.append(p)
  cam=self.runtime.fixed_camera(poses,self.previous);self.previous=poses[-1]
  return cam,{'latent_start':start,'control_seq':0,'keys':'fixed matched out-turn-return','input_received_epoch':time.time(),'sampled_epoch':time.time(),'camera_position':[0,0,float(z)],'camera_yaw_degrees':math.degrees(yaw),'model_time_seconds':t,'global_c2w':[p.tolist() for p in poses]}


def run_session(state,runtime,out,*,memory_enabled=False,route=None,live=False,max_chunks=None,verify_memory=False):
 import cv2,numpy as np,torch,imageio.v2 as imageio
 from spatial_memory_v1 import SpatialMemoryBank,NativeSinkMemoryAdapter
 from native_pixel_parts_v3 import NativePixelParts
 out=Path(out);out.mkdir(parents=True,exist_ok=False)
 write(out/'PROMPT.json',dict(state.prompt,checkpoint_unchanged=True,sampler_steps=4,negative_prompt_executed=False))
 runtime.reset();p=runtime.pipeline;device=runtime.device;decode_device=runtime.decoder_device
 bank=SpatialMemoryBank(out/'spatial_memory',session_id=out.name,map_revision_id=state.map_revision_id)
 adapter=NativeSinkMemoryAdapter(p,verify_tensor_hashes=verify_memory);pixel_parts=NativePixelParts(p.vae)
 source_noise=torch.Generator(device='cpu').manual_seed(42)
 jobs=queue.Queue(maxsize=2);jpeg_jobs=queue.Queue(maxsize=6);archive=queue.Queue(maxsize=3)
 frames=deque();condition=threading.Condition();done=threading.Event();playback={'start':None}
 records=[];presentations=[];mem_events=[];errors=[];ticks=time.perf_counter();cam_provider=route or LiveCamera(state,runtime,playback)
 state.memory_status={'enabled':memory_enabled,'entries':0,'hits':0,'misses':0,'status':'候选2D视觉记忆'}
 decode_stream=torch.cuda.Stream(device=decode_device)
 def offer(q,item,consumer):
  while consumer.is_alive():
   try:q.put(item,timeout=.1);return
   except queue.Full:continue
  raise RuntimeError('Downstream worker exited before queue drain')
 def fail(stage,error):errors.append(stage+':'+repr(error));state.stop_event.set()
 def writer():
  movie=None
  try:
   movie=imageio.get_writer(str(out/'native.mp4'),fps=state.fps,codec='libx264',quality=None,macro_block_size=1,ffmpeg_params=['-crf','18','-threads','2'],ffmpeg_log_level='error')
   while True:
    item=archive.get()
    if item is None:break
    raw,rgb,record=item;target=out/'chunks';target.mkdir(exist_ok=True)
    torch.save({'latent':raw,'rgb_uint8':rgb,'record':record},target/f"{record['latent_start']:08d}.pt")
    for frame in rgb.numpy():movie.append_data(frame)
  except BaseException as e:fail('archive',e)
  finally:
   if movie is not None:movie.close()
 def encoder():
  current=[]
  try:
   while True:
    item=jpeg_jobs.get()
    if item is None:break
    raw,rgb,record,part,last=item;current.append(rgb);encoded=[]
    for offset,frame in enumerate(rgb.numpy()):
     ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
     meta={'frame_index':record['first_frame']+part['pixel_offset']+offset,'control_seq':record['control_seq'],'session':out.name,'input_received_epoch':record['input_received_epoch'],'rgb_ready_epoch':part['rgb_ready_epoch'],'map_revision_id':state.map_revision_id}
     encoded.append((jpeg.tobytes(),meta))
    part['jpeg_ready_epoch']=time.time()
    if live:
     with condition:frames.extend(encoded);condition.notify_all()
    if last:
     whole=torch.cat(current,dim=0);current=[];record['native_rgb_sha256']=hashlib.sha256(whole.numpy().tobytes()).hexdigest();record['jpeg_ready_epoch']=time.time();record['pixel_count']=len(whole);records.append(record)
     with state.condition:state.actual_sample_fps=(sum(r['pixel_count'] for r in records)-1)/max(.001,time.perf_counter()-ticks)
     offer(archive,(raw,whole,record),write_thread)
     if not live:
      with state.condition:state.phase=f'流水线原生验证：{len(records)}块';state.condition.notify_all()
  except BaseException as e:fail('JPEG',e)
  finally:
   done.set()
   with condition:condition.notify_all()
 def decoder():
  torch.cuda.set_device(decode_device)
  try:
   with torch.no_grad(),torch.cuda.stream(decode_stream):
    while True:
     item=jobs.get()
     if item is None:break
     chunk,raw,record,event=item
     trace=ChunkTrace(record['latent_start'],decode_stream);trace.gpu_mark('before_source_wait');decode_stream.wait_event(event);trace.gpu_mark('after_source_wait')
     chunk=chunk.to(decode_device,non_blocking=True);chunk.record_stream(decode_stream);trace.gpu_mark('after_p2p')
     record['parts']=[];offset=0;native_times=[]
     for pixels,markers in pixel_parts.pixels(chunk):
      converted=((pixels*.5+.5).clamp(0,1)[0].permute(0,2,3,1)*255).to(torch.uint8);rgb=converted.cpu()
      native_ms=markers['native_gpu_begin'].elapsed_time(markers['native_gpu_end']);native_times.append(native_ms)
      part={'latent_offset':markers['latent_offset'],'pixel_offset':offset,'pixel_count':len(rgb),'rgb_ready_epoch':time.time(),'native_decoder_gpu_ms':native_ms};record['parts'].append(part);offset+=len(rgb)
      last=markers['latent_offset']==chunk.shape[1]-1
      if last:
       record['rgb_ready_epoch']=part['rgb_ready_epoch'];record['first_rgb_ready_epoch']=record['parts'][0]['rgb_ready_epoch']
       record['profile']={'gpu_ms':{'source_wait_gpu_ms':trace.events['before_source_wait'].elapsed_time(trace.events['after_source_wait']),'p2p_gpu_ms':trace.events['after_source_wait'].elapsed_time(trace.events['after_p2p']),'vae_gpu_ms':sum(native_times),'rgb_convert_gpu_ms':None},'definition':'sum of native decoder and unpatchify GPU spans; excludes full-chunk normalization/conv2, CPU waits and JPEG'}
      offer(jpeg_jobs,(raw,rgb,record,part,last),encode_thread)
  except BaseException as e:fail('decode',e)
  finally:
   try:offer(jpeg_jobs,None,encode_thread)
   except BaseException as e:fail('decode_shutdown',e)
 def publisher():
  deadline=None;waiting_since=None
  while True:
   with condition:
    condition.wait_for(lambda:frames or done.is_set(),timeout=1)
    if not frames:
     if done.is_set():break
     continue
    if deadline is None:
     if waiting_since is None:waiting_since=time.perf_counter()
     if len(frames)<4 and not done.is_set() and time.perf_counter()-waiting_since<.4:
      condition.wait(timeout=.01);continue
     deadline=time.perf_counter()+.0625;playback['start']=deadline
    jpeg,meta=frames.popleft()
   gap=deadline-time.perf_counter()
   if gap>0:time.sleep(gap)
   meta['published_epoch']=time.time();presentations.append(meta)
   with state.condition:state.latest=jpeg;state.latest_meta=meta;state.frames.append((jpeg,dict(meta)));state.phase='实时生成中';state.condition.notify_all()
   deadline=max(deadline+1/state.fps,time.perf_counter())
 write_thread=threading.Thread(target=writer,daemon=True);encode_thread=threading.Thread(target=encoder,daemon=True);decode_thread=threading.Thread(target=decoder,daemon=True)
 publish_thread=threading.Thread(target=publisher,daemon=True) if live else None
 for thread in [write_thread,encode_thread,decode_thread]:thread.start()
 if publish_thread:publish_thread.start()
 count=0;reason='user_stop'
 try:
  while not state.stop_event.is_set() and not state.close_event.is_set():
   if errors:raise RuntimeError(errors)
   if max_chunks is not None and count>=max_chunks:reason='fixed_bounded_test_complete';break
   start=runtime.stepper.cursor;sample=cam_provider(start)
   if sample is None:reason='route_complete' if route else 'user_stop';break
   camera,record=sample;entry=None
   if memory_enabled:
    entry,audit=bank.query(current_start=start,pose_xyz=record['camera_position'],yaw_degrees=record['camera_yaw_degrees'],model_time_seconds=record['model_time_seconds'])
    if entry is not None:
     change=adapter.recache(runtime.stepper.conditional,bank.load(entry));audit['recache']=change;bank.mark_applied(entry,model_time_seconds=record['model_time_seconds']);adapter.verify_tensor_hashes=False
    mem_events.append(audit)
   noise=torch.randn((1,3,48,runtime.height//16,runtime.width//16),generator=source_noise,dtype=torch.bfloat16).to(device)
   if start==0:noise[:,0]=runtime.i0
   record['input_noise_sha256']=hashlib.sha256(noise.cpu().view(torch.uint8).numpy().tobytes()).hexdigest();record['memory_enabled']=memory_enabled;record['memory_hit']=entry is not None
   chunk=runtime.stepper.step(noise,camera).detach();record['model_callback_epoch']=time.time();record['first_frame']=0 if start==0 else 4*start-3
   event=torch.cuda.Event();event.record(torch.cuda.current_stream(device));raw=chunk.cpu()
   if memory_enabled:bank.add(raw,start,camera=camera,pose_xyz=record['camera_position'],yaw_degrees=record['camera_yaw_degrees'],model_time_seconds=record['model_time_seconds'])
   offer(jobs,(chunk,raw,record,event),decode_thread);count+=1
   with state.condition:state.memory_status=dict(bank.status(),enabled=memory_enabled,status='候选2D视觉记忆；重访一致性待审核')
 except BaseException as e:fail('model',e);reason='implementation_failure'
 finally:
  try:offer(jobs,None,decode_thread)
  except BaseException as e:fail('model_shutdown',e)
  decode_thread.join();encode_thread.join()
  if publish_thread:publish_thread.join()
  try:offer(archive,None,write_thread)
  except BaseException as e:fail('archive_shutdown',e)
  write_thread.join()
  write(out/'CHUNKS.json',records);write(out/'PRESENTATIONS.json',presentations);write(out/'MEMORY_EVENTS.json',mem_events);write(out/'CLIENT_METRICS.json',state.client_metrics)
  runtime.hooks.save_audit(out/'MAP_HOOK_AUDIT.json')
  report={'reason':reason,'errors':errors,'native_rgb_frames':sum(r['pixel_count'] for r in records),'native_latent_chunks':len(records),'target_fps':state.fps,'native_width':runtime.width,'native_height':runtime.height,'memory_enabled':memory_enabled,'memory_status':bank.status(),'seconds':time.perf_counter()-ticks,'live_input':live,'fixed_route_diagnostic':not live,'map_revision_id':state.map_revision_id,'map_metadata':runtime.map_metadata,'raw_chunk_archive':True,'native_weights_sampler_codec_unchanged':True,'old_frame_playback':False,'frame_interpolation':False,'human_return_identity_acceptance':False,'pipeline_version':3,'native_part_contract':pixel_parts.contract,'JPEG_on_separate_thread':True,'control_lead_seconds':.85}
  write(out/'COMPLETE.json',report)
 return report
