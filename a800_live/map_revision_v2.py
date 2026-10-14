"""Bounded user map revisions; GPU encoding is owned by the native worker."""
import base64
import hashlib
import io
import json
from pathlib import Path
import threading
import time


def _sha(data):return hashlib.sha256(data).hexdigest()


def _save(path,data):
    path=Path(path);tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False));tmp.replace(path)


def canonical_png(data,expected_size):
    from PIL import Image
    if len(data)>12*1024*1024 or not data.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('地图必须是有效PNG，文件不超过12MiB')
    with Image.open(io.BytesIO(data)) as image:
        if image.format!='PNG' or getattr(image,'n_frames',1)!=1:
            raise ValueError('地图必须是单帧PNG')
        if image.width<32 or image.height<32:
            raise ValueError('地图宽高至少32像素')
        if image.width>4096 or image.height>4096 or image.width*image.height>8_000_000:
            raise ValueError('地图像素数超出限制')
        image.load()
        if image.convert('RGBA').getchannel('A').getextrema()!=(255,255):
            raise ValueError('地图不能含透明区域，请使用擦回底图')
        out=io.BytesIO();image.convert('RGB').save(out,format='PNG');return out.getvalue()


def initialize(state):
    from PIL import Image
    source=Path(state.inputs['map_path'])
    with Image.open(source) as im:
        state.map_size=im.size;out=io.BytesIO();im.convert('RGB').save(out,format='PNG')
    state.original_map_png=out.getvalue();state.map_png=state.original_map_png
    state.map_sha256=_sha(state.map_png);state.map_revision_id='original';state.map_tokens_sha256=None
    state.map_changing=False;state.map_error=None;state.map_job=None
    directory=state.root/'map_revisions_v1/original';directory.mkdir(parents=True,exist_ok=True)
    target=directory/'map.png'
    if target.exists():assert target.read_bytes()==state.original_map_png,'Original map pixels changed'
    else:target.write_bytes(state.original_map_png)
    state.map_path=source
    state.map_record={'map_revision_id':'original','map_sha256':state.map_sha256,
        'native_map_path':str(source),'original_file_sha256':_sha(source.read_bytes()),
        'canonical_png_path':str(target),'source_size':list(state.map_size),
        'map_encoding_resolution':[640,384],'spatial_fidelity_acceptance':False}


def request(state,data):
    if not isinstance(data,dict) or not isinstance(data.get('png_data_url'),str):
        raise ValueError('缺少PNG地图数据')
    with state.condition:
        if not state.ready or state.running or state.closed or state.close_event.is_set() or state.map_changing:
            raise ValueError('请停止本段并等待保存完成后应用地图')
        state.ready=False;state.map_changing=True;state.map_error=None;state.phase='校验地图…'
        state.last_touch=time.monotonic();state.condition.notify_all()
    try:
        prefix='data:image/png;base64,';value=data['png_data_url']
        if not value.startswith(prefix) or len(value)>16*1024*1024:
            raise ValueError('仅支持PNG地图，数据不超过16MiB')
        uploaded=base64.b64decode(value[len(prefix):],validate=True)
        png=canonical_png(uploaded,state.map_size);sha=_sha(png)
        from PIL import Image
        with Image.open(io.BytesIO(png)) as image:source_size=list(image.size)
        file_name=str(data.get('file_name') or '用户地图').replace(chr(92),'/').rsplit('/',1)[-1][:128]
        with state.condition:
            if sha==state.map_sha256:
                state.ready=True;state.map_changing=False;state.phase='地图未改变：点击开始'
                state.condition.notify_all()
                return {'map_revision_id':state.map_revision_id,'map_sha256':sha,
                    'tokens_sha256':state.map_tokens_sha256,'unchanged':True}
            revision=f'map_{time.time_ns()}_{sha[:10]}'
            directory=state.root/'map_revisions_v1'/revision;directory.mkdir()
            path=directory/'map.png';path.write_bytes(png)
            job={'map_revision_id':revision,'parent_revision':state.map_revision_id,
                'map_sha256':sha,'upload_sha256':_sha(uploaded),'path':path,'png':png,
                'source_size':source_size,'file_name':file_name,
                'event':threading.Event(),'requested_epoch':time.time(),'result':None,'error':None}
            state.map_job=job;state.phase='编码新地图…';state.start_event.set();state.condition.notify_all()
    except Exception as error:
        with state.condition:
            state.ready=True;state.map_changing=False;state.map_error=str(error);state.phase='地图无效：保留当前地图'
            state.condition.notify_all()
        _save(state.root/'audit'/f'MAP_REJECT_{time.time_ns()}.json',{'epoch':time.time(),
            'error':str(error),'active_revision_unchanged':state.map_revision_id,'native_compute_started':False})
        raise ValueError(str(error)) from error
    if not job['event'].wait(60):
        # The accepted job remains visible; do not requeue or claim failure.
        return {'pending':True,'request_id':revision,'message':'编码仍在进行，请查看地图状态'}
    if job['error']:raise ValueError(job['error'])
    return job['result']


def apply_job(state,pipeline,hooks,device,decoder_device,encoder):
    """Called only by idle GPU owner, after decoder/publisher threads joined."""
    with state.condition:job=state.map_job
    if job is None:return
    began=time.perf_counter();meta=None;restore_ok=False
    previous_tokens=hooks.tokens;previous_metadata=hooks.metadata
    try:
        try:
            pipeline.vae.model.clear_cache();pipeline.vae.to(device)
            tokens,meta=encoder(pipeline.vae,pipeline.generator.model.patch_embedding,job['path'])
        finally:
            pipeline.vae.to(decoder_device);pipeline.vae.model.clear_cache();restore_ok=True
        record={k:job[k] for k in ['map_revision_id','parent_revision','map_sha256','upload_sha256','requested_epoch']}
        record.update(native_map_path=str(job['path']),canonical_png_path=str(job['path']),
            source_size=job['source_size'],file_name=job['file_name'],map_encoding_resolution=[640,384],
            tokens_sha256=meta['tokens_sha256'],map_encoding_metadata=meta,
            encoding_seconds=time.perf_counter()-began,applied_epoch=time.time(),
            applies_at_next_explicit_session=True,initial_image_and_text_unchanged=True,
            spatial_fidelity_acceptance=False)
        # Reset all30 verification records so the next actual inference proves
        # new map K/V prefixes, rather than inheriting the old-map audit.
        hooks.reset_cross_attention();hooks.tokens=tokens.detach();hooks.metadata=dict(meta)
        hooks.pending_contexts.clear()
        hooks.records={str(i):{'calls':[],'cache_prefix_verified':False} for i in range(hooks.expected_blocks)}
        _save(job['path'].parent/'APPLIED.json',record)
        _save(state.root/'audit/MAP_ACTIVE_CURRENT_V1.json',record)
        with state.condition:
            state.map_png=job['png'];state.map_sha256=job['map_sha256'];state.map_tokens_sha256=meta['tokens_sha256']
            state.map_revision_id=job['map_revision_id'];state.map_record=record;state.map_path=job['path'];state.map_size=tuple(job['source_size'])
            state.frames.clear();state.latest=None;state.latest_meta={};state.path=[[0,0]]
            state.map_error=None;state.phase='地图已应用：点击开始新一段'
            job['result']={'map_revision_id':state.map_revision_id,'map_sha256':state.map_sha256,
                'tokens_sha256':state.map_tokens_sha256,'encoding_seconds':record['encoding_seconds']}
    except Exception as error:
        hooks.tokens=previous_tokens;hooks.metadata=previous_metadata
        job['error']=str(error)
        _save(job['path'].parent/'FAILED.json',{'epoch':time.time(),'error':repr(error),
            'active_revision':state.map_revision_id,'automatic_retry':False})
        with state.condition:
            state.map_error=str(error);state.phase='地图编码失败：保留当前地图'
            if not restore_ok:
                state.error=str(error);state.close_reason='map_encoder_restore_failure';state.close_event.set()
    finally:
        with state.condition:
            state.map_job=None;state.map_changing=False;state.ready=restore_ok and not state.close_event.is_set()
            state.last_touch=time.monotonic();job['event'].set();state.condition.notify_all()
