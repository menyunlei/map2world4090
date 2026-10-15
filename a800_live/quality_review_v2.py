"""Fixed prompt comparison audit and application-native quality review page."""
import hashlib,html,json,re,subprocess,zipfile
from pathlib import Path

LABELS=['original_caption','neutral_scene','detailed_scene']


def validate_and_package(code,presets):
    import cv2,torch,imageio_ffmpeg
    from PIL import Image
    root=code/'quality_comparison';reports={};all_records={};payloads=[]
    for label in LABELS:
        out=root/label;complete=json.loads((out/'COMPLETE.json').read_text());assert not complete['errors']
        records=json.loads((out/'CHUNKS.json').read_text());all_records[label]=records;count=0;raw_hashes={}
        for index,record in enumerate(records):
            path=out/'chunks'/f"{record['latent_start']:08d}.pt";data=torch.load(path,map_location='cpu',weights_only=True)
            rgb=data['rgb_uint8'];assert torch.isfinite(data['latent']).all();assert data['record']==record
            assert hashlib.sha256(rgb.numpy().tobytes()).hexdigest()==record['native_rgb_sha256']
            count+=len(rgb);raw_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            if index in [0,8,16,23]:Image.fromarray(rgb[-4].numpy()).save(out/f'anchor_{index:02d}.png')
        assert count==285==complete['native_rgb_frames']
        cap=cv2.VideoCapture(str(out/'native.mp4'));assert cap.isOpened();decoded=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(complete['native_height'],complete['native_width'],3);decoded+=1
        assert decoded==count and cap.get(cv2.CAP_PROP_FPS)==16;cap.release()
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-i',str(out/'native.mp4'),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        prompt=json.loads((out/'PROMPT.json').read_text());assert prompt['caption']==presets[label]['caption']
        reports[label]={'frames':count,'native_dimensions':[complete['native_width'],complete['native_height']],'fps':16,'full_OpenCV_and_FFmpeg_decode':True,'raw_chunks_sha256':raw_hashes,'prompt':prompt,'memory_enabled':False,'visual_acceptance':False}
    reference=all_records[LABELS[0]]
    for label in LABELS[1:]:
        assert len(all_records[label])==len(reference)
        for a,b in zip(reference,all_records[label]):
            assert a['input_noise_sha256']==b['input_noise_sha256']
            assert a['global_c2w']==b['global_c2w']
    report={'all_arms_complete':True,'same_noise_and_cameras':True,'arms':reports,'default_preset_predeclared':'neutral_scene','human_visual_acceptance':False,'improvement_not_established_by_hashes':True}
    (root/'REVIEW_V2.json').write_text(json.dumps(report,indent=2))
    # Native fixed-time contact plate; no generated image editing or replacement.
    first=reports[LABELS[0]];w,h=first['native_dimensions'];plate=Image.new('RGB',(3*w,4*(h+28)),(15,22,32))
    from PIL import ImageDraw
    draw=ImageDraw.Draw(plate)
    for row,index in enumerate([0,8,16,23]):
        for col,label in enumerate(LABELS):
            image=Image.open(root/label/f'anchor_{index:02d}.png');assert image.size==(w,h)
            plate.paste(image,(col*w,row*(h+28)+28));draw.text((col*w+8,row*(h+28)+5),f'{label} / chunk {index}',fill='white')
    plate.save(root/'fixed_native_comparison.jpg',quality=95)
    package=root/'REVIEW_PACKAGE_V2.zip'
    for path in root.rglob('*'):
        if path.is_file() and 'chunks' not in path.parts and 'spatial_memory' not in path.parts and path.suffix in ['.json','.png','.jpg','.mp4']:
            payloads.append((path,path.relative_to(root).as_posix()))
    manifest={name:hashlib.sha256(path.read_bytes()).hexdigest() for path,name in payloads}
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
        for path,name in payloads:archive.write(path,name)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    (root/'PACKAGE.json').write_text(json.dumps({'bytes':package.stat().st_size,'sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'members':len(manifest)},indent=2))


def serve_review(handler,req,code):
    root=code/'quality_comparison'
    if req.path=='/quality-review':
        ready=(root/'REVIEW_V2.json').exists();cards=''
        presets=json.loads((code/'PROMPT_PRESETS_V2.json').read_text())
        for label in LABELS:
            prompt=presets[label]
            cards+=f'<section><h2>{html.escape(prompt["label"])}</h2><p>{html.escape(prompt["caption"])}</p>'
            if (root/label/'COMPLETE.json').exists():cards+=f'<video controls preload="metadata" src="/quality-assets/{label}/native.mp4"></video>'
            else:cards+='<p>尚在生成，请稍后刷新。</p>'
            cards+='</section>'
        page=f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同条件Prompt画质对照</title><style>body{{font:16px/1.6 system-ui;background:#101722;color:#edf3fa;padding:24px;max-width:1100px;margin:auto}}section{{background:#1a2738;border-radius:12px;padding:18px;margin:20px 0}}video,img{{width:100%}}a{{color:#8acaff}}</style><h1>同条件Prompt画质对照</h1><p><a href="/">返回实时试玩</a></p><p>相同模型、地图、参考图、相机轨迹、噪声、4步采样及原生分辨率；每组285帧、16fps、17.8125秒。三组均关闭检索记忆，单独检查Prompt效果。</p><p>审核完成：{ready}。画质提升仍待人工比较；默认通用街景版在生成前已确定，没有挑seed或挑选最好片段。</p>{cards}<h2>固定时间原生帧</h2><img src="/quality-assets/fixed_native_comparison.jpg">'''
        handler.respond(200,page.encode(),'text/html; charset=utf-8');return True
    if not req.path.startswith('/quality-assets/'):return False
    name=req.path[len('/quality-assets/'):]
    allowed={'fixed_native_comparison.jpg','REVIEW_V2.json'}|{f'{label}/native.mp4' for label in LABELS}
    if name not in allowed or not (root/name).exists():handler.respond(404,b'{"error":"asset unavailable"}');return True
    path=root/name;mime='video/mp4' if path.suffix=='.mp4' else 'image/jpeg' if path.suffix=='.jpg' else 'application/json'
    size=path.stat().st_size;requested=handler.headers.get('Range')
    if requested:
        match=re.fullmatch(r'bytes=(\d+)-(\d*)',requested)
        if not match:handler.respond(416,b'',headers={'Content-Range':f'bytes */{size}'});return True
        start=int(match[1]);end=min(size-1,int(match[2]) if match[2] else size-1)
        if start>end:handler.respond(416,b'',headers={'Content-Range':f'bytes */{size}'});return True
        with path.open('rb') as stream:stream.seek(start);data=stream.read(end-start+1)
        handler.respond(206,data,mime,{'Accept-Ranges':'bytes','Content-Range':f'bytes {start}-{end}/{size}'})
    else:handler.respond(200,path.read_bytes(),mime,{'Accept-Ranges':'bytes'})
    return True
