"""Prompt selection and idle-only settings validation, independent of CUDA."""
import hashlib


def prompt_record(presets, preset, caption=None):
    if preset != 'custom' and preset not in presets:
        raise ValueError('Unknown prompt preset')
    text = presets[preset]['caption'] if caption is None and preset != 'custom' else caption
    if not isinstance(text, str) or not text.strip() or len(text) > 3000 or '\x00' in text:
        raise ValueError('描述应为1–3000字符的有效文本')
    text = text.strip()
    if preset != 'custom' and text != presets[preset]['caption']:
        preset = 'custom'
    label = presets[preset]['label'] if preset != 'custom' else '自定义描述'
    return {'preset':preset,'label':label,'caption':text,
            'caption_sha256':hashlib.sha256(text.encode()).hexdigest()}


def settings_job(state, data, presets):
    if not state.ready or state.running or state.closed or state.close_event.is_set() or state.map_changing:
        raise ValueError('请停止生成并等待保存完成')
    if set(data) - {'quality','memory_enabled','prompt_preset','caption'}:
        raise ValueError('Unknown settings')
    quality = data.get('quality',state.quality)
    if quality not in ['native640','native768']:
        raise ValueError('Unsupported native resolution')
    memory = data.get('memory_enabled',state.memory_enabled)
    if type(memory) is not bool:
        raise ValueError('memory_enabled must be boolean')
    prompt = state.prompt
    if 'prompt_preset' in data or 'caption' in data:
        preset = data.get('prompt_preset','custom')
        prompt = prompt_record(presets,preset,data.get('caption'))
    return {'quality':quality,'memory_enabled':memory,'prompt':prompt}
