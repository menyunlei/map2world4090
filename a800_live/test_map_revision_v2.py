"""CPU transaction tests; fake codec objects do not claim native-GPU parity."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

import map_revision_v2 as revision
from live_service_v5 import State, make_handler


SIZE = (1080, 2280)


def png(image):
    out = io.BytesIO()
    image.save(out, format='PNG')
    return out.getvalue()


def data_url(raw):
    return {'png_data_url': 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')}


class FakeToken:
    def __init__(self, name): self.name = name
    def detach(self): return self


class FakeVAE:
    def __init__(self):
        self.moves = []
        self.clears = 0
        self.model = SimpleNamespace(clear_cache=self.clear)
    def clear(self): self.clears += 1
    def to(self, device): self.moves.append(device); return self


class FakeHooks:
    def __init__(self):
        self.tokens = FakeToken('original')
        self.metadata = {'tokens_sha256': 'old-token-hash'}
        self.expected_blocks = 30
        self.cache = {'is_init': True}
        self.pending_contexts = {0: 'old-context'}
        self.records = {str(i): {'calls': ['old-call'], 'cache_prefix_verified': True}
                        for i in range(30)}
    def reset_cross_attention(self): self.cache['is_init'] = False


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='map-revision-cpu-')
        self.root = Path(self.tmp.name)
        (self.root / 'inputs').mkdir()
        (self.root / 'audit').mkdir()
        self.original = png(Image.new('RGB', SIZE, (180, 190, 160)))
        self.original_path = self.root / 'inputs' / 'map.png'
        self.original_path.write_bytes(self.original)
        (self.root / 'inputs' / 'input.json').write_text(json.dumps({
            'map_path': str(self.original_path),
            'image_path': str(self.original_path),
            'caption': 'unchanged text'}))
        self.state = State(self.root)
        self.state.ready = True
        self.state.phase = 'ready'
        self.state.map_tokens_sha256 = 'old-token-hash'
        edited = Image.open(io.BytesIO(self.original)).convert('RGB')
        ImageDraw.Draw(edited).rectangle((50, 60, 200, 300), fill=(240, 180, 110))
        self.edited = png(edited)
        self.pipeline = SimpleNamespace(vae=FakeVAE(),
            generator=SimpleNamespace(model=SimpleNamespace(patch_embedding='unchanged-patch')))
        self.hooks = FakeHooks()

    def tearDown(self): self.tmp.cleanup()

    def visible(self):
        return (self.state.ready, self.state.running, self.state.closed,
                self.state.map_changing, self.state.map_revision_id,
                self.state.map_sha256, self.state.map_png, self.state.phase)

    def async_request(self, upload=None):
        result = {}
        def run():
            try: result['value'] = self.state.post('/map', data_url(upload or self.edited))
            except Exception as error: result['error'] = error
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        with self.state.condition:
            queued = self.state.condition.wait_for(lambda: self.state.map_job is not None, timeout=3)
        self.assertTrue(queued, result)
        return worker, result

    def endpoint(self, path):
        handler = object.__new__(make_handler(self.state))
        handler.path = path
        captured = {}
        def respond(code, body=b'', content_type='application/json', headers=None):
            captured.update(code=code, body=body, content_type=content_type, headers=headers)
        handler.respond = respond
        handler.do_GET()
        return captured

    def test_wrong_dimensions_and_non_png_rejected(self):
        self.assertTrue(revision.canonical_png(png(Image.new('RGB', (640, 384))), SIZE))
        for size in [(31,64),(4097,32),(3000,3000)]:
            with self.assertRaises(ValueError):revision.canonical_png(png(Image.new('RGB',size)),SIZE)
        with self.assertRaises(ValueError):
            revision.canonical_png(b'not a PNG', SIZE)

    def test_partial_rgba_alpha_rejected(self):
        im = Image.new('RGBA', SIZE, (10, 20, 30, 255))
        im.putpixel((50, 50), (10, 20, 30, 128))
        with self.assertRaises(ValueError): revision.canonical_png(png(im), SIZE)

    def test_palette_transparency_rejected(self):
        im = Image.new('P', SIZE, 0)
        im.putpalette([10, 20, 30] + [0] * 765)
        im.info['transparency'] = 0
        with self.assertRaises(ValueError): revision.canonical_png(png(im), SIZE)

    def test_busy_request_changes_no_state_or_files(self):
        self.state.running = True
        before = self.visible()
        files = set(self.root.rglob('*'))
        with self.assertRaises(ValueError): self.state.post('/map', data_url(self.edited))
        self.assertEqual(before, self.visible())
        self.assertEqual(files, set(self.root.rglob('*')))
        self.assertIsNone(self.state.map_job)

    def test_reservation_blocks_concurrent_start_during_validation(self):
        entered = threading.Event()
        released = threading.Event()
        result = {}
        def delayed_validation(raw, size):
            entered.set()
            if not released.wait(3): raise RuntimeError('test release timeout')
            raise ValueError('intentional invalid upload')
        def run():
            try: self.state.post('/map', data_url(self.edited))
            except Exception as error: result['error'] = error
        with patch.object(revision, 'canonical_png', delayed_validation):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            self.assertTrue(entered.wait(3))
            try:
                self.assertFalse(self.state.ready)
                self.assertTrue(self.state.map_changing)
                with self.assertRaises(ValueError): self.state.post('/start', {})
                self.assertFalse(self.state.running)
            finally: released.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(result.get('error'), ValueError)
        self.assertTrue(self.state.ready)
        self.assertFalse(self.state.map_changing)
        self.assertEqual(self.state.map_revision_id, 'original')
        self.assertFalse(self.state.close_event.is_set())

    def test_valid_commit_unblocks_waiter_and_get_is_exact_active_map(self):
        before_png = self.state.map_png
        worker, result = self.async_request()
        job = self.state.map_job
        self.assertEqual(self.state.map_png, before_png)
        self.assertFalse(self.state.ready)
        with self.assertRaises(ValueError): self.state.post('/start', {})
        with self.assertRaises(ValueError): self.state.post('/close', {})
        calls = []
        token = FakeToken('edited')
        def encoder(vae, patch_embedding, path):
            calls.append((threading.get_ident(), vae, patch_embedding, path.read_bytes()))
            self.assertEqual(path.read_bytes(), self.edited)
            return token, {'tokens_sha256': 'new-token-hash', 'token_count': 16}
        revision.apply_job(self.state, self.pipeline, self.hooks, 'dit-device', 'decoder-device', encoder)
        worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertNotIn('error', result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], threading.get_ident())
        self.assertEqual(self.pipeline.vae.moves, ['dit-device', 'decoder-device'])
        self.assertEqual(self.pipeline.vae.clears, 2)
        self.assertIs(self.hooks.tokens, token)
        self.assertFalse(self.hooks.cache['is_init'])
        self.assertEqual(self.hooks.pending_contexts, {})
        self.assertTrue(all(r == {'calls': [], 'cache_prefix_verified': False}
                            for r in self.hooks.records.values()))
        self.assertEqual(self.state.map_revision_id, job['map_revision_id'])
        self.assertEqual(self.state.map_tokens_sha256, 'new-token-hash')
        self.assertTrue(self.state.ready)
        self.assertFalse(self.state.map_changing)
        self.assertIsNone(self.state.map_job)
        active = self.endpoint('/map.png')
        self.assertEqual((active['code'], active['content_type']), (200, 'image/png'))
        self.assertEqual(active['body'], self.edited)
        self.assertEqual(hashlib.sha256(active['body']).hexdigest(), result['value']['map_sha256'])
        self.assertEqual(self.endpoint('/map-original.png')['body'], before_png)
        applied = json.loads((job['path'].parent / 'APPLIED.json').read_text())
        self.assertEqual(applied['parent_revision'], 'original')
        self.assertEqual(applied['source_size'], list(SIZE))
        self.assertEqual(applied['map_encoding_resolution'], [640, 384])
        self.assertFalse(applied['spatial_fidelity_acceptance'])
        self.assertTrue(applied['initial_image_and_text_unchanged'])
        started = self.state.post('/start', {})
        self.assertEqual(started['map_revision_id'], job['map_revision_id'])

    def test_entire_different_size_map_replacement(self):
        new_size=(640,512)
        replacement=png(Image.new('RGB',new_size,(90,150,210)))
        worker,result=self.async_request(replacement)
        job=self.state.map_job
        def encoder(*args):return FakeToken('replacement'),{'tokens_sha256':'replacement-token-hash','token_count':16}
        revision.apply_job(self.state,self.pipeline,self.hooks,'dit-device','decoder-device',encoder)
        worker.join(3)
        self.assertNotIn('error',result)
        self.assertEqual(self.state.map_size,new_size)
        self.assertEqual(self.state.map_png,replacement)
        self.assertEqual(self.state.map_record['source_size'],list(new_size))
        self.assertEqual(self.state.map_record['map_encoding_resolution'],[640,384])
        self.assertEqual(self.endpoint('/map-original.png')['body'],self.original)

    def test_encoder_failure_rolls_back_without_owner_shutdown(self):
        before = (self.state.map_png, self.state.map_sha256, self.state.map_revision_id)
        old_token = self.hooks.tokens
        old_metadata = dict(self.hooks.metadata)
        worker, result = self.async_request()
        job = self.state.map_job
        def encoder(*args): raise ValueError('fake deterministic encoder failure')
        revision.apply_job(self.state, self.pipeline, self.hooks, 'dit-device', 'decoder-device', encoder)
        worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(result.get('error'), ValueError)
        self.assertEqual(before, (self.state.map_png, self.state.map_sha256, self.state.map_revision_id))
        self.assertIs(self.hooks.tokens, old_token)
        self.assertEqual(self.hooks.metadata, old_metadata)
        self.assertEqual(self.pipeline.vae.moves, ['dit-device', 'decoder-device'])
        self.assertTrue(self.state.ready)
        self.assertFalse(self.state.map_changing)
        self.assertFalse(self.state.close_event.is_set())
        self.assertTrue((job['path'].parent / 'FAILED.json').exists())
        self.assertFalse((job['path'].parent / 'APPLIED.json').exists())

    def test_unchanged_upload_does_not_enqueue_or_encode(self):
        before = self.visible()
        result = self.state.post('/map', data_url(self.original))
        self.assertTrue(result['unchanged'])
        self.assertEqual(result['map_revision_id'], 'original')
        self.assertEqual(before[:7], self.visible()[:7])
        self.assertIsNone(self.state.map_job)
        self.assertFalse(self.state.start_event.is_set())

    def test_invalid_base64_keeps_old_map_and_owner_ready(self):
        before = (self.state.map_png, self.state.map_sha256, self.state.map_revision_id)
        with self.assertRaises(ValueError):
            self.state.post('/map', {'png_data_url': 'data:image/png;base64,!!!!'})
        self.assertEqual(before, (self.state.map_png, self.state.map_sha256, self.state.map_revision_id))
        self.assertTrue(self.state.ready)
        self.assertFalse(self.state.map_changing)
        self.assertFalse(self.state.close_event.is_set())

    def test_handler_negative_or_oversize_content_length_rejected_before_read(self):
        for length in ('-1', str(16 * 1024 * 1024 + 1)):
            handler = object.__new__(make_handler(self.state))
            handler.path = '/map'
            handler.headers = {'Content-Length': length}
            class MustNotRead:
                def read(self, size): raise AssertionError('oversized request was read')
            handler.rfile = MustNotRead()
            captured = {}
            handler.respond = lambda code, body=b'', *a, **kw: captured.update(code=code, body=body)
            handler.do_POST()
            self.assertEqual(captured['code'], 413)
            self.assertTrue(handler.close_connection)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RevisionTests)
    began = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    directory = Path(__file__).resolve().parent
    report = {
        'tests_run': result.testsRun,
        'passed': result.wasSuccessful(),
        'failures': [{'test': str(test), 'traceback': trace} for test, trace in result.failures],
        'errors': [{'test': str(test), 'traceback': trace} for test, trace in result.errors],
        'elapsed_seconds': time.perf_counter() - began,
        'source_sha256': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                          for name in ('map_revision_v2.py', 'live_service_v5.py', 'test_map_revision_v2.py')},
        'scope': 'CPU upload validation, queue reservation, transaction commit/rollback, handler map bytes',
        'native_gpu_parity_claimed': False,
        'HTTP_wire_server_started': False,
    }
    (directory / 'MAP_REVISION_CPU_TEST_V2.json').write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
