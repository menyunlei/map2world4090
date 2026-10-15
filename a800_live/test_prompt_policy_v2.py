import json,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from prompt_policy_v2 import prompt_record,settings_job


class PromptPolicyTests(unittest.TestCase):
    def setUp(self):
        self.presets=json.loads((Path(__file__).parent/'PROMPT_PRESETS_V2.json').read_text())
        self.state=SimpleNamespace(ready=True,running=False,closed=False,close_event=threading.Event(),map_changing=False,quality='native768',memory_enabled=True,prompt=prompt_record(self.presets,'neutral_scene'))
    def test_original_preserved_and_neutral_has_no_fixed_turn(self):
        self.assertIn('RIGHT turn',self.presets['original_caption']['caption'])
        for label in ['neutral_scene','detailed_scene']:
            self.assertNotIn('90 degrees',self.presets[label]['caption'])
            self.assertNotIn('RIGHT',self.presets[label]['caption'])
    def test_running_and_closing_refused(self):
        self.state.running=True
        with self.assertRaises(ValueError):settings_job(self.state,{'prompt_preset':'original_caption'},self.presets)
        self.state.running=False;self.state.close_event.set()
        with self.assertRaises(ValueError):settings_job(self.state,{},self.presets)
    def test_idle_update_preserves_other_settings_and_state(self):
        old=self.state.prompt.copy();job=settings_job(self.state,{'prompt_preset':'detailed_scene'},self.presets)
        self.assertEqual(job['quality'],'native768');self.assertTrue(job['memory_enabled']);self.assertEqual(self.state.prompt,old)
        self.assertNotEqual(job['prompt']['caption_sha256'],old['caption_sha256'])
    def test_custom_is_honestly_labelled_and_invalid_text_rejected(self):
        self.assertEqual(prompt_record(self.presets,'neutral_scene','A custom scene')['preset'],'custom')
        for value in ['',None,'x'*3001,'a\x00b']:
            with self.assertRaises(ValueError):prompt_record(self.presets,'custom',value)
        with self.assertRaises(ValueError):settings_job(self.state,{'memory_enabled':1},self.presets)
        with self.assertRaises(ValueError):settings_job(self.state,{'unknown':True},self.presets)


if __name__=='__main__':unittest.main()
