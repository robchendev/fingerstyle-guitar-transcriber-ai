from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import transcribe_video as launcher
from scripts.dataset_io import ROOT, publish_json, read_json
from scripts.transcriber_audio import HarnessError


class VideoLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="launcher-test-", dir=ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(patch.object(launcher, "ROOT", self.root))
        self.video = self.root / "player's performance.mkv"
        self.video.write_bytes(b"synthetic input; no model inference")
        self.output = self.root / "runs" / "output folder"
        asset = self.root / "asset"
        asset.write_bytes(b"synthetic model/template")
        self.settings = self.root / "settings.json"
        publish_json(self.settings, {
            "schemaVersion": 1, "checkpoint": "asset", "template": "asset",
            "beatCheckpoint": "asset", "device": "cpu", "beatDevice": "cpu", "pluckingScreenSide": "left",
            "exportProfile": {"strict-note-confidence": True, "rhythm-policy": "fingerstyle", "brush-threshold": .995},
        })
        self.args = [
            "--video", str(self.video), "--note-cutoff", ".78", "--x-cutoff", ".1",
            "--output-directory", str(self.output), "--settings", str(self.settings),
            "--tuning", "D2", "G2", "D3", "F#3", "A3", "D4", "--capo", "0",
            "--bpm", "70", "--beat-unit", "1/4", "--time-signature", "4/4",
            "--first-full-bar-downbeat", "1.25",
        ]
        self.stdout = self.enterContext(patch("sys.stdout", new=StringIO()))
        self.stderr = self.enterContext(patch("sys.stderr", new=StringIO()))
        self.pipeline = self.enterContext(patch.object(launcher.transcription_pipeline, "run_transcription", return_value={
            "status": "ready", "outputs": {"singleVoice": str(self.output / "single.gp"), "fullVoices": str(self.output / "full.gp")},
            "report": str(self.output / "transcription.json"), "actions": [],
        }))
        self.enterContext(patch("os.startfile", create=True, side_effect=AssertionError("Do not open GP")))

    def test_inline_metadata_and_cutoffs_reach_shared_pipeline_without_sidecar(self):
        self.assertEqual(launcher.main(self.args), 0)
        args = self.pipeline.call_args.args[0]
        self.assertEqual(args.video, str(self.video))
        self.assertEqual(args.output_directory, str(self.output))
        self.assertEqual(args.draft_note_threshold, .78)
        self.assertEqual(args.thumb_slap_threshold, .1)
        self.assertTrue(args.strict_note_confidence)
        self.assertEqual(args.rhythm_policy, "fingerstyle")
        self.assertEqual(args.checkpoint, str(self.root / "asset"))
        metadata = read_json(args.metadata)
        self.assertEqual(metadata["openStringMidi"], [38, 43, 50, 54, 57, 62])
        self.assertEqual(metadata["capoFret"], 0)
        self.assertEqual(metadata["tempo"], {"bpm": 70, "beatUnit": [1, 4]})
        self.assertEqual(metadata["timeSignature"], [4, 4])
        self.assertEqual(metadata["firstFullBarDownbeatSeconds"], 1.25)
        self.assertFalse(self.video.with_suffix(".metadata.json").exists())
        self.assertFalse(Path(args.metadata).is_relative_to(self.output))
        self.assertIn("Transcription complete: both GP files saved.", self.stdout.getvalue())
        self.assertEqual(self.video.read_bytes(), b"synthetic input; no model inference")

    def test_repeat_reuses_metadata_and_changed_settings_preserve_prior_inputs(self):
        self.assertEqual(launcher.main(self.args), 0)
        path = Path(self.pipeline.call_args.args[0].metadata)
        before = path.read_bytes(), path.stat().st_mtime_ns
        self.assertEqual(launcher.main(self.args), 0)
        self.assertEqual(Path(self.pipeline.call_args.args[0].metadata), path)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        self.assertEqual(launcher.main([*self.args, "--capo", "2", "--plucking-screen-side", "right"]), 0)
        self.assertNotEqual(Path(self.pipeline.call_args.args[0].metadata), path)
        self.assertEqual(self.pipeline.call_args.args[0].plucking_screen_side, "right")
        self.assertEqual(path.read_bytes(), before[0])
        publish_json(path, {"changed": True})
        self.pipeline.reset_mock()
        self.assertEqual(launcher.main(self.args), 1)
        self.pipeline.assert_not_called()
        self.assertEqual(read_json(path), {"changed": True})
        self.assertIn("refusing to overwrite", self.stderr.getvalue())

    def test_public_model_default_and_audio_only_work_without_local_settings(self):
        model = self.root / "models" / "transcriber.pt"
        model.parent.mkdir()
        model.write_bytes(b"synthetic model")
        args = list(self.args)
        index = args.index("--settings")
        del args[index:index + 2]
        args[args.index("--video")] = "--audio"
        args.extend(["--template", str(self.root / "asset"), "--beat-checkpoint", str(self.root / "asset")])
        self.assertEqual(launcher.main(args), 0)
        parsed = self.pipeline.call_args.args[0]
        self.assertEqual(parsed.audio, str(self.video))
        self.assertIsNone(parsed.video)
        self.assertEqual(parsed.checkpoint, str(model))
        self.assertEqual(parsed.plucking_screen_side, "geometry")
        self.assertEqual(parsed.device, "cpu")

    def test_explicit_assets_override_preset_and_missing_assets_are_actionable(self):
        replacement = self.root / "replacement"
        replacement.write_bytes(b"synthetic replacement")
        flags = ("--checkpoint", "--template", "--beat-checkpoint")
        extras = [value for flag in flags for value in (flag, str(replacement))]
        self.assertEqual(launcher.main([*self.args, *extras, "--device", "cpu"]), 0)
        parsed = self.pipeline.call_args.args[0]
        for flag in flags:
            self.assertEqual(getattr(parsed, flag[2:].replace("-", "_")), str(replacement))
        args = list(self.args)
        index = args.index("--settings")
        del args[index:index + 2]
        for extra, message in (([], "--template"), (["--template", str(replacement)], "--beat-checkpoint")):
            self.pipeline.reset_mock()
            self.assertEqual(launcher.main([*args, *extra]), 1)
            self.pipeline.assert_not_called()
            self.assertIn(message, self.stderr.getvalue())

    def test_invalid_metadata_and_preset_overrides_stop_before_dispatch(self):
        for extra in (["--capo", "25"], ["--bpm", "nan"], ["--bpm", "0"], ["--time-signature", "4/3"], ["--beat-unit", "1/16"],
                      ["--output-directory", str(self.root)], ["--video", str(self.root / "missing.mp4")]):
            with self.subTest(extra=extra):
                self.assertEqual(launcher.main([*self.args, *extra]), 1)
                self.pipeline.assert_not_called()
                self.assertFalse(self.output.exists())
        settings = read_json(self.settings)
        settings["exportProfile"]["draft-note-threshold"] = .99
        publish_json(self.settings, settings)
        self.assertEqual(launcher.main(self.args), 1)
        self.pipeline.assert_not_called()
        self.assertFalse((self.root / "runs" / "transcription-metadata").exists())

    def test_required_musical_fields_and_invalid_syntax_fail_without_files(self):
        for name, count in (("--tuning", 6), ("--capo", 1), ("--bpm", 1), ("--beat-unit", 1), ("--time-signature", 1), ("--first-full-bar-downbeat", 1)):
            args = list(self.args)
            index = args.index(name)
            del args[index:index + count + 1]
            with self.subTest(missing=name), self.assertRaises(SystemExit) as error:
                launcher.main(args)
            self.assertEqual(error.exception.code, 2)
        for extra in (["--note-cutoff", "nan"], ["--x-cutoff", "1.1"], ["--beat-unit", "1/0"], ["--first-full-bar-downbeat", "-1"],
                      ["--tuning", "D", "G2", "D3", "F#3", "A3", "D4"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit) as error:
                launcher.main([*self.args, *extra])
            self.assertEqual(error.exception.code, 2)
        self.pipeline.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_failures_and_review_actions_never_report_completion(self):
        self.pipeline.side_effect = HarnessError("synthetic model failure")
        self.assertEqual(launcher.main(self.args), 1)
        self.assertIn("Transcription failed: synthetic model failure", self.stderr.getvalue())
        self.pipeline.side_effect = None
        self.pipeline.return_value = {"status": "needs-review", "actions": [{"reason": "Review source alignment"}]}
        self.assertEqual(launcher.main(self.args), 1)
        self.assertIn("needs-review; action required", self.stdout.getvalue())
        self.assertIn("Review source alignment", self.stdout.getvalue())
        self.assertNotIn("Transcription complete:", self.stdout.getvalue())
        self.assertNotIn("Single-voice GP:", self.stdout.getvalue())

    def test_real_python_dry_run_resolves_caller_paths_without_writing_files(self):
        settings = read_json(self.settings)
        for key in ("checkpoint", "template", "beatCheckpoint"):
            settings[key] = str(self.root / "asset")
        publish_json(self.settings, settings)
        args = list(self.args)
        for flag, path in (("--video", self.video), ("--settings", self.settings), ("--output-directory", self.output)):
            args[args.index(flag) + 1] = str(path.relative_to(self.root))
        result = subprocess.run([sys.executable, str(ROOT / "transcribe_video.py"), *args, "--dry-run"],
                                cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        preview = json.loads(result.stdout[result.stdout.index("{"):])
        values = preview["arguments"]
        self.assertEqual(values[values.index("--video") + 1], str(self.video))
        self.assertEqual(values[values.index("--output-directory") + 1], str(self.output))
        self.assertEqual(preview["metadata"]["openStringMidi"], [38, 43, 50, 54, 57, 62])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.video.with_suffix(".metadata.json").exists())
        self.assertIn("Preview only: no files written or conversion started.", result.stdout)


if __name__ == "__main__":
    unittest.main()
