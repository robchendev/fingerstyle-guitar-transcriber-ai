from dataclasses import asdict
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import soundfile as sf
import torch

from scripts import transcriber
from scripts.dataset_io import ROOT, read_json, publish_json
from scripts.transcriber_audio import FeatureConfig, HarnessError


class HarnessCommandTests(unittest.TestCase):
    def test_default_manifest_and_explicit_overrides_always_use_training_loader(self):
        config, features, model, _ = transcriber.load_config()
        self.assertEqual(config["data"]["manifest"], "data\\releases\\dataset-v1\\manifest.json")
        for root, override, expected in (
            (ROOT, None, ROOT / "data" / "releases" / "dataset-v1" / "manifest.json"),
            (ROOT / "private-workspace", "releases\\v2\\manifest.json", ROOT / "private-workspace" / "releases" / "v2" / "manifest.json"),
            (ROOT, ROOT / "custom" / "manifest.json", ROOT / "custom" / "manifest.json"),
        ):
            with self.subTest(root=root, override=override), patch.object(transcriber, "TrainingDataset") as loader, patch.object(transcriber, "read_json", side_effect=AssertionError("No format dispatch")):
                transcriber.make_dataset(config, features, model, "train", root, override)
                loader.assert_called_once_with(expected, "train", features, model, root=root, cache_dir=root / "cache" / "transcriber")

    def test_config_path_roundtrip_and_worker_contract(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "config.json"
            config = transcriber.default_config()
            publish_json(path, config)
            loaded, features, model, training = transcriber.load_config(str(path))
            self.assertEqual(loaded, config)
            self.assertEqual(features.n_mels, model.n_mels)
            config["data"]["num_workers"] = 1
            publish_json(path, config)
            with self.assertRaisesRegex(HarnessError, "num_workers"):
                transcriber.load_config(str(path))

    def metadata(self):
        return {"openStringMidi": [40, 45, 50, 55, 59, 64], "capoFret": 0, "tempo": {"bpm": 100, "beatUnit": [1, 4]}, "timeSignature": [4, 4], "firstFullBarDownbeatSeconds": 0}

    def test_train_and_resume_require_no_acknowledgement_flag(self):
        from scripts.transcriber_video import VideoConfig

        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            config = transcriber.load_config()
            config[0]["video"] = {"index": "synthetic-index.json", "model": asdict(VideoConfig())}
            identity = {"model": config[0]["model"], "video": {"config": config[0]["video"]["model"]}}
            dataset = MagicMock(
                manifest_sha256="synthetic-manifest", records=[{}, {}],
                video_identity={"synthetic": True},
                video_coverage={"excludedWindows": 1, "excludedRecordings": 1, "audioFramesWithUsableVideo": 8, "preparedAudioFrames": 10, "trackingGapOnlyWindows": 0},
            )
            dataset.__len__.return_value = 4
            model, train_loader, validation_loader = object(), object(), object()
            for resume in (None, str(root / "runs" / "example" / "latest.pt")):
                summary = {
                    "elapsed_seconds": 2., "epoch": 2, "epochs_requested": 2, "epochs_completed_this_run": 2,
                    "global_step": 4, "training_steps_processed": 4, "stopped_by": "epochs",
                    "training_windows_processed": 8, "validation_windows_processed": 12,
                    "dataset_windows": {"train": 4, "validation": 4}, "validation": {"loss": .5}, "best_score": .4,
                    "latest_checkpoint": str(root / "runs" / "example" / "latest.pt"),
                    "best_checkpoint": str(root / "runs" / "example" / "best.pt"),
                }
                with self.subTest(resume=resume), patch.object(transcriber, "load_config", return_value=config), patch.object(transcriber, "seed_everything"), patch.object(transcriber.torch, "set_num_threads"), patch.object(transcriber, "make_dataset", return_value=dataset), patch.object(transcriber, "make_loader", side_effect=[train_loader, validation_loader]), patch.object(transcriber, "run_identity", return_value=identity), patch.object(transcriber, "_wrap_video_model", return_value=model), patch("scripts.transcriber_model.FingerstyleTranscriber", return_value=model), patch("scripts.transcriber_runtime.load_checkpoint", return_value={"identity": identity}) as load_checkpoint, patch("scripts.transcriber_runtime.run_training", return_value=summary) as run_training, patch.object(transcriber.time, "perf_counter", side_effect=[100., 103., 105.]), patch("sys.stdout", new=StringIO()) as output:
                    args = ["train", "--data-root", str(root), "--run-dir", "runs\\example"]
                    if resume:
                        args.extend(["--resume", resume])
                    self.assertEqual(transcriber.main(args), 0)
                    self.assertEqual(load_checkpoint.call_count, int(resume is not None))
                    run_training.assert_called_once_with(model, train_loader, validation_loader, config[3], root / "runs" / "example", identity, resume=resume, progress=transcriber.log_progress, event_evaluator=ANY, started_at=100., deadline=None)
                    self.assertTrue(callable(run_training.call_args.kwargs["event_evaluator"]))
                    text = output.getvalue()
                    self.assertIn("Loading training data", text)
                    self.assertIn("Loading validation data", text)
                    self.assertIn("Total elapsed: 00:00:05", text)
                    self.assertIn("Window visits this invocation: 8 training; 12 validation", text)
                    saved = read_json(root / "runs" / "example" / "summary.json")
                    self.assertEqual(saved["elapsed_seconds"], 5.)
                    self.assertEqual(saved["training_elapsed_seconds"], 2.)
                    self.assertEqual(saved["setup_seconds"], 3.)
                    self.assertEqual(saved["dataset_recordings"], {"train": 2, "validation": 2})

    def test_resume_rejects_historical_paired_checkpoint_before_dataset_loading(self):
        from scripts.transcriber_video import VideoConfig

        config = transcriber.load_config()
        config[0]["video"] = {"index": "synthetic-index.json", "model": asdict(VideoConfig())}
        args = SimpleNamespace(config=None, resume="historical.pt")
        with patch.object(transcriber, "load_config", return_value=config), patch("scripts.transcriber_runtime.load_checkpoint", side_effect=ValueError("schema-2 paired checkpoints are unsupported")), patch.object(transcriber, "make_dataset") as dataset, patch.object(transcriber, "seed_everything") as seed:
            with self.assertRaisesRegex(ValueError, "schema-2 paired checkpoints"):
                transcriber.train(args)
        dataset.assert_not_called()
        seed.assert_not_called()

    def test_fresh_audio_training_uses_base_model_without_video_or_imported_checkpoint(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            path = root / "audio-config.json"
            publish_json(path, transcriber.default_config())
            for config_args in ([], ["--config", str(path)]):
                dataset = MagicMock(manifest_sha256="synthetic", records=[{}])
                dataset.__len__.return_value = 1
                run_dir = f"runs\\audio-{len(config_args)}"
                with self.subTest(config_args=config_args), patch.object(transcriber, "make_dataset", return_value=dataset), patch.object(transcriber, "make_loader"), patch.object(transcriber, "run_identity", return_value={}), patch.object(transcriber, "seed_everything"), patch.object(transcriber, "print_training_summary"), patch.object(transcriber, "_wrap_video_model") as wrap, patch("scripts.transcriber_model.FingerstyleTranscriber") as model, patch("scripts.transcriber_runtime.load_checkpoint") as load, patch("scripts.transcriber_runtime.run_training", return_value={"elapsed_seconds": 0.}) as run:
                    result = transcriber.main(["train", "--data-root", str(root), "--run-dir", run_dir, *config_args])
                self.assertEqual(result, 0)
                model.assert_called_once()
                wrap.assert_not_called()
                load.assert_not_called()
                self.assertIs(run.call_args.args[0], model.return_value)
                self.assertIsNone(run.call_args.kwargs["resume"])
                self.assertEqual(read_json(root / run_dir / "summary.json")["trainingMode"], "audio-from-scratch")

    def test_audio_configuration_accepts_explicit_manifest_without_video_index(self):
        with TemporaryDirectory(dir=ROOT / "runs") as directory:
            path = Path(directory) / "config.json"
            manifest = "data\\releases\\custom\\manifest.json"
            self.assertEqual(transcriber.main(["config", "--manifest", manifest, "--output", str(path)]), 0)
            config = read_json(path)
            self.assertEqual(config["data"]["manifest"], manifest)
            self.assertNotIn("video", config)

    def test_progress_logs_flush_immediately(self):
        with patch("builtins.print") as output:
            transcriber.log_progress("Epoch progress")
        output.assert_called_once_with(ANY, flush=True)
        self.assertIn("Epoch progress", output.call_args.args[0])

    def test_preflight_reports_incremental_progress_and_never_success_on_scan_failure(self):
        config = transcriber.load_config()
        item = {"features": torch.zeros(2, config[1].n_mels), "metadata": {"stringFrameCollisionsMasked": 0}, "masks": {}}
        owner = self
        for fail in (False, True):
            with self.subTest(fail=fail), TemporaryDirectory(dir=ROOT / "runs") as directory:
                report = Path(directory) / "preflight.json"
                output = StringIO()

                class Dataset(list):
                    manifest_sha256 = "synthetic"

                    def __iter__(self):
                        owner.assertIn("scanning features, targets and masks", output.getvalue())
                        for index, row in enumerate(super().__iter__()):
                            if index == 1:
                                owner.assertIn("windows 1/23", output.getvalue())
                                if fail:
                                    raise HarnessError("synthetic scan failure")
                            yield row

                with patch.object(transcriber, "load_config", return_value=config), patch.object(transcriber, "make_dataset", return_value=Dataset([item] * 23)), patch("sys.stdout", new=output), patch("sys.stderr", new=StringIO()) as error, patch("torch.optim.AdamW", side_effect=AssertionError("Preflight cannot train")):
                    code = transcriber.main(["preflight", "--output", str(report)])
                if fail:
                    self.assertEqual(code, 1)
                    self.assertIn("synthetic scan failure", error.getvalue())
                    self.assertNotIn("scan completed", output.getvalue())
                    self.assertNotIn("report saved", output.getvalue())
                    self.assertFalse(report.exists())
                else:
                    self.assertEqual(code, 0)
                    for split in ("train", "validation"):
                        for count in (1, 10, 20, 23):
                            self.assertIn(f"Preflight {split}: windows {count}/23", output.getvalue())
                        self.assertEqual(read_json(report)["splits"][split]["windows"], 23)
                    self.assertIn("Preflight: report saved", output.getvalue())




    def test_event_evaluation_records_explicitly_selected_release_without_training(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            _, features, model_config, _ = transcriber.load_config()
            identity = {"features": asdict(features), "model": asdict(model_config), "manifest_sha256": "training-release"}
            dataset = SimpleNamespace(manifest_sha256="evaluation-release")
            model = object()
            with patch.object(transcriber, "checkpoint_model", return_value=(model, {"identity": identity}, torch.device("cpu"))), patch.object(transcriber, "make_dataset", return_value=dataset), patch.object(transcriber, "sha256", return_value="checkpoint-hash"), patch("scripts.transcriber_events.evaluate_events", return_value={"metricsByToleranceSeconds": {}, "trainingPerformed": False}) as evaluate, patch("sys.stdout", new=StringIO()):
                self.assertEqual(transcriber.main(["evaluate-events", "--data-root", str(root), "--manifest", "releases\\v2\\manifest.json", "--checkpoint", "synthetic.pt", "--output", "runs\\events.json"]), 0)
            evaluate.assert_called_once()
            report = read_json(root / "runs" / "events.json")
            self.assertEqual(report["trainingManifestSha256"], "training-release")
            self.assertEqual(report["manifestSha256"], "evaluation-release")
            self.assertFalse(report["sameReleaseAsTraining"])
            self.assertFalse(report["trainingPerformed"])

    def test_inference_metadata_never_defaults_missing_tuning_or_timing(self):
        value = self.metadata()
        tempos, meters = transcriber.inference_metadata(value)
        self.assertEqual(tempos[0]["bpm"], 100)
        self.assertEqual(meters[0]["timeSignature"], [4, 4])
        historical = dict(value)
        del historical["firstFullBarDownbeatSeconds"]
        transcriber.inference_metadata(historical, conditioning_dim=12)
        for field in value:
            invalid = dict(value)
            del invalid[field]
            with self.subTest(field=field), self.assertRaises(HarnessError):
                transcriber.inference_metadata(invalid)
        for change in ({"tempo": None}, {"tempoChanges": [{"bpm": 120}]}, {"timeSignatureChanges": None}, {"capoChanges": []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                transcriber.inference_metadata({**value, **change})

    def test_output_must_remain_in_private_runtime_directories(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            self.assertEqual(transcriber.private_output("runs/result.json", root), root / "runs" / "result.json")
            for path in ("data/output.json", "docs/output.json", "..\\public.json"):
                with self.subTest(path=path), self.assertRaises(HarnessError):
                    transcriber.private_output(path, root)

    def test_export_gp_cli_uses_shared_cleanup_defaults(self):
        parser = []
        with patch("scripts.gp_output.write_gp_outputs") as write, patch.object(transcriber, "read_json", return_value={"notes": [], "percussion": []}), patch.object(transcriber, "sha256", return_value="hash"), patch.object(transcriber, "publish_json"), patch.object(transcriber, "private_output", side_effect=lambda path, root=None: Path(path)), patch("sys.stdout", new=StringIO()):
            write.return_value = {"fullVoices": {"measureCount": 1}, "singleVoice": {}}
            parser = [
                "export-gp", "--predictions", "predictions.json", "--template", "template.gpt",
                "--full-output", "runs\\full.gp", "--single-output", "runs\\single.gp",
            ]
            self.assertEqual(transcriber.main(parser), 0)
        profile = write.call_args.kwargs["profile"]
        self.assertEqual(profile.note_threshold, .9)
        self.assertEqual(profile.percussion_threshold, .6)
        self.assertFalse(profile.include_harmonics)
        self.assertEqual(profile.connection_threshold, .8)
        self.assertEqual(profile.note_technique_threshold, .8)
        self.assertEqual(profile.chord_tolerance_seconds, .04)
        self.assertEqual(profile.same_string_gap_seconds, 0.)
        self.assertFalse(profile.strict_note_confidence)

    def test_export_accepts_strict_confidence_for_completed_chord_notes(self):
        args = transcriber.argument_parser().parse_args([
            "export-gp", "--predictions", "runs\\predictions.json", "--template", "template.gpt",
            "--draft-note-threshold", ".98", "--strict-note-confidence", "--brush-threshold", ".995",
        ])
        self.assertEqual(args.draft_note_threshold, .98)
        self.assertEqual(args.brush_threshold, .995)
        self.assertTrue(args.strict_note_confidence)

    def test_analyze_beats_cli_publishes_private_evidence(self):
        report = {"beatCount": 10, "downbeatCount": 3}
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            audio = root / "audio.flac"
            checkpoint = root / "beat.ckpt"
            audio.write_bytes(b"audio")
            checkpoint.write_bytes(b"checkpoint")
            with patch("scripts.beat_tracking.track_beats", return_value=report) as track:
                self.assertEqual(transcriber.main([
                    "analyze-beats", "--data-root", str(root), "--audio", str(audio),
                    "--checkpoint", str(checkpoint), "--output", "runs\\beats.json",
                ]), 0)
            track.assert_called_once_with(audio.resolve(), checkpoint.resolve(), device="cpu")
            self.assertEqual(read_json(root / "runs" / "beats.json"), report)

    def test_removed_optional_export_inputs_are_rejected(self):
        for option in ("--fingering-arranger", "--symbolic-completer", "--playing-evidence", "--hand-position-evidence"):
            with self.subTest(option=option), patch("sys.stderr", new=StringIO()), self.assertRaises(SystemExit):
                transcriber.argument_parser().parse_args([
                    "export-gp", "--predictions", "predictions.json", "--template", "template.gpt", option, "optional.json",
                ])

    def test_whole_audio_stitching_has_no_uncovered_or_duplicated_frame_positions(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, features, conditioning, lengths):
                return {"note_onset_logits": torch.full((1, features.shape[1], 6), 2., device=features.device)}

        with TemporaryDirectory(prefix="harness-inference-test-", dir=ROOT) as directory:
            root = Path(directory)
            rate = 8000
            samples = (.1 * np.sin(2 * np.pi * 440 * np.arange(rate * 14) / rate)).astype(np.float32)
            audio = root / "synthetic.wav"
            sf.write(audio, samples, rate)
            metadata = root / "metadata.json"
            publish_json(metadata, self.metadata())
            checkpoint_path = root / "synthetic.pt"
            checkpoint_path.write_bytes(b"synthetic checkpoint identity")
            feature_config = FeatureConfig(sample_rate=rate, n_fft=512, hop_length=160, n_mels=16, f_max=3000.)
            captured = {}

            def decode(outputs, times, **kwargs):
                captured.update(outputs=outputs, times=times)
                return {"notes": [], "percussion": [], "policy": {"synthetic": True}}

            args = SimpleNamespace(checkpoint=str(checkpoint_path), device="cpu", metadata=str(metadata), audio=str(audio), output="runs/prediction.json", data_root=root, onset_threshold=.5, percussion_threshold=.5)
            identity = {"features": asdict(feature_config), "model": {"conditioning_dim": 16}}
            with patch.object(transcriber, "checkpoint_model", return_value=(ConstantModel(), {"identity": identity}, torch.device("cpu"))), patch.dict("sys.modules", {"scripts.transcriber_model": SimpleNamespace(decode_events=decode)}):
                result = transcriber.infer(args)
            self.assertEqual(len(captured["times"]), 700)
            torch.testing.assert_close(captured["outputs"]["note_onset_logits"], torch.full((700, 6), 2.))
            self.assertEqual(result["audioDurationSeconds"], 14)
            self.assertTrue(result["gpWriterImplemented"])
            self.assertFalse(result["gpWrittenByThisCommand"])
            self.assertFalse(result["modelTrainingPerformedByThisCommand"])
            self.assertEqual(read_json(root / "runs" / "prediction.json")["visibility"], "private")


if __name__ == "__main__":
    unittest.main()
