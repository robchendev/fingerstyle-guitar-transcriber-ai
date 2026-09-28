from dataclasses import asdict
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from scripts import transcriber
from scripts.dataset_io import ROOT, publish_json, read_json
from scripts.transcriber_audio import HarnessError
from scripts.transcriber_model import FingerstyleTranscriber, ModelConfig
from scripts.fretboard_features import SCHEMA_VERSION, STRUCTURED_DIM


class PairedCommandTests(unittest.TestCase):
    def video_config(self):
        from scripts.transcriber_video import VideoConfig

        config = transcriber.default_config()
        config["video"] = {"index": "runs\\video-index.json", "model": asdict(VideoConfig())}
        return config

    def test_optional_video_config_is_strict_and_audio_default_unchanged(self):
        self.assertNotIn("video", transcriber.default_config())
        with TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "config.json"
            config = self.video_config()
            publish_json(path, config)
            self.assertEqual(transcriber.load_config(path)[0], config)
            config["video"]["model"]["unexpected"] = 1
            publish_json(path, config)
            with self.assertRaisesRegex(HarnessError, "Unknown video"):
                transcriber.load_config(path)

    def test_rgb_checkpoint_configuration_is_rejected_instead_of_reinterpreted(self):
        values = self.video_config()["video"]["model"]
        self.assertNotIn("image_size", values)
        self.assertEqual(values["input_schema_version"], SCHEMA_VERSION)
        with self.assertRaisesRegex(HarnessError, "not RGB checkpoints"):
            transcriber.video_model_config({**values, "image_size": 96})

    def test_preflight_uses_numeric_masks_without_image_arrays(self):
        config = self.video_config()
        base = transcriber.load_config()
        video = {
            "structured": torch.zeros(1, 4, STRUCTURED_DIM),
            "structured_available": torch.zeros(1, 4, STRUCTURED_DIM, dtype=torch.bool),
            "technique_available": torch.ones(1, dtype=torch.bool),
            "segment_id": torch.tensor([[0, -1, 1, -1]]),
            "frame_indices": torch.tensor([0, -1]),
        }
        video["structured_available"][:, 0, :98] = True
        video["structured_available"][:, 0, 42:84] = False
        video["structured"][:, 0, 84] = 1
        video["structured_available"][:, 2, 98:140] = True
        percussion = torch.zeros(2, 3)
        percussion[0, 0] = 1
        item = {"features": torch.zeros(2, base[1].n_mels), "conditioning": torch.zeros(2, 16), "video": video,
                "metadata": {"stringFrameCollisionsMasked": 0}, "targets": {"percussion": percussion},
                "masks": {"percussion": torch.ones(2, 3, dtype=torch.bool)}}

        class Dataset(list):
            video_coverage = {"releaseWindows": 1, "pairedWindows": 1, "releaseRecordings": 1, "pairedRecordings": 1, "audioFramesWithUsableVideo": 1}
            video_identity = {"featureDimension": STRUCTURED_DIM}
            manifest_sha256 = "synthetic"

        gradient_modes = []
        wrap = transcriber._wrap_video_model

        def inspect_forward(model, config):
            model = wrap(model, config)
            model.register_forward_pre_hook(lambda *_: gradient_modes.append(torch.is_grad_enabled()))
            return model

        with TemporaryDirectory(dir=ROOT / "runs") as directory:
            args = SimpleNamespace(config=None, forward=True, data_root=ROOT, manifest=None, output=str(Path(directory) / "preflight.json"))
            with patch.object(transcriber, "load_config", return_value=(config, *base[1:])), patch.object(transcriber, "make_dataset", return_value=Dataset([item])), patch.object(transcriber, "_wrap_video_model", side_effect=inspect_forward), patch("torch.optim.AdamW", side_effect=AssertionError("Preflight must not construct an optimizer")), patch("scripts.transcriber_runtime.load_checkpoint", side_effect=AssertionError("No checkpoint needed")), patch("sys.stdout", new=StringIO()) as output:
                result = transcriber.preflight(args)
        for split in ("train", "validation"):
            self.assertIn(f"Preflight {split}: loaded 1 windows", output.getvalue())
            self.assertIn(f"Preflight {split}: running no-gradient forward pass", output.getvalue())
            self.assertIn(f"Preflight {split}: forward pass completed", output.getvalue())
        self.assertEqual(gradient_modes, [False, False])
        self.assertEqual(result["weights"], "untrained-in-memory-only")
        self.assertEqual(result["initialization"], "joint-audio-numeric-video-from-scratch")
        for split in result["splits"].values():
            self.assertEqual(split["video_frames"], 1)
            self.assertEqual(split["video_available_view_frames"], 2)
            self.assertEqual(split["structured_available_values"], 98)
            self.assertEqual(split["guitar_relative_observation_view_frames"], 1)
            self.assertEqual(split["independent_hand_observation_view_frames"], 1)
            self.assertEqual(split["unassigned_hand_observation_view_frames"], 1)
            self.assertEqual(split["positiveTargetFrames"]["wrist_thump"], 1)
            self.assertEqual(split["positiveTargetFramesWithUsablePluckingVideo"]["wrist_thump"], 0)
            self.assertIn("forwardShapes", split)
        self.assertIn("no RGB neural-network inputs", result["videoInputs"])
        self.assertNotIn("warnings", result)
        Dataset.video_coverage = {**Dataset.video_coverage, "audioFramesWithUsableVideo": 0}
        video["structured_available"].zero_()
        video["structured"].zero_()
        video["segment_id"].fill_(-1)
        video["frame_indices"].fill_(-1)
        with TemporaryDirectory(dir=ROOT / "runs") as directory:
            args.output = str(Path(directory) / "missing-video.json")
            args.forward = False
            with patch.object(transcriber, "load_config", return_value=(config, *base[1:])), patch.object(transcriber, "make_dataset", return_value=Dataset([item])), patch("sys.stdout", new=StringIO()):
                missing = transcriber.preflight(args)
        self.assertEqual(len(missing["warnings"]), 2)
        self.assertTrue(all("no usable numeric video" in warning for warning in missing["warnings"]))
        self.assertEqual(missing["splits"]["train"]["tracking_gap_only_windows"], 1)

    def test_configuration_starts_joint_v4_without_reading_an_audio_checkpoint(self):
        from scripts.paired_video import build_index
        from tests.test_dataset_release import synthetic_release
        from tests.test_paired_video import bundle_fixture, schema5_bundle_fixture

        with TemporaryDirectory(dir=ROOT / "runs") as directory:
            root = Path(directory)
            path = root / "config.json"
            manifest = synthetic_release(root / "custom-release")
            bundle, _, _ = schema5_bundle_fixture(root, manifest.parent / "audio" / "piece-0.flac")
            index, index_document = build_index(manifest, [bundle], root / "paired-index.json", root=ROOT)
            base = transcriber.default_config()
            with patch("scripts.transcriber_runtime.load_checkpoint", side_effect=AssertionError("Fresh joint configuration must not read weights")), patch("sys.stdout", new=StringIO()) as output:
                result = transcriber.main(["config", "--video-index", str(index), "--output", str(path)])
            self.assertEqual(result, 0)
            self.assertIn("Configuration: validating paired index", output.getvalue())
            self.assertIn("Configuration: paired index validated", output.getvalue())
            self.assertIn("Configuration: completed", output.getvalue())
            value = read_json(path)
            self.assertEqual(value["model"], base["model"])
            self.assertEqual(value["training"], base["training"])
            self.assertEqual(value["video"]["index"], str(index))
            self.assertEqual(value["data"]["manifest"], str(manifest.relative_to(ROOT)))
            self.assertEqual(value["video"]["model"]["architecture_version"], 6)
            self.assertEqual(value["video"]["model"]["input_schema_version"], SCHEMA_VERSION)
            self.assertEqual(value["video"]["model"]["structured_dim"], STRUCTURED_DIM)
            self.assertEqual(value["video"]["model"]["modality_dropout"], .2)
            self.assertNotIn("freeze_audio", value["video"]["model"])
            self.assertEqual(transcriber.load_config(path)[0], value)

            legacy = root / "index-without-manifest-path.json"
            del index_document["manifestPath"]
            publish_json(legacy, index_document)
            wrong = root / "wrong-manifest.json"
            publish_json(wrong, {"notTheRelease": True})
            base["data"]["manifest"] = str(wrong.relative_to(ROOT))
            output = root / "legacy-config.json"
            arguments = ["config", "--video-index", str(legacy), "--output", str(output)]
            with patch.object(transcriber, "default_config", return_value=base), patch("sys.stdout", new=StringIO()), patch("sys.stderr", new=StringIO()) as error:
                self.assertEqual(transcriber.main(arguments), 1)
                self.assertIn("does not match the paired index manifest hash", error.getvalue())
                self.assertFalse(output.exists())
                self.assertEqual(transcriber.main([*arguments, "--manifest", str(manifest)]), 0)
            self.assertEqual(read_json(output)["data"]["manifest"], str(manifest.relative_to(ROOT)))

    def test_frozen_and_unversioned_video_configurations_are_rejected(self):
        for change in (
            {"freeze_audio": True}, {"architecture_version": 1}, {"architecture_version": 2},
            {"architecture_version": None}, {"input_schema_version": 2}, {"structured_dim": 98},
            {"architecture_version": 3}, {"input_schema_version": 3}, {"structured_dim": 186},
        ):
            with self.subTest(change=change), self.assertRaises(HarnessError):
                transcriber.video_model_config({**self.video_config()["video"]["model"], **change})

    def test_obsolete_audio_checkpoint_flags_are_removed(self):
        for command in (["config"], ["preflight"], ["train", "--run-dir", "runs\\new"]):
            with self.subTest(command=command), patch("sys.stderr", new=StringIO()), self.assertRaises(SystemExit):
                transcriber.argument_parser().parse_args([*command, "--audio-checkpoint", "old.pt"])
        with patch("sys.stderr", new=StringIO()), self.assertRaises(SystemExit):
            transcriber.argument_parser().parse_args(["train", "--run-dir", "runs\\new", "--initialize-from", "old.pt"])

    def test_paired_selection_uses_same_dataset_and_preserves_immutable_release_counts(self):
        config = self.video_config()
        _, features, model_config, _ = transcriber.load_config()
        train, other = {"row": {"id": "train"}}, {"row": {"id": "unpaired"}}
        dataset = SimpleNamespace(
            video_identity={"featureDimension": STRUCTURED_DIM},
            video_paired_coverage={"preparedAudioFrames": 10, "audioFramesWithUsableVideo": 8},
            video_paired_window_indices=[1], records=[train, other],
            windows=[(other, {"windowId": "zero"}), (train, {"windowId": "one"})],
        )
        with patch.object(transcriber, "TrainingDataset", return_value=dataset) as loader:
            selected = transcriber.make_dataset(config, features, model_config, "train", ROOT)
        self.assertIs(selected, dataset)
        self.assertEqual(selected.records, [train])
        self.assertEqual(selected.windows, [(train, {"windowId": "one"})])
        self.assertEqual(selected.video_coverage["releaseWindows"], 2)
        self.assertEqual(selected.video_coverage["pairedWindows"], 1)
        self.assertEqual(selected.video_coverage["excludedWindows"], 1)
        self.assertEqual(selected.video_coverage["excludedRecordings"], 1)
        self.assertEqual(selected.video_coverage["preparedAudioFrames"], 10)
        self.assertEqual(selected.video_coverage["audioFramesWithUsableVideo"], 8)
        self.assertTrue(selected.paired_only)
        self.assertEqual(loader.call_args.kwargs["video_index_path"], ROOT / "runs" / "video-index.json")

    def test_paired_checkpoint_evaluation_requires_explicit_video_or_audio_ablation(self):
        identity = {"video": {"config": self.video_config()["video"]["model"]}}
        with self.assertRaises(HarnessError):
            transcriber._evaluation_video_config(SimpleNamespace(), identity, {})
        config = {}
        transcriber._evaluation_video_config(SimpleNamespace(video_index="v.json"), identity, config)
        self.assertEqual(config["video"]["index"], "v.json")
        config = {}
        transcriber._evaluation_video_config(SimpleNamespace(audio_only=True), identity, config)
        self.assertNotIn("video", config)
        with self.assertRaises(HarnessError):
            transcriber._evaluation_video_config(SimpleNamespace(video_index="v.json"), {}, {})

    def test_fresh_paired_training_wraps_random_audio_without_loading_checkpoint(self):
        config = self.video_config()
        ordinary = transcriber.load_config()
        class Dataset(list):
            records = [{"row": {"id": "synthetic"}}]
            manifest_sha256 = "synthetic"
            video_identity = {"synthetic": True}
            video_coverage = {"excludedWindows": 1, "excludedRecordings": 1, "audioFramesWithUsableVideo": 8, "preparedAudioFrames": 10, "trackingGapOnlyWindows": 0}

        dataset = Dataset([None])
        with TemporaryDirectory(dir=ROOT / "runs") as directory:
            args = SimpleNamespace(config=None, resume=None, initialize_from=None, data_root=ROOT, manifest=None, run_dir=directory)
            with patch.object(transcriber, "load_config", return_value=(config, *ordinary[1:])), patch.object(transcriber, "make_dataset", return_value=dataset), patch.object(transcriber, "make_loader"), patch.object(transcriber, "print_training_summary"), patch("scripts.transcriber_runtime.run_training", return_value={"elapsed_seconds": 0.}) as run, patch("scripts.transcriber_runtime.load_checkpoint", side_effect=AssertionError("Must not import old audio")), patch("sys.stdout", new=StringIO()):
                transcriber.train(args)
        model = run.call_args.args[0]
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(run.call_args.args[5]["initialization"]["kind"], "joint-audio-numeric-video-from-scratch")
        self.assertIsNone(run.call_args.kwargs["resume"])

    def test_joint_training_blocks_an_entirely_unavailable_visual_split_before_model_or_optimizer(self):
        config = self.video_config()
        ordinary = transcriber.load_config()
        class Dataset(list):
            records = [{}]
            video_coverage = {"audioFramesWithUsableVideo": 0}

        dataset = Dataset([None])
        args = SimpleNamespace(config=None, resume=None, data_root=ROOT, manifest=None, run_dir="runs\\synthetic-unavailable-video")
        with patch.object(transcriber, "load_config", return_value=(config, *ordinary[1:])), patch.object(transcriber, "make_dataset", return_value=dataset), patch("scripts.transcriber_model.FingerstyleTranscriber") as model, patch("scripts.transcriber_runtime.run_training") as run, patch("sys.stdout", new=StringIO()):
            with self.assertRaisesRegex(HarnessError, "no usable numeric video"):
                transcriber.train(args)
        model.assert_not_called()
        run.assert_not_called()

    def test_checkpoint_loader_reconstructs_self_contained_combined_model(self):
        from scripts.transcriber_video import AudioVideoTranscriber, VideoConfig

        config = ModelConfig(architecture_version=2, n_mels=4, hidden_size=4, recurrent_layers=1, dropout=0)
        video_config = VideoConfig(hidden_size=8)
        original = AudioVideoTranscriber(FingerstyleTranscriber(config), video_config).eval()
        checkpoint = {
            "global_step": 1,
            "history": [{"optimizer_updates": 1}],
            "identity": {"model": asdict(config), "video": {"config": asdict(video_config)}},
            "model_state": original.state_dict(),
        }
        with patch("scripts.transcriber_runtime.load_checkpoint", return_value=checkpoint):
            restored, _, device = transcriber.checkpoint_model("synthetic.pt", "cpu")
        self.assertEqual(str(device), "cpu")
        self.assertIsInstance(restored, AudioVideoTranscriber)
        features, conditioning = torch.randn(1, 4, 4), torch.zeros(1, 4, 12)
        with torch.no_grad():
            expected = original(features, conditioning)
            actual = restored(features, conditioning)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_training_summary_does_not_call_skipped_batches_optimizer_updates(self):
        result = {
            "dataset_recordings": {"train": 1, "validation": 1}, "dataset_windows": {"train": 2, "validation": 1},
            "elapsed_seconds": 2., "setup_seconds": 0., "training_elapsed_seconds": 2.,
            "training_windows_processed": 2, "validation_windows_processed": 1,
            "training_steps_processed": 2, "global_step": 2, "epoch": 1, "epochs_requested": 1,
            "epochs_completed_this_run": 1, "stopped_by": "epochs", "validation": {"loss": 1},
            "best_score": 1, "latest_checkpoint": "latest.pt", "best_checkpoint": "best.pt",
            "optimizer_updates": 1, "optimizer_skipped_batches": 1,
        }
        with patch("sys.stdout", new=StringIO()) as output:
            transcriber.print_training_summary(result, Path("summary.json"))
        self.assertIn("optimizer updates this invocation: 1 | skipped updates: 1", output.getvalue())
        self.assertNotIn("Optimizer steps: 2", output.getvalue())

    def test_infer_passes_both_modalities_to_existing_decoder_and_preserves_audio_fallback(self):
        from scripts.transcriber_video import AudioVideoTranscriber, VideoConfig

        _, features, _, _ = transcriber.load_config()
        config = ModelConfig(architecture_version=2, n_mels=features.n_mels, hidden_size=4, recurrent_layers=1, dropout=0)
        visual = VideoConfig(hidden_size=8)
        model = AudioVideoTranscriber(FingerstyleTranscriber(config), visual).eval()
        with torch.no_grad():
            model.audio.heads["note_onset_logits"].weight.zero_()
            model.audio.heads["note_onset_logits"].bias.fill_(-12)
            model.audio.heads["percussion_logits"].weight.zero_()
            model.audio.heads["percussion_logits"].bias.fill_(-12)
            model.audio.heads["percussion_logits"].bias[0] = 8
        checkpoint = {"identity": {"features": asdict(features), "model": asdict(config), "video": {"config": asdict(visual)}}}
        bundle = SimpleNamespace(identity={"featureDimension": STRUCTURED_DIM}, check_unchanged=lambda: None)
        calls = []

        def window(times):
            calls.append(times.copy())
            count = len(times)
            video = {
                "technique_available": torch.ones(1, dtype=torch.bool),
                "segment_id": torch.zeros(1, 4, dtype=torch.long),
                "structured": torch.ones(1, 4, STRUCTURED_DIM) * .1,
                "structured_available": torch.ones(1, 4, STRUCTURED_DIM, dtype=torch.bool),
                "frame_indices": torch.zeros(count, dtype=torch.long),
            }
            video["structured_available"][:, 2:, 186:] = False
            video["structured"].masked_fill_(~video["structured_available"], 0)
            return video

        bundle.window = window
        with TemporaryDirectory(dir=ROOT / "runs") as directory:
            root = Path(directory)
            audio, meta, weights = root / "audio.wav", root / "metadata.json", root / "synthetic.pt"
            sf.write(audio, np.sin(np.arange(6000) * .05).astype(np.float32) * .1, features.sample_rate)
            publish_json(meta, {"openStringMidi": [40, 45, 50, 55, 59, 64], "capoFret": 0, "tempo": {"bpm": 120, "beatUnit": [1, 4]}, "timeSignature": [4, 4]})
            weights.write_bytes(b"synthetic-checkpoint-identity")
            common = ["infer", "--checkpoint", str(weights), "--audio", str(audio), "--metadata", str(meta)]
            with patch.object(transcriber, "checkpoint_model", return_value=(model, checkpoint, torch.device("cpu"))), patch("scripts.paired_video.load_inference_video", return_value=bundle), patch("sys.stdout", new=StringIO()):
                self.assertEqual(transcriber.main([*common, "--video-bundle", "inputs.json", "--output", str(root / "paired.json")]), 0)
                self.assertEqual(transcriber.main([*common, "--output", str(root / "audio-only.json")]), 0)
            paired, base = read_json(root / "paired.json"), read_json(root / "audio-only.json")
            self.assertEqual(len(calls), 1)
            self.assertTrue(paired["pairedVideo"]["provided"])
            self.assertEqual(paired["pairedVideo"]["architectureVersion"], 6)
            self.assertEqual(paired["pairedVideo"]["inputSchemaVersion"], SCHEMA_VERSION)
            self.assertEqual(paired["pairedVideo"]["featureDimension"], STRUCTURED_DIM)
            self.assertTrue(base["pairedVideo"]["audioOnlyFallback"])
            self.assertEqual(base["pairedVideo"]["audioOnlyPolicy"], "jointly-learned-audio-path")
            for name in ("notes", "percussion", "techniques"):
                self.assertIsInstance(paired[name], list)
                self.assertIsInstance(base[name], list)
            from scripts.gp_output import write_gp_outputs
            from tests.test_gp_output import output_template
            from tests.test_gp_normalization import archive_bytes

            template = root / "template.gpt"
            template.write_bytes(archive_bytes(output_template()))
            report = write_gp_outputs(template, paired, root / "full.gp", root / "single.gp")
            self.assertFalse(report["trainingPerformed"])
            self.assertTrue((root / "single.gp").is_file())


if __name__ == "__main__":
    unittest.main()
