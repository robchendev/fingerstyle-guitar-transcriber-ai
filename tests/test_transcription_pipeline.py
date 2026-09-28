"""Synthetic files only; the integration fixture uses one tiny numerical update."""

from copy import deepcopy
from dataclasses import asdict
from io import StringIO
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from scripts import transcriber
from scripts import transcription_pipeline as pipeline
from scripts.dataset_io import ROOT, publish_json, read_json, sha256
from scripts.transcriber_audio import FeatureConfig, HarnessError
from scripts.transcriber_model import FingerstyleTranscriber, ModelConfig
from scripts.transcriber_runtime import TrainingConfig, load_checkpoint, run_training
from scripts.video_features import INPUT_REPRESENTATION, SCHEMA_VERSION, STRUCTURED_DIM
from tests.test_gp_normalization import archive_bytes
from tests.test_gp_output import gp_root, output_template
from tests.test_transcriber_model import synthetic_targets


class TranscriptionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.fixture_directory = TemporaryDirectory(prefix="transcription-fixture-", dir=ROOT / "runs")
        cls.fixture_root = Path(cls.fixture_directory.name)
        features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3900)
        model_config = ModelConfig(n_mels=16, hidden_size=4, recurrent_layers=1, dropout=0)
        model = FingerstyleTranscriber(model_config)
        with torch.no_grad():
            for name, head in model.heads.items():
                head.weight.zero_()
                head.bias.zero_()
            model.heads["note_onset_logits"].bias.fill_(-12)
            model.heads["note_onset_logits"].bias[0] = 12
            model.heads["percussion_logits"].bias.fill_(-12)
            model.heads["harmonic_logits"].bias.fill_(-12)
            model.heads["duration_log"].bias.fill_(math.log1p(1))
            for name, size, selected in (("fret_logits", 37, 0), ("pitch_logits", 128, 40), ("voice_logits", 4, 0)):
                values = model.heads[name].bias.view(6, size)
                values.fill_(-12)
                values[:, selected] = 12
        targets, masks, valid = synthetic_targets(batch=1, frames=8)
        targets["note_onset"].zero_()
        targets["note_onset"][:, :, 0] = 1
        masks["note_onset"].fill_(True)
        item = {
            "features": torch.zeros(8, features.n_mels), "conditioning": torch.zeros(8, 12),
            "lengths": torch.tensor(8), "valid_frames": valid[0],
            "targets": {name: value[0] for name, value in targets.items()},
            "masks": {name: value[0] for name, value in masks.items()},
        }
        loaders = [DataLoader([item], batch_size=1, generator=torch.Generator().manual_seed(seed)) for seed in (1, 2)]
        identity = {"manifest_sha256": "synthetic-only", "features": asdict(features), "model": asdict(model_config)}
        from scripts.transcriber_video import AudioVideoTranscriber, VideoConfig

        joint = AudioVideoTranscriber(deepcopy(model), VideoConfig(
            hidden_size=8, structured_dim=194, input_schema_version=4,
            architecture_version=5,
            feature_group_version="anatomy-representation-groups-v1",
        ))
        with patch("sys.stdout", new=StringIO()):
            result = run_training(model, *loaders, TrainingConfig(epochs=1, max_steps=1, device="cpu", learning_rate=1e-5),
                                  cls.fixture_root / "toy-checkpoint", identity)
        cls.checkpoint = Path(result["latest_checkpoint"])
        video = {
            "structured": torch.full((8, 4, STRUCTURED_DIM), .1),
            "structured_available": torch.ones(8, 4, STRUCTURED_DIM, dtype=torch.bool),
            "technique_available": torch.ones(8, dtype=torch.bool),
            "segment_id": torch.zeros(8, 4, dtype=torch.long),
            "frame_indices": torch.arange(8),
        }
        video["structured_available"][:, 2:, 186:] = False
        video["structured"].masked_fill_(~video["structured_available"], 0)
        loaders = [DataLoader([{**item, "video": video}], batch_size=1, generator=torch.Generator().manual_seed(seed)) for seed in (1, 2)]
        result = run_training(
            joint, *loaders, TrainingConfig(epochs=1, max_steps=1, device="cpu", learning_rate=1e-5),
            cls.fixture_root / "toy-joint-checkpoint", {**identity, "video": {"config": asdict(joint.video_config)}},
        )
        cls.joint_checkpoint = Path(result["latest_checkpoint"])

    @classmethod
    def tearDownClass(cls):
        cls.fixture_directory.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="transcription-tests-", dir=ROOT / "runs")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.audio = self.root / "audio.wav"
        sf.write(self.audio, .1 * np.sin(2 * np.pi * 110 * np.arange(16000) / 8000), 8000)
        self.metadata = self.root / "metadata.json"
        self.musical_metadata = {"openStringMidi": [38, 45, 50, 55, 59, 64], "capoFret": 2,
                                 "tempo": {"bpm": 100, "beatUnit": [1, 4]}, "timeSignature": [3, 4]}
        publish_json(self.metadata, self.musical_metadata)
        self.weights = self.root / "trained.pt"
        shutil.copyfile(self.checkpoint, self.weights)
        self.template = self.root / "template.gpt"
        self.template.write_bytes(archive_bytes(output_template()))
        self.beat_checkpoint = self.root / "beat.ckpt"
        self.beat_checkpoint.write_bytes(b"synthetic-beat-model")
        self.output = self.root / "runs" / "job"
        self.common = ["transcribe", "--audio", str(self.audio), "--metadata", str(self.metadata),
                       "--checkpoint", str(self.weights), "--template", str(self.template),
                       "--beat-checkpoint", str(self.beat_checkpoint), "--output-directory", str(self.output), "--device", "cpu"]
        self.stdout = self.enterContext(patch("sys.stdout", new=StringIO()))
        self.beat_tracker = self.enterContext(patch("scripts.beat_tracking.track_beats", side_effect=self.beat_evidence))
        self.train = self.enterContext(patch("scripts.transcriber_runtime.run_training", side_effect=AssertionError("Transcription must not train")))

    def beat_evidence(self, audio, checkpoint, *, device):
        return {"schemaVersion": 1, "kind": "audio-beat-evidence", "audioSha256": sha256(audio),
                "checkpointSha256": sha256(checkpoint), "beatSeconds": [0., .6, 1.2, 1.8],
                "downbeatSeconds": [0., 1.8], "beatCount": 4, "downbeatCount": 2, "trainingPerformed": False}

    def args(self, *extra):
        return transcriber.argument_parser().parse_args([*self.common, *extra])

    def run_job(self, *extra):
        return pipeline.run_transcription(self.args(*extra))

    def state(self):
        return read_json(self.output / pipeline.STATE_NAME)

    def status_args(self):
        return transcriber.argument_parser().parse_args(["transcribe-status", "--output-directory", str(self.output)])

    def embedded_video_args(self, *, audio_streams=1, offset=.4, compressed=False):
        video = self.root / ("embedded.mp4" if compressed else "embedded.mkv")
        command = [pipeline.executable("ffmpeg"), "-nostdin", "-v", "error", "-n", "-threads", "1",
                   "-f", "lavfi", "-i", "color=c=black:s=64x48:r=5:d=2.6"]
        if audio_streams:
            command.extend(["-itsoffset", str(offset), "-i", str(self.audio)])
        command.extend(["-map", "0:v:0"])
        for _ in range(audio_streams):
            command.extend(["-map", "1:a:0"])
        command.extend(["-c:v", "mpeg4" if compressed else "ffv1", "-c:a", "aac" if compressed else "pcm_f32le", "-threads", "1", str(video)])
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        model = self.root / "hand.task"
        model.write_bytes(b"synthetic hand model; worker mocked")
        shutil.copyfile(self.joint_checkpoint, self.weights)
        arguments = list(self.common)
        index = arguments.index("--audio")
        del arguments[index:index + 2]
        video_python = os.environ.get("VIDEO_PYTHON", str(ROOT / "scripts" / "video-evidence" / ".venv" / "Scripts" / "python.exe"))
        return [*arguments, "--video", str(video), "--video-python", video_python, "--hand-model", str(model), "--plucking-screen-side", "left"], video

    def embedded_video_worker(self, command, **kwargs):
        from tests.test_paired_video import bundle_fixture

        request = read_json(command[command.index("--request") + 1])
        output = Path(command[command.index("--output") + 1])
        directory = Path(request["outputDirectory"])
        directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, directory.parent)
        alignment_path = Path(request["reuse"]["alignment"])
        alignment = read_json(alignment_path)
        offset = alignment["videoStartSecondsForTrimmedAudioZero"]
        first_pts = round(offset * 1000)
        pts = tuple(first_pts + step for step in (0, 40, 80, 120))
        bundle, report, arrays = bundle_fixture(directory, Path(request["audio"]), identifier="input", pts=pts)
        arrays["audio_seconds"] = (np.asarray(pts, dtype=np.float64) - first_pts) / 1000
        np.savez_compressed(bundle.with_name("inputs.npz"), **arrays)
        report["arraysSha256"] = sha256(bundle.with_name("inputs.npz"))
        report["inputPaths"].update(video=request["video"], alignment=str(alignment_path))
        report["inputSha256"].update(video=sha256(request["video"]), alignment=sha256(alignment_path))
        report["videoSha256"] = sha256(request["video"])
        report["clock"]["offsetSamples"] = -round(offset * report["clock"]["sampleRate"])
        publish_json(bundle, report)
        publish_json(output, {
            "schemaVersion": 1, "kind": "paired-video-preparation-result", "id": "input", "status": "ready",
            "inputSha256": {name: sha256(request[name]) for name in ("audio", "video")},
            "artifacts": {"bundle": str(bundle), "alignment": str(alignment_path)}, "actions": [],
        })
        return subprocess.CompletedProcess(command, 0)

    def test_single_video_extracts_soundtrack_with_source_timing_exports_gp_and_resumes(self):
        arguments, video = self.embedded_video_args()
        tool_directories = {str(Path(pipeline.executable(name)).parent) for name in ("ffmpeg", "ffprobe")}
        arguments.extend(["--ffmpeg-dir", str(Path(pipeline.executable("ffmpeg")).parent)])
        original_video = sha256(video)

        def failed_extract(attempt, state):
            (attempt / "soundtrack.wav").write_bytes(b"preserve interrupted decode")
            raise HarnessError("synthetic decode failure")

        with patch.object(pipeline, "_extract_audio", side_effect=failed_extract):
            self.assertEqual(transcriber.main(arguments), 1)
        partial = Path(self.state()["stages"]["audio"]["directory"]) / "soundtrack.wav"
        self.assertEqual(self.state()["stages"]["audio"]["status"], "failed")
        actual_run = subprocess.run

        def dispatch(command, **kwargs):
            if "align-audio" in command or "--request" in command:
                for directory in tool_directories:
                    self.assertIn(directory, kwargs["env"]["PATH"].split(os.pathsep))
            if len(command) > 1 and Path(command[1]).name == "batch_video.py":
                return self.embedded_video_worker(command, **kwargs)
            return actual_run(command, **kwargs)

        path_without_tools = os.pathsep.join(value for value in os.environ.get("PATH", "").split(os.pathsep) if value not in tool_directories)
        with patch.dict(os.environ, PATH=path_without_tools), patch("scripts.transcription_pipeline.subprocess.run", side_effect=dispatch):
            self.assertEqual(transcriber.main(arguments), 0)
        result = read_json(self.output / pipeline.SUMMARY_NAME)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["stages"]["audio"], "complete")
        self.assertNotIn("audio", result["inputIdentity"])
        extracted = Path(result["outputs"]["audio"])
        receipt = read_json(result["outputs"]["audioExtraction"])
        self.assertAlmostEqual(receipt["videoStartSecondsForAudioZero"], .4)
        samples, rate = sf.read(extracted)
        expected, expected_rate = sf.read(self.audio)
        self.assertEqual(rate, expected_rate)
        np.testing.assert_array_equal(samples, expected)
        alignment = read_json(result["outputs"]["alignment"])
        self.assertAlmostEqual(alignment["videoStartSecondsForTrimmedAudioZero"], .4)
        self.assertEqual(alignment["trimmedAudioSha256"], sha256(extracted))
        self.assertFalse(alignment["reviewRequired"])
        checked = subprocess.run([
            self.state()["options"]["video_python"], "-c",
            "import sys; from hand_motion import load_audio_clock; load_audio_clock(sys.argv[1], sys.argv[2], sys.argv[3])",
            original_video, str(extracted), self.state()["videoAlignment"],
        ], cwd=ROOT / "scripts" / "video-evidence", check=False, capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(self.beat_tracker.call_args.args[0], extracted)
        self.assertEqual(read_json(result["outputs"]["predictions"])["audioSha256"], sha256(extracted))
        for name in ("fullVoices", "singleVoice"):
            self.assertGreater(len(gp_root(result["outputs"][name]).findall("./Notes/Note")), 0)
        self.assertEqual(partial.read_bytes(), b"preserve interrupted decode")
        self.assertEqual(sha256(video), original_video)
        with patch.object(pipeline, "_extract_audio", side_effect=AssertionError("Do not decode twice")), patch.object(pipeline, "_worker", side_effect=AssertionError("Do not prepare video twice")):
            self.assertEqual(transcriber.main(arguments), 0)
            self.assertEqual(pipeline.transcription_status(self.status_args()), result)
        extracted.write_bytes(b"modified extracted audio")
        with self.assertRaisesRegex(HarnessError, "artifact changed"):
            pipeline.transcription_status(self.status_args())
        self.train.assert_not_called()

    def test_video_only_aac_decode_and_review_resume_reuse_the_soundtrack(self):
        arguments, video = self.embedded_video_args(offset=0, compressed=True)
        def review(state, output, review_flags=None):
            request = read_json(state["videoRequest"])
            self.assertNotEqual(request["audio"], str(video))
            result = {"status": "needs-review", "artifacts": {}, "actions": [{"stage": "shots", "reason": "synthetic review"}]}
            publish_json(output, result)
            return result

        with patch.object(pipeline, "_worker", side_effect=review):
            pending = pipeline.run_transcription(transcriber.argument_parser().parse_args(arguments))
        self.addCleanup(shutil.rmtree, Path(self.state()["videoDirectory"]).parent)
        self.assertEqual(pending["stages"]["audio"], "complete")
        extracted = Path(pending["outputs"]["audio"])
        self.assertGreater(sf.info(extracted).frames, 0)
        digest = sha256(extracted)
        with patch.object(pipeline, "_extract_audio", side_effect=AssertionError("Review must reuse decoded audio")), patch.object(pipeline, "_worker", side_effect=review):
            repeated = pipeline.run_transcription(transcriber.argument_parser().parse_args(arguments))
        self.assertEqual(repeated["status"], "needs-review")
        self.assertEqual(repeated["outputs"]["audio"], str(extracted))
        self.assertEqual(sha256(extracted), digest)

    def test_root_python_launcher_generates_metadata_exports_gp_and_resumes(self):
        import transcribe_video as launcher
        from scripts.transcriber_runtime import INFERENCE_FORMAT, INFERENCE_SCHEMA_VERSION

        arguments, video = self.embedded_video_args()
        checkpoint = load_checkpoint(self.weights)
        model_path = self.root / "models" / "transcriber.pt"
        model_path.parent.mkdir()
        torch.save({
            "format": INFERENCE_FORMAT, "schema_version": INFERENCE_SCHEMA_VERSION,
            "model_config": checkpoint["identity"]["model"],
            "feature_config": checkpoint["identity"]["features"],
            "video_config": checkpoint["identity"]["video"]["config"],
            "model_state": checkpoint["model_state"],
        }, model_path)
        command = [
            "--video", str(video), "--output-directory", str(self.output),
            "--template", str(self.template), "--beat-checkpoint", str(self.beat_checkpoint),
            "--plucking-screen-side", "left",
            "--tuning", "38", "45", "50", "55", "59", "64", "--capo", "2",
            "--bpm", "100", "--beat-unit", "1/4", "--time-signature", "3/4",
            "--first-full-bar-downbeat", "0",
            "--note-cutoff", ".78", "--x-cutoff", ".1",
        ]
        real_pipeline = pipeline.run_transcription

        def run(args):
            args.hand_model = arguments[arguments.index("--hand-model") + 1]
            args.video_python = arguments[arguments.index("--video-python") + 1]
            return real_pipeline(args)

        def worker(state, output, review_flags=None):
            self.embedded_video_worker(["batch_video.py", "--request", state["videoRequest"], "--output", str(output)])
            return read_json(output)

        with patch.object(launcher, "ROOT", self.root), patch.object(pipeline, "run_transcription", side_effect=run), patch.object(pipeline, "_worker", side_effect=worker):
            self.assertEqual(launcher.main(command), 0)
            result = read_json(self.output / pipeline.SUMMARY_NAME)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["inputIdentity"]["checkpoint"]["path"], str(model_path))
            metadata_path = Path(result["inputIdentity"]["metadata"]["path"])
            metadata = read_json(metadata_path)
            for key, value in self.musical_metadata.items():
                self.assertEqual(metadata[key], value)
            self.assertFalse(video.with_suffix(".metadata.json").exists())
            for name in ("fullVoices", "singleVoice"):
                self.assertGreater(len(gp_root(result["outputs"][name]).findall("./Notes/Note")), 0)
            before = (self.output / pipeline.STATE_NAME).read_bytes()
            with patch.object(pipeline, "_extract_audio", side_effect=AssertionError("Do not decode twice")), patch.object(pipeline, "_worker", side_effect=AssertionError("Do not prepare video twice")):
                self.assertEqual(launcher.main(command), 0)
            with patch("sys.stderr", new=StringIO()) as errors:
                self.assertEqual(launcher.main([*command, "--capo", "3"]), 1)
                self.assertIn("Transcription options changed", errors.getvalue())
            self.assertEqual((self.output / pipeline.STATE_NAME).read_bytes(), before)
            self.assertEqual(read_json(metadata_path), metadata)
            self.assertEqual(read_json(self.output / pipeline.SUMMARY_NAME), result)
        self.train.assert_not_called()

    def test_video_without_one_audio_stream_stops_before_inference(self):
        for count, message in ((0, "no audio stream"), (2, "multiple audio streams")):
            with self.subTest(audio_streams=count):
                arguments, _ = self.embedded_video_args(audio_streams=count)
                arguments.extend(["--output-directory", str(self.root / "runs" / f"audio-streams-{count}")])
                with patch.object(pipeline, "_prepare_video") as prepare, patch.object(transcriber, "infer") as infer:
                    with self.assertRaisesRegex(HarnessError, message):
                        pipeline.run_transcription(transcriber.argument_parser().parse_args(arguments))
                prepare.assert_not_called()
                infer.assert_not_called()
                (self.root / "embedded.mkv").unlink()

    def test_missing_input_and_discontinuous_audio_timestamps_are_rejected(self):
        arguments = list(self.common)
        del arguments[1:3]
        with self.assertRaisesRegex(HarnessError, "Supply --audio, --video"):
            pipeline.run_transcription(transcriber.argument_parser().parse_args(arguments))
        streams = {"streams": [
            {"index": 0, "codec_type": "video"},
            {"index": 1, "codec_type": "audio", "sample_rate": "8000", "channels": 1, "time_base": "1/8000"},
        ]}
        frames = {"streams": [streams["streams"][1]], "frames": [
            {"stream_index": 1, "pts": 0, "nb_samples": 800}, {"stream_index": 1, "pts": 1600, "nb_samples": 800}]}
        with patch.object(pipeline, "run_media", return_value=json.dumps(streams)), patch("scripts.local_media.run_media", return_value=json.dumps(frames)):
            with self.assertRaisesRegex(HarnessError, "Discontinuous soundtrack PTS"):
                pipeline._soundtrack_timeline({"ffprobe": "ffprobe", "video": "synthetic.mp4"})

    def test_real_checkpoint_audio_to_both_gp_outputs_matches_explicit_commands_and_resumes(self):
        sources = {path: sha256(path) for path in (self.audio, self.metadata, self.weights, self.template, self.beat_checkpoint)}
        self.assertEqual(transcriber.main(self.common), 0)
        result = read_json(self.output / pipeline.SUMMARY_NAME)
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["trainingPerformed"])
        self.assertEqual(set(result["outputs"]), {"predictions", "beatEvidence", "fullVoices", "singleVoice", "gpReport"})
        predictions = read_json(result["outputs"]["predictions"])
        self.assertEqual(predictions["metadata"], self.musical_metadata)
        self.assertGreater(len(predictions["notes"]), 0)
        self.assertFalse(read_json(result["outputs"]["gpReport"])["trainingPerformed"])
        for name in ("fullVoices", "singleVoice"):
            score = gp_root(result["outputs"][name])
            self.assertGreater(len(score.findall("./Notes/Note")), 0)
        manual = self.root / "runs" / "manual"
        self.assertEqual(transcriber.main(["infer", "--audio", str(self.audio), "--metadata", str(self.metadata),
                                          "--checkpoint", str(self.weights), "--device", "cpu", "--output", str(manual / "predictions.json")]), 0)
        self.assertEqual(transcriber.main(["analyze-beats", "--audio", str(self.audio), "--checkpoint", str(self.beat_checkpoint),
                                          "--output", str(manual / "beats.json")]), 0)
        self.assertEqual(transcriber.main(["export-gp", "--predictions", str(manual / "predictions.json"),
                                          "--template", str(self.template), "--beat-evidence", str(manual / "beats.json"),
                                          "--full-output", str(manual / "full.gp"), "--single-output", str(manual / "single.gp"),
                                          "--report", str(manual / "report.json")]), 0)
        self.assertEqual(predictions, read_json(manual / "predictions.json"))
        for name, file in (("fullVoices", "full.gp"), ("singleVoice", "single.gp")):
            self.assertEqual(ET.tostring(gp_root(result["outputs"][name])), ET.tostring(gp_root(manual / file)))
        before = self.state()
        with patch.object(transcriber, "infer", side_effect=AssertionError("Do not rerun completed inference")), patch.object(transcriber, "export_gp", side_effect=AssertionError("Do not reexport")):
            self.assertEqual(self.run_job(), result)
            self.assertEqual(pipeline.transcription_status(self.status_args()), result)
        self.assertEqual(self.state(), before)
        self.assertEqual(sources, {path: sha256(path) for path in sources})
        self.train.assert_not_called()
        logs = self.stdout.getvalue()
        for message in (
            "Transcription setup: completed.", "Transcription preflight: completed.",
            "Transcription inference: completed.", "Transcription beats: completed.",
            "GP export: confidence filtering completed", "GP export: rhythm quantization completed",
            "GP export: downstroke normalization completed.", "GP export: fingering optimization completed.",
            "GP export: voice assignment completed.", "GP export: full-voice notation completed",
            "GP export: single-voice notation completed.", "GP export: both GP files saved.",
            "GP export: report saved:", "Transcription export: completed.", "Transcription: already completed;",
        ):
            self.assertIn(message, logs)

    def test_failed_export_keeps_partial_attempt_and_reuses_completed_stages(self):
        def failed(args):
            Path(args.full_output).write_bytes(b"preserve partial GP")
            raise HarnessError("synthetic export failure")

        with patch.object(transcriber, "export_gp", side_effect=failed):
            with self.assertRaisesRegex(HarnessError, "synthetic export"):
                self.run_job()
        state = self.state()
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(state["stages"]["export"]["status"], "failed")
        self.assertIn("Transcription: blocked - synthetic export failure", self.stdout.getvalue())
        self.assertNotIn("Transcription export: completed.", self.stdout.getvalue())
        partial = Path(state["stages"]["export"]["directory"]) / "transcription.full-voices.gp"
        before = partial.read_bytes()
        with patch.object(transcriber, "infer", side_effect=AssertionError("Do not rerun inference")), patch.object(transcriber, "analyze_beats", side_effect=AssertionError("Do not rerun beats")):
            result = self.run_job()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(partial.read_bytes(), before)
        self.assertNotEqual(result["outputs"]["fullVoices"], str(partial))
        self.assertEqual(len(self.state()["attempts"]), 4)

    def test_changed_sources_and_options_never_reuse_or_overwrite_ready_job(self):
        result = self.run_job()
        original_state = (self.output / pipeline.STATE_NAME).read_bytes()
        for path in (self.audio, self.metadata, self.weights, self.template, self.beat_checkpoint):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                try:
                    with self.assertRaisesRegex(HarnessError, "changed"):
                        self.run_job()
                    with self.assertRaisesRegex(HarnessError, "changed"):
                        pipeline.transcription_status(self.status_args())
                finally:
                    path.write_bytes(original)
        with self.assertRaisesRegex(HarnessError, "options changed"):
            self.run_job("--beat-device", "cuda")
        with patch.object(pipeline, "_implementation", return_value={"changed": "implementation"}):
            with self.assertRaisesRegex(HarnessError, "implementation changed"):
                self.run_job()
        self.assertEqual((self.output / pipeline.STATE_NAME).read_bytes(), original_state)
        self.assertEqual(pipeline.transcription_status(self.status_args()), result)

    def test_mutated_artifact_and_summary_fail_closed(self):
        result = self.run_job()
        for path in [*map(Path, result["outputs"].values()), self.output / pipeline.SUMMARY_NAME]:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                try:
                    with self.assertRaisesRegex(HarnessError, "artifact changed"):
                        self.run_job()
                    with self.assertRaisesRegex(HarnessError, "artifact changed"):
                        pipeline.transcription_status(self.status_args())
                finally:
                    path.write_bytes(original)

    def test_mid_inference_source_mutation_is_not_completed(self):
        original = transcriber.infer

        def mutate(args):
            result = original(args)
            self.template.write_bytes(b"mutated during inference")
            return result

        with patch.object(transcriber, "infer", side_effect=mutate), patch.object(transcriber, "analyze_beats") as beats:
            with self.assertRaisesRegex(HarnessError, "changed"):
                self.run_job()
        self.assertEqual(self.state()["status"], "blocked")
        self.assertEqual(self.state()["stages"]["inference"]["status"], "failed")
        beats.assert_not_called()

    def test_nonempty_output_source_overlap_and_hardlinks_are_rejected(self):
        self.output.mkdir(parents=True)
        unrelated = self.output / "precious.gp"
        unrelated.write_bytes(b"owner")
        with self.assertRaisesRegex(HarnessError, "must be empty"):
            self.run_job()
        self.assertEqual(unrelated.read_bytes(), b"owner")
        with self.assertRaisesRegex(HarnessError, "Source files must be outside"):
            self.run_job("--output-directory", str(self.root))
        linked = self.root / "linked.wav"
        os.link(self.audio, linked)
        try:
            with self.assertRaisesRegex(ValueError, "without aliases"):
                self.run_job("--audio", str(linked))
        finally:
            linked.unlink()

    def test_bad_metadata_and_untrained_checkpoints_stop_before_inference(self):
        invalid = deepcopy(self.musical_metadata)
        del invalid["tempo"]
        publish_json(self.metadata, invalid)
        with patch.object(transcriber, "infer") as infer:
            with self.assertRaisesRegex(HarnessError, "requires tuning"):
                self.run_job()
        infer.assert_not_called()
        self.assertEqual(self.state()["status"], "blocked")
        publish_json(self.metadata, self.musical_metadata)
        checkpoint = load_checkpoint(self.weights)
        checkpoint["global_step"] = 0
        with patch("scripts.transcriber_runtime.load_checkpoint", return_value=checkpoint):
            with self.assertRaisesRegex(HarnessError, "trained checkpoint"):
                self.run_job("--output-directory", str(self.root / "runs" / "untrained"))

    def video_args(self):
        video, model = self.root / "video.mp4", self.root / "hand.task"
        video.write_bytes(b"synthetic original video")
        model.write_bytes(b"existing cached hand model")
        return ["--video", str(video), "--video-python", sys.executable, "--hand-model", str(model), "--plucking-screen-side", "left"]

    def paired_checkpoint(self):
        return load_checkpoint(self.joint_checkpoint)

    def test_audio_only_checkpoint_with_video_fails_without_worker_or_random_refiner(self):
        with patch("scripts.transcription_pipeline.subprocess.run") as worker, patch.object(transcriber, "infer") as infer:
            with self.assertRaisesRegex(HarnessError, "paired-trained joint checkpoint"):
                self.run_job(*self.video_args())
        worker.assert_not_called()
        infer.assert_not_called()

    def test_paired_checkpoint_without_video_uses_learned_audio_path(self):
        self.assertIn("transcriber_video.py", pipeline._implementation(False))
        shutil.copyfile(self.joint_checkpoint, self.weights)
        with patch("scripts.transcription_pipeline.subprocess.run") as worker:
            paired = self.run_job("--output-directory", str(self.root / "runs" / "paired-audio"))
        actual = read_json(paired["outputs"]["predictions"])
        self.assertTrue(actual["pairedVideo"]["audioOnlyFallback"])
        self.assertEqual(actual["pairedVideo"]["audioOnlyPolicy"], "jointly-learned-audio-path")
        self.assertFalse(actual["modelTrainingPerformedByThisCommand"])
        self.assertTrue(Path(paired["outputs"]["fullVoices"]).is_file())
        worker.assert_not_called()

    def test_raw_video_review_then_anonymous_hand_bundle_reaches_existing_inference_and_gp_writer(self):
        from tests.test_paired_video import bundle_fixture

        flags = self.video_args()
        checkpoint = self.paired_checkpoint()
        calls = []
        video_directory = None

        def worker(command, **kwargs):
            nonlocal video_directory
            request = read_json(command[command.index("--request") + 1])
            output = Path(command[command.index("--output") + 1])
            calls.append((command, request))
            video_directory = Path(request["outputDirectory"])
            self.assertNotIn("gp", request)
            self.assertNotIn("correspondenceReview", request)
            self.assertNotIn("imageSize", request)
            self.assertEqual(request["audio"], str(self.audio))
            self.assertEqual(request["pluckingScreenSide"], "left")
            self.assertTrue(video_directory.is_relative_to(ROOT / "runs" / "video-evidence" / "batches"))
            self.assertTrue(video_directory.parent.name.startswith("transcription-"))
            self.assertEqual(video_directory.name, "input")
            artifacts = {}
            status = "needs-review"
            if command[2] == "review":
                self.assertIn("--accept-shots", command)
                video_directory.mkdir(parents=True, exist_ok=True)
                bundle, document, arrays = bundle_fixture(video_directory, self.audio, identifier="input", pts=(0, 40, 80, 120))
                arrays["structured"].fill(0)
                arrays["structured_available"].fill(False)
                arrays["structured_available"][:, 2:, 98:140] = True
                arrays["structured_available"][:, 2:, 184:186] = True
                arrays["structured"][:, 2:, 100:140] = .25
                arrays["structured"][:, 2:, 184] = 1
                arrays["segment_id"][:] = [-1, -1, 0, 1]
                np.savez_compressed(bundle.with_name(document["arraysPath"]), **arrays)
                document["arraysSha256"] = sha256(bundle.with_name(document["arraysPath"]))
                alignment_path = Path(document["inputPaths"]["alignment"])
                alignment = read_json(alignment_path)
                alignment["videoSha256"] = sha256(request["video"])
                publish_json(alignment_path, alignment)
                document["inputPaths"]["video"] = request["video"]
                document["inputSha256"]["video"] = sha256(request["video"])
                document["inputSha256"]["alignment"] = sha256(alignment_path)
                document["videoSha256"] = sha256(request["video"])
                publish_json(bundle, document)
                artifacts["bundle"] = str(bundle)
                status = "ready"
            publish_json(output, {"schemaVersion": 1, "kind": "paired-video-preparation-result", "id": "input",
                                  "status": status, "inputSha256": {name: sha256(request[name]) for name in ("audio", "video")},
                                  "artifacts": artifacts, "actions": [{"stage": "shots", "reason": "Inspect cuts"}] if status == "needs-review" else []})
            return subprocess.CompletedProcess(command, 0)

        try:
            with patch("scripts.transcriber_runtime.load_checkpoint", return_value=checkpoint), patch("scripts.transcription_pipeline.subprocess.run", side_effect=worker), patch.object(transcriber, "infer", wraps=transcriber.infer) as infer:
                pending = self.run_job(*flags)
                self.assertEqual(pending["status"], "needs-review")
                self.assertIn("Transcription video: needs-review", self.stdout.getvalue())
                self.assertNotIn("Transcription video: completed", self.stdout.getvalue())
                self.assertEqual(pending["outputs"], {})
                infer.assert_not_called()
                self.assertEqual(self.beat_tracker.call_count, 0)
                review = transcriber.argument_parser().parse_args(["transcribe-review", "--output-directory", str(self.output), "--accept-shots", "--add-cut", ".5"])
                ready = pipeline.review_transcription(review)
                self.assertEqual(ready["status"], "ready")
                self.assertIn("Transcription video: completed; aligned numeric hand inputs are ready.", self.stdout.getvalue())
                self.assertEqual(infer.call_count, 1)
                self.assertEqual(infer.call_args.args[0].video_bundle, ready["outputs"]["videoBundle"])
            prediction = read_json(ready["outputs"]["predictions"])
            self.assertTrue(prediction["pairedVideo"]["provided"])
            self.assertFalse(prediction["pairedVideo"]["audioOnlyFallback"])
            self.assertEqual(prediction["pairedVideo"]["inputIdentity"]["schemaVersion"], SCHEMA_VERSION)
            self.assertEqual(prediction["pairedVideo"]["inputIdentity"]["inputRepresentation"], INPUT_REPRESENTATION)
            self.assertEqual(prediction["pairedVideo"]["inputIdentity"]["featureDimension"], STRUCTURED_DIM)
            self.assertNotIn("imageSize", prediction["pairedVideo"]["inputIdentity"])
            self.assertEqual(len(calls), 2)
            self.assertGreater(len(gp_root(ready["outputs"]["fullVoices"]).findall("./Notes/Note")), 0)
            with np.load(Path(ready["outputs"]["videoBundle"]).with_name("inputs.npz"), allow_pickle=False) as inputs:
                self.assertEqual(len(inputs.files), 6)
                self.assertFalse(inputs["structured_available"][:, :, :98].any())
                self.assertFalse(inputs["structured_available"][:, :2].any())
                self.assertTrue(inputs["structured_available"][:, 2:, 98:140].all())
            arrays = Path(ready["outputs"]["videoBundle"]).with_name("inputs.npz")
            arrays.write_bytes(b"changed arrays")
            with self.assertRaisesRegex(HarnessError, "artifact changed"):
                pipeline.transcription_status(self.status_args())
        finally:
            if video_directory is not None and video_directory.exists():
                shutil.rmtree(video_directory)
                video_directory.parent.rmdir()

    def test_failed_video_worker_never_falls_back_to_audio(self):
        checkpoint = self.paired_checkpoint()
        with patch("scripts.transcriber_runtime.load_checkpoint", return_value=checkpoint), patch("scripts.transcription_pipeline.subprocess.run", return_value=subprocess.CompletedProcess([], 7)), patch.object(transcriber, "infer") as infer:
            with self.assertRaisesRegex(HarnessError, "no audio-only substitution"):
                self.run_job(*self.video_args())
        infer.assert_not_called()
        self.assertEqual(self.state()["status"], "blocked")

    def test_joint_checkpoint_with_zero_actual_updates_is_not_trained(self):
        checkpoint = self.paired_checkpoint()
        checkpoint["history"][-1]["optimizer_updates"] = 0
        checkpoint["resume_state"]["optimizer_updates"] = 0
        with patch("scripts.transcriber_runtime.load_checkpoint", return_value=checkpoint), patch("scripts.transcription_pipeline.subprocess.run") as worker:
            with self.assertRaisesRegex(HarnessError, "actual optimizer updates"):
                self.run_job(*self.video_args())
        worker.assert_not_called()

    def test_rgb_checkpoint_stops_before_raw_video_preparation(self):
        checkpoint = self.paired_checkpoint()
        checkpoint["identity"]["video"]["config"]["image_size"] = 96
        with patch("scripts.transcriber_runtime.load_checkpoint", return_value=checkpoint), patch("scripts.transcription_pipeline.subprocess.run") as worker:
            with self.assertRaisesRegex(HarnessError, "not RGB checkpoints"):
                self.run_job(*self.video_args())
        worker.assert_not_called()

    def test_historical_numeric_checkpoint_stops_before_inputs_or_video_preparation(self):
        checkpoint = self.paired_checkpoint()
        checkpoint["identity"]["video"]["config"].update(
            architecture_version=2, input_schema_version=2, structured_dim=98,
        )
        with patch("scripts.transcriber_runtime.load_checkpoint", return_value=checkpoint), patch("scripts.transcription_pipeline.subprocess.run") as worker, patch.object(pipeline, "_identity") as identity:
            with self.assertRaisesRegex(HarnessError, "cannot be resumed or reinterpreted"):
                self.run_job(*self.video_args())
        worker.assert_not_called()
        identity.assert_not_called()

    def test_worker_source_identity_mismatch_is_rejected_before_inference(self):
        checkpoint = self.paired_checkpoint()

        def wrong_result(command, **kwargs):
            request = read_json(command[command.index("--request") + 1])
            self.assertIn("--reset-failed", command)
            output = Path(command[command.index("--output") + 1])
            publish_json(output, {"schemaVersion": 1, "kind": "paired-video-preparation-result", "id": "input",
                                  "status": "needs-review", "inputSha256": {"audio": sha256(request["audio"]), "video": "0" * 64},
                                  "artifacts": {}, "actions": []})
            return subprocess.CompletedProcess(command, 0)

        with patch("scripts.transcriber_runtime.load_checkpoint", return_value=checkpoint), patch("scripts.transcription_pipeline.subprocess.run", side_effect=wrong_result), patch.object(transcriber, "infer") as infer:
            with self.assertRaisesRegex(HarnessError, "differently source-bound"):
                self.run_job(*self.video_args())
        infer.assert_not_called()

    def test_transcribe_forwards_and_binds_confidence_and_rhythm_settings(self):
        result = self.run_job("--rhythm-policy", "fingerstyle", "--strict-note-confidence", "--draft-note-threshold", ".93",
                              "--brush-threshold", ".98", "--brush-membership-threshold", ".9")
        profile = read_json(result["outputs"]["gpReport"])["draftCleanup"]["profile"]
        self.assertEqual(profile["rhythm_policy"], "fingerstyle")
        self.assertTrue(profile["strict_note_confidence"])
        self.assertEqual(profile["note_threshold"], .93)
        self.assertEqual(profile["brush_threshold"], .98)
        self.assertEqual(profile["brush_membership_threshold"], .9)
        with self.assertRaisesRegex(HarnessError, "options changed"):
            self.run_job("--rhythm-policy", "adaptive")

    def test_lower_x_threshold_is_available_before_inference_filtering(self):
        with patch.object(transcriber, "infer", wraps=transcriber.infer) as inference:
            result = self.run_job("--thumb-slap-threshold", ".2", "--draft-percussion-threshold", ".8")
        self.assertEqual(inference.call_args.args[0].percussion_threshold, .2)
        profile = read_json(result["outputs"]["gpReport"])["draftCleanup"]["profile"]
        self.assertEqual(profile["thumb_slap_threshold"], .2)
        self.assertEqual(profile["percussion_threshold"], .8)


if __name__ == "__main__":
    unittest.main()
