from contextlib import ExitStack
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
import wave

import av
import numpy as np

import batch_video as batch
from core import EvidenceError, PRIVATE_OUTPUT_ROOT, sha256
from tests.video.test_hand_roles import hands


class BatchVideoTests(unittest.TestCase):
    def setUp(self):
        self.root = PRIVATE_OUTPUT_ROOT / "batches" / f"batch-test-{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.video = self.root / "source.mkv"
        self.audio = self.root / "trimmed.wav"
        silent_video = self.root / "pictures.mkv"
        source_audio = self.root / "soundtrack.wav"
        rng = np.random.default_rng(42)
        amplitude = np.repeat(rng.uniform(.05, .9, 800), 40)
        waveform = (np.sin(np.arange(32000) * 2 * np.pi * 440 / 8000) * amplitude * 25000).astype(np.int16)
        for path, samples in ((source_audio, waveform), (self.audio, waveform[3200:22400])):
            with wave.open(str(path), "wb") as stream:
                stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                stream.writeframes(samples.tobytes())
        image = rng.integers(20, 100, (180, 320, 3), dtype=np.uint8)
        with av.open(str(silent_video), "w") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p"
            for i in range(100):
                frame = av.VideoFrame.from_ndarray(image if i < 50 else 255 - image, format="rgb24")
                frame.pts, frame.time_base = i * 40, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        subprocess.run([str(batch._installed_tools()[0]), "-v", "error", "-i", str(silent_video),
                        "-i", str(source_audio), "-c", "copy", str(self.video)], check=True, capture_output=True)
        self.gp, self.model = self.root / "raw.gp", self.root / "hand.task"
        self.gp.write_bytes(b"opaque source identity; never decoded as features")
        self.model.write_bytes(b"synthetic detector model")
        self.request = self.root / "request.json"
        self.config = {
            "schemaVersion": 1, "kind": "paired-video-preparation-request", "id": "synthetic",
            "video": str(self.video), "audio": str(self.audio),
            "outputDirectory": str(self.root / "worker"),
            "reviewMode": "manual",
            "pluckingScreenSide": "left", "handModel": str(self.model),
        }
        self.write_request()
        self.detector_times = []
        self.tracker_count = 0

        def tracker(*_args, **_kwargs):
            self.tracker_count += 1
            points = [[SimpleNamespace(x=float(p[0]), y=float(p[1]), z=0.) for p in hand] for hand in hands()]
            result = SimpleNamespace(hand_landmarks=points, hand_world_landmarks=points,
                                     handedness=[[SimpleNamespace(category_name="Left", score=.9)],
                                                 [SimpleNamespace(category_name="Right", score=.9)]])

            def detect(_image, timestamp):
                self.detector_times.append(timestamp)
                return result

            return SimpleNamespace(detect_for_video=detect, close=lambda: None)

        self.tracker_patch = patch("hand_tracking._hand_landmarker", side_effect=tracker)
        self.tracker_patch.start()
        self.addCleanup(self.tracker_patch.stop)

    def write_request(self):
        self.request.write_text(json.dumps(self.config), encoding="utf-8")

    def read(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def ready(self):
        result = batch.run_request(self.request)
        self.assertEqual(result["actions"][0]["stage"], "shots")
        self.assertFalse(self.detector_times)
        return batch.review_request(self.request, accept_shots=True)

    def adopt(self, result, *, name="adopted", clips=None):
        self.config["outputDirectory"] = str(self.root / name)
        self.config["reuse"] = dict(result["artifacts"])
        if clips is not None:
            self.config["clips"] = clips
            self.config["reuse"].pop("bundle")
        self.request = self.root / f"{name}-request.json"
        self.write_request()

    def test_fresh_reviews_exact_audio_range_real_downstream_and_zero_work_resume(self):
        original = sha256(self.video)
        with patch("builtins.print") as output:
            result = self.ready()
        messages = [call.args[0] for call in output.call_args_list if call.args and str(call.args[0]).startswith("synthetic:")]
        self.assertIn("synthetic: inputs: verified.", messages)
        self.assertIn("synthetic: hands: processing...", messages)
        self.assertIn("synthetic: hands: completed (processed).", messages)
        self.assertIn("synthetic: bundle: completed (processed).", messages)
        self.assertTrue(all(call.kwargs.get("flush") for call in output.call_args_list if call.args and str(call.args[0]).startswith("synthetic:")))
        self.assertEqual(result["status"], "ready")
        self.assertEqual(set(result["artifacts"]), set(batch.STAGES) - {"fretboard"})
        shots = self.read(result["artifacts"]["shots"])
        self.assertEqual((shots["firstPts"], shots["lastPts"], shots["frameCount"]), (400, 2760, 60))
        self.assertEqual(shots["shots"][-1]["endPtsExclusive"], 2800)
        self.assertEqual(shots["shots"][0]["boundaryType"], "range_start")
        self.assertEqual(len(self.detector_times), 60)
        self.assertEqual(self.tracker_count, shots["shotCount"])
        self.assertEqual(sha256(self.video), original)
        bundle = self.read(result["artifacts"]["bundle"])
        self.assertEqual(bundle["clips"], [{"startPts": 400, "endPtsExclusive": 2800}])
        self.assertNotIn("coarse", result["artifacts"])
        with np.load(Path(result["artifacts"]["bundle"]).with_name("inputs.npz")) as arrays:
            self.assertEqual(set(arrays.files), {"structured", "structured_available", "pts", "audio_seconds", "technique_available", "segment_id"})
            self.assertEqual(arrays["structured"].shape, (60, 4, 194))
            self.assertEqual(arrays["structured_available"].shape, (60, 4, 194))
            self.assertTrue(np.all(arrays["audio_seconds"] >= 0))
            self.assertTrue(np.all(arrays["audio_seconds"] < 2.4))
            self.assertNotIn("targets", arrays.files)
        with ExitStack() as stack:
            output = stack.enter_context(patch("builtins.print"))
            calls = [stack.enter_context(patch.object(batch, name, wraps=getattr(batch, name))) for name in
                     ("align_soundtrack", "inspect_shots", "track_hands",
                      "assign_hand_roles", "prepare_paired_inputs", "_timeline", "_select_interval")]
            resumed = batch.run_request(self.request)
            self.assertEqual(resumed["status"], "ready")
            self.assertTrue(all(item["status"] == "reused" for item in resumed["stageSummary"]))
            self.assertFalse(any(call.called for call in calls))
            self.assertTrue(any(call.args[0] == "synthetic: bundle: reused verified output." for call in output.call_args_list))
        self.assertFalse((Path(self.config["outputDirectory"]) / ".worker.lock").exists())

    def test_unknown_roles_preserve_independent_hand_evidence(self):
        self.config["reviewMode"] = "automatic"
        self.config["pluckingScreenSide"] = "geometry"
        self.write_request()
        result = batch.run_request(self.request)
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["actions"])
        with np.load(Path(result["artifacts"]["bundle"]).with_name("inputs.npz")) as arrays:
            self.assertFalse(arrays["structured_available"][..., :98].any())
            self.assertTrue(arrays["structured_available"][:, 2:, 98:140].any())
            self.assertFalse(arrays["structured_available"][:, :2].any())

    def test_fast_preparation_skips_experimental_geometry_and_retains_hands(self):
        self.config.update(reviewMode="automatic")
        self.write_request()
        result = batch.run_request(self.request)
        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["actions"])
        self.assertNotIn("coarse", result["artifacts"])
        self.assertFalse(self.read(result["artifacts"]["annotations"])["reviewComplete"])
        with np.load(Path(result["artifacts"]["bundle"]).with_name("inputs.npz")) as arrays:
            self.assertTrue(arrays["structured_available"][:, :2, 98:140].any())
            self.assertFalse(arrays["structured_available"][..., :98].any())
            self.assertFalse(arrays["structured_available"][..., 186:].any())

    def test_fretboard_model_adds_resumable_stage_and_schema_five_bundle(self):
        fretboard_model = self.root / "fretboard.pt"
        fretboard_model.write_bytes(b"synthetic six-point detector")
        self.config.update(
            reviewMode="automatic", fretboardModel=str(fretboard_model),
            fretboardDevice="0",
        )
        self.write_request()

        def track(_video, shots_path, model_path, output, *, detector_config):
            self.assertEqual(detector_config.device, "0")
            shots = self.read(shots_path)
            pts = np.asarray(shots["framePts"], np.int64)
            shot_ids = np.zeros(len(pts), np.int32)
            for row in shots["shots"]:
                shot_ids[(pts >= row["startPts"]) & (pts < row["endPtsExclusive"])] = row["shotId"]
            output.mkdir()
            arrays = output / "fretboard.npz"
            np.savez_compressed(
                arrays, pts=pts, shot_id=shot_ids,
                keypoints=np.full((len(pts), 3, 2, 2), np.nan, np.float32),
                available=np.zeros((len(pts), 3, 2), bool),
                confidence=np.zeros(len(pts), np.float32),
                source=np.zeros(len(pts), np.int8),
                age_seconds=np.zeros(len(pts), np.float32),
                flow_error=np.zeros(len(pts), np.float32),
                detector_anchor=np.zeros(len(pts), bool),
            )
            report = {
                "schemaVersion": 1, "kind": "six-point-fretboard-observations",
                "videoSha256": sha256(self.video), "shotsSha256": sha256(shots_path),
                "modelSha256": sha256(model_path), "timeBase": shots["timeBase"],
                "frameCount": len(pts), "shotCount": len(shots["shots"]),
                "sourceEncoding": {"unavailable": 0, "detector": 1, "optical_flow": 2},
                "arrays": "fretboard.npz", "arraysSha256": sha256(arrays),
            }
            path = output / "fretboard.json"
            path.write_text(json.dumps(report))
            return path, report

        with patch.object(batch, "track_fretboard", side_effect=track):
            result = batch.run_request(self.request)
        self.assertEqual(result["status"], "ready")
        self.assertIn("fretboard", result["artifacts"])
        bundle = self.read(result["artifacts"]["bundle"])
        self.assertEqual(bundle["schemaVersion"], 5)
        self.assertEqual(bundle["featureDimension"], 233)
        with np.load(Path(result["artifacts"]["bundle"]).with_name("inputs.npz")) as arrays:
            self.assertEqual(arrays["structured"].shape[-1], 233)

    def test_reuse_full_chain_and_multiple_clips_never_redetect(self):
        result = self.ready()
        count = len(self.detector_times)
        self.adopt(result)
        with patch.object(batch, "track_hands", wraps=batch.track_hands) as detector:
            adopted = batch.run_request(self.request)
            self.assertEqual(adopted["actions"][0]["stage"], "shots")
            adopted = batch.review_request(self.request, accept_shots=True)
        self.assertEqual(adopted["status"], "ready")
        self.assertFalse(detector.called)
        self.assertEqual(adopted["artifacts"]["bundle"], result["artifacts"]["bundle"])
        self.adopt(result, name="clips", clips=[[400, 640], [2000, 2240]])
        selected = batch.run_request(self.request)
        self.assertEqual(selected["actions"][0]["stage"], "shots")
        selected = batch.review_request(self.request, accept_shots=True)
        self.assertEqual(selected["status"], "ready")
        self.assertEqual(len(self.detector_times), count)
        with np.load(Path(selected["artifacts"]["bundle"]).with_name("inputs.npz")) as arrays:
            np.testing.assert_array_equal(arrays["pts"], [400, 440, 480, 520, 560, 600, 2000, 2040, 2080, 2120, 2160, 2200])

    def test_full_range_reuses_observations_when_audio_ends_between_video_frames(self):
        result = self.ready()
        with wave.open(str(self.audio), "rb") as source:
            parameters, pcm = source.getparams(), source.readframes(19040)
        shorter = self.root / "shorter.wav"
        with wave.open(str(shorter), "wb") as output:
            output.setparams(parameters)
            output.writeframes(pcm)
        self.config["audio"] = str(shorter)
        self.adopt(result, name="shorter")
        for name in ("alignment", "bundle"):
            self.config["reuse"].pop(name)
        self.write_request()
        with patch.object(batch, "track_hands", side_effect=AssertionError("Do not repeat detection")):
            result = batch.run_request(self.request)
            self.assertEqual(result["actions"][0]["stage"], "shots")
            result = batch.review_request(self.request, accept_shots=True)
        self.assertEqual(result["status"], "ready")
        report = self.read(result["artifacts"]["bundle"])
        self.assertEqual(report["clips"], [{"startPts": 400, "endPtsExclusive": 2780}])
        with np.load(Path(result["artifacts"]["bundle"]).with_name("inputs.npz")) as arrays:
            self.assertEqual(arrays["pts"][-1], 2760)

    def test_sub_tick_audio_tail_excludes_only_the_unrepresentable_final_frame(self):
        result = self.ready()
        with wave.open(str(self.audio), "rb") as source:
            parameters, pcm = source.getparams(), source.readframes(18881)
        shorter = self.root / "sub-tick.wav"
        with wave.open(str(shorter), "wb") as output:
            output.setparams(parameters)
            output.writeframes(pcm)
        self.config["audio"] = str(shorter)
        self.adopt(result, name="sub-tick")
        for name in ("alignment", "bundle"):
            self.config["reuse"].pop(name)
        self.write_request()
        result = batch.run_request(self.request)
        self.assertEqual(result["actions"][0]["stage"], "shots")
        result = batch.review_request(self.request, accept_shots=True)
        self.assertEqual(result["status"], "ready")
        report = self.read(result["artifacts"]["bundle"])
        self.assertEqual(report["clips"], [{"startPts": 400, "endPtsExclusive": 2760}])
        with np.load(Path(result["artifacts"]["bundle"]).with_name("inputs.npz")) as arrays:
            self.assertEqual(arrays["pts"][-1], 2720)

    def test_changed_import_array_hash_is_rejected_before_any_processing(self):
        result = self.ready()
        self.adopt(result)
        arrays = Path(result["artifacts"]["roles"]).with_name("roles.npz")
        with arrays.open("ab") as stream:
            stream.write(b"changed")
        with patch.object(batch, "align_soundtrack", wraps=batch.align_soundtrack) as alignment:
            with self.assertRaisesRegex(EvidenceError, "arrays"):
                batch.run_request(self.request)
            self.assertFalse(alignment.called)

    def test_placeholder_edit_after_complete_is_stale(self):
        result = self.ready()
        path = Path(result["artifacts"]["annotations"])
        document = self.read(path)
        document["shots"][0]["reviewNote"] = "changed downstream dependency"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "Stale annotations"):
            batch.run_request(self.request)

    def test_changed_source_and_configuration_block_resume(self):
        batch.run_request(self.request)
        original = self.audio.read_bytes()
        self.audio.write_bytes(b"revised audio")
        with self.assertRaisesRegex(EvidenceError, "Stale input"):
            batch.run_request(self.request)
        self.audio.write_bytes(original)
        self.config["pluckingScreenSide"] = "right"
        self.write_request()
        with self.assertRaisesRegex(EvidenceError, "Stale input"):
            batch.run_request(self.request)

    def test_failure_partial_is_never_adopted_and_explicit_reset_preserves_it(self):
        batch.run_request(self.request)
        original = batch.track_hands

        def broken(*args, **kwargs):
            directory = Path(args[3])
            directory.mkdir()
            (directory / "partial.txt").write_text("failed attempt", encoding="utf-8")
            raise EvidenceError("synthetic detector failure")

        with patch.object(batch, "track_hands", side_effect=broken), patch("builtins.print") as log:
            with self.assertRaisesRegex(EvidenceError, "synthetic detector failure"):
                batch.review_request(self.request, accept_shots=True)
        messages = [call.args[0] for call in log.call_args_list]
        self.assertIn("synthetic: hands: failed: synthetic detector failure", messages)
        self.assertNotIn("synthetic: hands: completed (processed).", messages)
        output = Path(self.config["outputDirectory"])
        self.assertFalse((output / ".worker.lock").exists())
        with patch.object(batch, "track_hands", wraps=original) as detector:
            with self.assertRaisesRegex(EvidenceError, "reset-failed"):
                batch.run_request(self.request)
            self.assertFalse(detector.called)
            result = batch.run_request(self.request, reset_failed=True)
            self.assertEqual(result["status"], "ready")
            self.assertTrue(detector.called)
        self.assertEqual(len(list(output.glob("hands.failed-*/partial.txt"))), 1)

    def test_shot_review_add_cut_reinspects_preserves_source_boundary_and_receipt(self):
        batch.run_request(self.request)
        with patch("builtins.print"):
            self.assertEqual(batch.main(["review", "--request", str(self.request), "--accept-shots",
                                         "--add-cut", "1.2"]), 0)
        result = self.read(self.request.with_name("result.json"))
        self.assertEqual(result["status"], "ready")
        shots = self.read(result["artifacts"]["shots"])
        self.assertIn(1200, [row["startPts"] for row in shots["shots"]])
        receipt = self.read(Path(self.config["outputDirectory"]) / "workerreviewreceipt.json")
        self.assertEqual(receipt["shots"]["addedCutSeconds"], [1.2])
        self.assertEqual(receipt["shots"]["reviewedShotsSha256"], sha256(receipt["shots"]["path"]))
        self.assertEqual(self.tracker_count, shots["shotCount"])

    def test_ambiguous_alignment_manual_review_keeps_original_digest(self):
        output = self.root / "supplied-alignment.json"
        _, document = batch.align_soundtrack(self.video, self.audio, output)
        document.update(status="ambiguous", correlation=.4, ambiguityRatio=.99)
        output.write_text(json.dumps(document), encoding="utf-8")
        self.config["reuse"] = {"alignment": str(output)}
        self.write_request()
        result = batch.run_request(self.request)
        self.assertEqual(result["actions"][0]["stage"], "alignment")
        digest = sha256(output)
        result = batch.review_request(self.request, alignment_offset=.4)
        self.assertEqual(result["actions"][0]["stage"], "shots")
        reviewed = self.read(result["artifacts"]["alignment"])
        self.assertEqual(reviewed["method"], "manual-timestamp")
        self.assertEqual(reviewed["originalAlignmentSha256"], digest)
        self.assertEqual(sha256(output), digest)
        with self.assertRaisesRegex(EvidenceError, "already reviewed differently"):
            batch.review_request(self.request, alignment_offset=.3)

    def test_malformed_clips_and_foreign_reuse_dependency_fail_early(self):
        for clips in ([[0, 0]], [[False, 40]], [[400, 800], [600, 1000]], [], "all"):
            with self.subTest(clips=clips):
                self.config["clips"] = clips
                self.write_request()
                with self.assertRaises(EvidenceError):
                    batch.run_request(self.request)
        self.config.pop("clips")
        self.write_request()
        result = self.ready()
        self.adopt(result, name="wrong-bound")
        alignment = self.read(result["artifacts"]["alignment"])
        alignment["trimmedAudioSha256"] = "0" * 64
        foreign = self.root / "foreign-alignment.json"
        foreign.write_text(json.dumps(alignment), encoding="utf-8")
        self.config["reuse"]["alignment"] = str(foreign)
        self.write_request()
        with self.assertRaisesRegex(EvidenceError, "trimmed audio"):
            batch.run_request(self.request)

    def test_output_alias_fails_before_heavy_work_and_blocked_result_is_explicit(self):
        with patch.object(batch, "align_soundtrack", wraps=batch.align_soundtrack) as alignment:
            self.assertEqual(batch.main(["run", "--request", str(self.request), "--output", str(self.audio)]), 1)
            self.assertFalse(alignment.called)
        self.config["clips"] = [[1, 1]]
        self.write_request()
        result_path = self.root / "result.json"
        self.assertEqual(batch.main(["run", "--request", str(self.request), "--output", str(result_path)]), 1)
        self.assertEqual(self.read(result_path)["status"], "blocked")
        self.assertEqual(self.gp.read_bytes(), b"opaque source identity; never decoded as features")

    def test_unsafe_keyframe_paths_are_rejected(self):
        result = self.ready()
        path = Path(result["artifacts"]["annotations"])
        document = self.read(path)
        for image in (str(self.audio), "..\\outside.jpg"):
            document["shots"][0]["keyframeImage"] = image
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(EvidenceError, "local relative"):
                batch._artifacts(path, "annotations")

    def test_non_source_clip_end_and_outside_audio_fail_before_detection(self):
        for name, clips in (("between", [[400, 801]]), ("before", [[0, 400]])):
            self.config.update(outputDirectory=str(self.root / name), clips=clips)
            self.write_request()
            batch.run_request(self.request)
            with self.assertRaisesRegex(EvidenceError, "aligned source frames"):
                batch.review_request(self.request, accept_shots=True)
            self.assertFalse(self.detector_times)

    def test_fresh_multiple_clips_detects_only_envelope_and_packages_only_clips(self):
        self.config["clips"] = [[400, 640], [2000, 2240]]
        self.write_request()
        result = self.ready()
        self.assertEqual(len(self.detector_times), 46)
        with np.load(Path(result["artifacts"]["bundle"]).with_name("inputs.npz")) as arrays:
            self.assertEqual(len(arrays["pts"]), 12)
        shots = self.read(result["artifacts"]["shots"])
        self.assertEqual(shots["requestedClips"], [
            {"startPts": 400, "endPtsExclusive": 640}, {"startPts": 2000, "endPtsExclusive": 2240}])

    def test_final_video_frame_uses_real_duration_not_last_pts_plus_one(self):
        shutil.copyfile(self.root / "soundtrack.wav", self.audio)
        result = self.ready()
        shots = self.read(result["artifacts"]["shots"])
        self.assertEqual((shots["lastPts"], shots["shots"][-1]["endPtsExclusive"]), (3960, 4000))
        self.assertEqual(shots["frameCount"], 100)

    def test_audio_end_between_source_frames_uses_exact_exclusive_bound(self):
        with wave.open(str(self.root / "soundtrack.wav"), "rb") as stream:
            samples = stream.readframes(32000)[3200 * 2:22440 * 2]
        with wave.open(str(self.audio), "wb") as stream:
            stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            stream.writeframes(samples)
        result = self.ready()
        bundle = self.read(result["artifacts"]["bundle"])
        self.assertEqual(bundle["clips"], [{"startPts": 400, "endPtsExclusive": 2805}])
        with np.load(Path(result["artifacts"]["bundle"]).with_name("inputs.npz")) as arrays:
            self.assertEqual(arrays["pts"][-1], 2800)
            self.assertLess(arrays["audio_seconds"][-1], 2.405)

    def test_single_writer_lock_and_unrecorded_partial_are_not_adopted(self):
        output = Path(self.config["outputDirectory"])
        output.mkdir()
        lock = output / ".worker.lock"
        lock.write_text("12345", encoding="ascii")
        with self.assertRaisesRegex(EvidenceError, "Worker locked"):
            batch.run_request(self.request)
        self.assertEqual(lock.read_text(), "12345")
        lock.unlink()
        (output / "hands").mkdir()
        with self.assertRaisesRegex(EvidenceError, "without a worker ledger"):
            batch.run_request(self.request, reset_failed=True)
        self.assertTrue((output / "hands").is_dir())

    def test_identical_media_reuses_alignment_inspection_cache_not_review_approval(self):
        self.config["outputDirectory"] = str(self.root / "first" / "video")
        self.write_request()
        first = batch.run_request(self.request)
        self.config.update(id="second", outputDirectory=str(self.root / "second" / "video"))
        self.request = self.root / "second-request.json"
        self.write_request()
        with patch.object(batch, "align_soundtrack", wraps=batch.align_soundtrack) as alignment, \
                patch.object(batch, "inspect_shots", wraps=batch.inspect_shots) as inspection:
            second = batch.run_request(self.request)
        self.assertFalse(alignment.called)
        self.assertFalse(inspection.called)
        self.assertEqual(second["artifacts"]["alignment"], first["artifacts"]["alignment"])
        self.assertEqual(second["actions"][0]["stage"], "shots")
        self.assertTrue(all(row["provenance"] == "cached" for row in second["stageSummary"] if row["status"] == "reused"))

    def test_parent_layout_rebuilds_bundle_once_then_publishes_reused_result(self):
        prepared = self.ready()
        self.adopt(prepared, name="parent-record")
        self.config["outputDirectory"] = str(self.root / "parent-record" / "video")
        self.config["reuse"].pop("bundle")
        self.request = self.root / "parent-record" / "request.json"
        self.request.parent.mkdir()
        self.write_request()
        result_path = self.request.with_name("result.json")
        with patch.object(batch, "prepare_paired_inputs", wraps=batch.prepare_paired_inputs) as package, \
                patch.object(batch, "track_hands", wraps=batch.track_hands) as detector, patch("builtins.print"):
            for _ in range(2):
                self.assertEqual(batch.main(["run", "--request", str(self.request), "--output", str(result_path)]), 0)
                result = self.read(result_path)
                if result["status"] == "needs-review":
                    self.assertEqual(result["actions"][0]["stage"], "shots")
                    self.assertEqual(batch.main(["review", "--request", str(self.request), "--output", str(result_path), "--accept-shots"]), 0)
                    result = self.read(result_path)
                self.assertEqual(result["status"], "ready")
                self.assertEqual(Path(result["artifacts"]["bundle"]), Path(self.config["outputDirectory"]) / "bundle" / "inputs.json")
            self.assertEqual(package.call_count, 1)
            self.assertFalse(detector.called)
        self.assertTrue(all(row["status"] == "reused" for row in result["stageSummary"]))

    def test_inference_without_gp_uses_same_pipeline_and_omits_gp_identity(self):
        self.gp.unlink()
        self.write_request()
        result = self.ready()
        self.assertEqual(result["inputSha256"], {"video": sha256(self.video), "audio": sha256(self.audio)})
        bundle = self.read(result["artifacts"]["bundle"])
        self.assertNotIn("correspondenceReferenceGp", bundle["inputPaths"])
        self.assertNotIn("correspondenceReferenceGp", bundle["inputSha256"])
        with patch.object(batch, "prepare_paired_inputs", wraps=batch.prepare_paired_inputs) as package:
            resumed = batch.run_request(self.request)
            self.assertEqual(resumed["status"], "ready")
            self.assertEqual(resumed["inputSha256"], result["inputSha256"])
            self.assertFalse(package.called)

    def test_placeholder_image_is_immutable_and_receipt_mutation_is_stale(self):
        batch.run_request(self.request)
        result = batch.review_request(self.request, accept_shots=True)
        output = Path(self.config["outputDirectory"])
        receipt_path = output / "workerreviewreceipt.json"
        original = receipt_path.read_bytes()
        receipt = self.read(receipt_path)
        receipt["shots"]["reviewedShotsSha256"] = "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "Stale review receipt"):
            batch.run_request(self.request)
        receipt_path.write_bytes(original)
        annotations = self.read(result["artifacts"]["annotations"])
        image = Path(result["artifacts"]["annotations"]).parent / annotations["shots"][0]["keyframeImage"]
        with image.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(EvidenceError, "Stale annotations"):
            batch.run_request(self.request)

    def test_source_mutated_during_stage_never_gets_complete_status(self):
        original = batch.align_soundtrack

        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            self.audio.write_bytes(b"changed while aligning")
            return result

        with patch.object(batch, "align_soundtrack", side_effect=mutate):
            with self.assertRaisesRegex(EvidenceError, "Stale source"):
                batch.run_request(self.request)
        ledger = self.read(Path(self.config["outputDirectory"]) / "status.json")
        self.assertEqual(ledger["stages"]["alignment"]["status"], "failed")

    def test_input_and_output_hardlink_aliases_are_refused(self):
        alias = self.root / "audio-alias.wav"
        alias.hardlink_to(self.audio)
        with self.assertRaisesRegex(EvidenceError, "Hardlinked"):
            batch.run_request(self.request)

    def test_experimental_configuration_is_removed_not_a_disable_flag(self):
        for field, value in (("imageSize", 96), ("geometryMode", "automatic"), ("coarseContext", True),
                             ("geometryReference", str(self.gp)), ("correspondenceReview", str(self.gp)),
                             ("sourceMapping", str(self.gp)), ("videoReceipt", str(self.gp)), ("gp", str(self.gp))):
            self.config[field] = value
            self.write_request()
            with self.subTest(field=field), patch.object(batch, "align_soundtrack", wraps=batch.align_soundtrack) as align:
                with self.assertRaisesRegex(EvidenceError, "Unknown video request fields"):
                    batch.run_request(self.request)
                self.assertFalse(align.called)
            self.config.pop(field)


if __name__ == "__main__":
    unittest.main()
