from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

from scripts.dataset_io import ROOT, read_json, sha256
from scripts.dataset_release import validate_release
from scripts.paired_preparation import _lock, batch_status, load_batch, paired_coverage, review_batch, run_batch, train_batch
from scripts import prepare_training_data
from scripts.paired_video import build_index, load_inference_video
from tests.test_dataset_release import synthetic_release
from tests.test_paired_video import bundle_fixture


class PairedBatchTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="batch-preparation-test-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.release = synthetic_release(self.root / "release")
        _, entries, _ = validate_release(self.release)
        rows, self.canonical_records, self.bundles = [], [], {}
        for entry, _ in entries:
            identifier = entry["id"]
            audio = self.release.parent / entry["audioPath"]
            bundle, report, _ = bundle_fixture(self.root, audio, identifier=identifier)
            video = Path(report["inputPaths"]["video"])
            video.write_text(identifier + "-synthetic-video")
            alignment = Path(report["inputPaths"]["alignment"])
            alignment_document = read_json(alignment)
            alignment_document["videoSha256"] = sha256(video)
            alignment.write_text(json.dumps(alignment_document))
            report["videoSha256"] = sha256(video)
            report["inputSha256"].update(video=sha256(video), alignment=sha256(alignment))
            bundle.write_text(json.dumps(report))
            gp = self.root / (identifier + ".gp")
            gp.write_bytes(identifier.encode())
            row = {
                "id": identifier, "groupId": entry["groupId"], "split": entry["split"],
                "gp": str(gp), "audio": str(audio), "video": report["inputPaths"]["video"],
                "reuse": {"bundle": str(bundle)}, "clips": [[1000, 1160]], "pluckingScreenSide": "right",
            }
            rows.append(row)
            self.canonical_records.append({**row, "targets": str(self.release.parent / entry["targetsPath"])})
            self.bundles[identifier] = bundle
        self.document = {
            "schemaVersion": 1, "kind": "paired-preparation-batch", "records": rows,
            "workspace": str(self.root / "workspace"), "releaseManifest": str(self.release),
            "videoPython": sys.executable, "handModel": "models\\hand_landmarker.task",
        }
        self.manifest = self.root / "batch.json"
        self.save()
        self.output = self.root / "runs" / "video-evidence" / "batches" / "pilot"
        self.calls = []

    def save(self):
        self.manifest.write_text(json.dumps(self.document), encoding="utf-8")

    def canonical(self, batch, *, root):
        return {"status": "ready", "manifestPath": str(self.release), "records": self.canonical_records, "actions": []}

    def worker(self, command, **kwargs):
        self.calls.append(command)
        request = read_json(Path(command[command.index("--request") + 1]))
        self.assertEqual(set(request), {"schemaVersion", "kind", "id", "video", "audio", "outputDirectory", "pluckingScreenSide", "clips", "reuse", "handModel", "poseModel", "fretboardModel", "fretboardDevice", "reviewMode"})
        path = Path(command[command.index("--output") + 1])
        result = {
            "schemaVersion": 1, "kind": "paired-video-preparation-result", "id": request["id"], "status": "ready",
            "inputSha256": {key: sha256(request[key]) for key in ("video", "audio")},
            "artifacts": {"bundle": str(self.bundles[request["id"]])}, "actions": [],
            "stageSummary": [{"stage": "bundle", "status": "reused"}],
        }
        path.write_text(json.dumps(result))
        return SimpleNamespace(returncode=0)

    def run_batch(self):
        with patch.dict("sys.modules", {"scripts.batch_canonical": SimpleNamespace(ensure_canonical=self.canonical)}):
            return run_batch(self.manifest, self.output, root=self.root, runner=self.worker, progress=lambda message: None)

    def test_repeat_run_retains_index_sources_splits_and_distinguishes_coverage_from_quality(self):
        source_hashes = {name: sha256(path) for row in self.document["records"] for name, path in ((row["id"] + key, row[key]) for key in ("gp", "audio", "video"))}
        messages = []
        with patch.dict("sys.modules", {"scripts.batch_canonical": SimpleNamespace(ensure_canonical=self.canonical)}):
            result = run_batch(self.manifest, self.output, root=self.root, runner=self.worker, progress=messages.append)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(messages[0], "Loading batch manifest...")
        self.assertIn("Canonical GP/audio inputs ready.", messages)
        self.assertIn("Building paired index...", messages)
        self.assertIn("Paired visual/target coverage counted.", messages)
        self.assertTrue(messages[-1].startswith("Batch ready; summary saved:"))
        self.assertFalse(result["trainingPerformed"])
        self.assertEqual({row["split"] for row in read_json(Path(result["indexPath"]))["records"]}, {"train", "validation"})
        index_hash = sha256(Path(result["indexPath"]))
        again = self.run_batch()
        self.assertEqual(sha256(Path(again["indexPath"])), index_hash)
        self.assertEqual(result["coverage"], again["coverage"])
        self.assertTrue(result["coverage"]["missingUsableTrainingPositiveClasses"])
        self.assertTrue(Path(result["nextActionsPath"]).is_file())
        self.assertTrue(all("train" not in command for command in self.calls))
        for row in self.document["records"]:
            for key in ("gp", "audio", "video"):
                self.assertEqual(sha256(row[key]), source_hashes[row["id"] + key])

    def test_changed_source_or_manifest_never_silently_reuses_batch(self):
        self.run_batch()
        self.document["records"][0]["pluckingScreenSide"] = "left"
        self.save()
        with self.assertRaisesRegex(ValueError, "configuration or source"):
            self.run_batch()
        self.document["records"][0]["pluckingScreenSide"] = "right"
        self.save()
        Path(self.document["records"][0]["video"]).write_text("changed video")
        with self.assertRaisesRegex(ValueError, "configuration or source"):
            self.run_batch()

    def test_public_manifest_rejects_removed_source_and_geometry_surfaces(self):
        for key in ("sourceMapping", "geometryMode", "coarseContext", "acceptOwnerConventions"):
            with self.subTest(key=key):
                self.document[key] = "obsolete"
                self.save()
                with self.assertRaises(ValueError):
                    load_batch(self.manifest, root=self.root)
                del self.document[key]
        for key in ("videoReceipt", "geometryReference", "correspondenceReview"):
            with self.subTest(key=key):
                self.document["records"][0][key] = "obsolete"
                self.save()
                with self.assertRaises(ValueError):
                    load_batch(self.manifest, root=self.root)
                del self.document["records"][0][key]
        del self.document["records"][0]["gp"]
        self.save()
        with self.assertRaisesRegex(ValueError, "gp"):
            load_batch(self.manifest, root=self.root)

    def test_review_pause_is_persistent_and_resumes_with_same_inputs(self):
        with patch.dict("sys.modules", {"scripts.batch_canonical": SimpleNamespace(ensure_canonical=lambda *args, **kwargs: {"status": "needs-review", "actions": [{"stage": "canonical-review", "reason": "Explicit ranges needed"}]})}):
            result = run_batch(self.manifest, self.output, root=self.root, runner=self.worker, progress=lambda message: None)
        self.assertEqual(result["status"], "needs-review")
        self.assertEqual(self.calls, [])
        self.assertEqual(batch_status(self.manifest, self.output, root=self.root)["actions"], result["actions"])
        self.assertEqual(self.run_batch()["status"], "ready")

    def test_invalid_group_split_clip_and_changed_worker_identity_rejected(self):
        changed = deepcopy(self.document)
        changed["records"][1]["groupId"] = changed["records"][0]["groupId"]
        self.manifest.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "group cannot cross"):
            load_batch(self.manifest, root=self.root)
        self.save()
        self.document["records"][0]["clips"] = [[1000, 1100], [1050, 1200]]
        self.save()
        with self.assertRaisesRegex(ValueError, "Clips"):
            load_batch(self.manifest, root=self.root)

    def test_failed_worker_never_publishes_ready_or_leaves_lock(self):
        with patch.dict("sys.modules", {"scripts.batch_canonical": SimpleNamespace(ensure_canonical=self.canonical)}):
            with self.assertRaisesRegex(ValueError, "exit code 2"):
                run_batch(self.manifest, self.output, root=self.root, runner=lambda *args, **kwargs: SimpleNamespace(returncode=2), progress=lambda message: None)
        self.assertFalse((self.output / ".batch.lock").exists())
        self.assertFalse((self.output / "paired-index.json").exists())
        self.assertEqual(read_json(self.output / "summary.json")["status"], "blocked")
        self.assertEqual(self.run_batch()["status"], "ready")

    def test_concurrent_batch_lock_cannot_be_stolen(self):
        self.output.mkdir(parents=True)
        with _lock(self.output):
            with self.assertRaisesRegex(ValueError, "already running"):
                with _lock(self.output):
                    self.fail("Second lock acquired")
            self.assertTrue((self.output / ".batch.lock").exists())
        self.assertFalse((self.output / ".batch.lock").exists())

    def test_identical_source_video_cannot_cross_split_even_under_different_names(self):
        source, copied = (Path(row["video"]) for row in self.document["records"])
        copied.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "source video"):
            self.run_batch()
        self.assertEqual(self.calls, [])

    def test_coverage_uses_exact_frame_mapping_without_materializing_window_tensors(self):
        index, _ = build_index(self.release, list(self.bundles.values()), self.root / "index.json", root=self.root)
        with patch("scripts.paired_video._Bundle.window", side_effect=AssertionError("No window tensors in coverage report")):
            result = paired_coverage(self.release, index, root=self.root)
        self.assertEqual(sum(row["recordings"] for row in result["splits"].values()), 2)
        video = load_inference_video(next(iter(self.bundles.values())), read_json(next(iter(self.bundles.values())))["audioSha256"])
        times = [.98, 1., 1.02, 1.16]
        available, technique = video.availability_at(times)
        sample = video.window(times)
        expected = sample["frame_indices"].numpy() >= 0
        np.testing.assert_array_equal(available.any(-1), expected)
        np.testing.assert_array_equal(technique, expected)

    def test_removed_rgb_configuration_is_rejected_explicitly(self):
        self.document["imageSize"] = 96
        self.save()
        with self.assertRaisesRegex(ValueError, "RGB crop inputs are no longer supported"):
            load_batch(self.manifest, root=self.root)

    def test_public_commands_dispatch_into_single_preparation_workflow(self):
        from io import StringIO

        base = ["--manifest", str(self.manifest), "--output-directory", str(self.output)]
        with patch("scripts.paired_preparation.run_batch", return_value={"status": "needs-review"}) as run, patch("sys.stdout", new=StringIO()):
            self.assertEqual(prepare_training_data.main(["batch", *base]), 0)
            run.assert_called_once_with(self.manifest, self.output)
        with patch("scripts.paired_preparation.review_batch", return_value={"status": "reviewed"}) as review, patch("sys.stdout", new=StringIO()):
            self.assertEqual(prepare_training_data.main(["batch-review", *base, "--id", "piece-0", "--accept-shots", "--add-cut", "1.25"]), 0)
            self.assertEqual(review.call_args.kwargs["video_flags"], ["--accept-shots", "--add-cut", "1.25"])
        with patch("scripts.paired_preparation.finalize_batch", return_value={"manifestPath": str(self.release)}) as finalize, patch("sys.stdout", new=StringIO()):
            self.assertEqual(prepare_training_data.main(["batch-release", *base, "--reviewer", "owner"]), 0)
            finalize.assert_called_once_with(self.manifest, self.output, "owner")
        with patch("scripts.paired_preparation.train_batch", return_value={"status": "complete"}) as train, patch("sys.stdout", new=StringIO()):
            self.assertEqual(prepare_training_data.main(["batch-train", *base, "--epochs", "10", "--video-dropout", ".2", "--cpu-threads", "12"]), 0)
            self.assertEqual(train.call_args.kwargs["epochs"], 10)
            self.assertEqual(train.call_args.kwargs["video_dropout"], .2)
            self.assertEqual(train.call_args.kwargs["cpu_threads"], 12)
            self.assertNotIn("audio_checkpoint", train.call_args.kwargs)
        with patch("sys.stderr", new=StringIO()), self.assertRaises(SystemExit):
            prepare_training_data.main(["batch-train", *base, "--audio-epochs", "20", "--video-epochs", "10"])

    def test_roles_are_explicit_and_pose_is_not_enabled_by_default(self):
        batch = load_batch(self.manifest, root=self.root)
        self.assertIsNone(batch["poseModel"])
        self.assertEqual(batch["fretboardDevice"], "cpu")
        self.assertNotIn("geometryMode", batch)
        del self.document["videoPython"]
        self.save()
        with patch.dict("os.environ", {"VIDEO_PYTHON": sys.executable}):
            self.assertEqual(load_batch(self.manifest, root=self.root)["videoPython"], str(Path(sys.executable).absolute()))
        self.document["videoPython"] = sys.executable
        self.document["fretboardDevice"] = "0"
        self.save()
        self.assertEqual(load_batch(self.manifest, root=self.root)["fretboardDevice"], "0")
        self.document["fretboardDevice"] = ""
        self.save()
        with self.assertRaisesRegex(ValueError, "fretboardDevice"):
            load_batch(self.manifest, root=self.root)
        self.document.pop("fretboardDevice")
        for side in (None, "geometry", "unknown"):
            self.document["records"][0]["pluckingScreenSide"] = side
            self.save()
            with self.assertRaisesRegex(ValueError, "left or right"):
                load_batch(self.manifest, root=self.root)

    def test_automatic_optional_review_queue_is_bounded_and_nonblocking(self):
        self.document["reviewBudget"] = 3
        self.save()
        def worker(command, **kwargs):
            completed = self.worker(command, **kwargs)
            path = Path(command[command.index("--output") + 1])
            result = read_json(path)
            result["actions"] = [
                {"stage": "shots", "optional": True, "shotId": index,
                 "reason": "Optional shot review", "priority": index}
                for index in range(20)
            ]
            path.write_text(json.dumps(result))
            return completed
        with patch.dict("sys.modules", {"scripts.batch_canonical": SimpleNamespace(ensure_canonical=self.canonical)}):
            result = run_batch(self.manifest, self.output, root=self.root, runner=worker, progress=lambda message: None)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["optionalReviewCount"], 40)
        self.assertEqual(len(result["actions"]), 3)
        self.assertTrue(all(action["optional"] for action in result["actions"]))
        self.assertEqual([action["priority"] for action in result["actions"]], [19, 19, 18])

    def test_parallel_workers_publish_complete_deterministic_index_and_logs(self):
        self.document["workers"] = 2
        self.save()
        result = self.run_batch()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(set(result["records"]), {row["id"] for row in self.document["records"]})
        digest = sha256(result["indexPath"])
        self.assertTrue(all((self.output / row["id"] / "worker.log").is_file() for row in self.document["records"]))
        replay = self.run_batch()
        self.assertEqual(sha256(replay["indexPath"]), digest)

    def test_all_unavailable_training_geometry_is_not_reported_ready(self):
        identifier = next(row["id"] for row in self.document["records"] if row["split"] == "train")
        report_path = self.bundles[identifier]
        arrays_path = report_path.with_name("inputs.npz")
        with np.load(arrays_path, allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
        arrays["structured"][:] = 0
        arrays["structured_available"][:] = False
        arrays["segment_id"][:] = -1
        np.savez_compressed(arrays_path, **arrays)
        report = read_json(report_path)
        report["arraysSha256"] = sha256(arrays_path)
        report_path.write_text(json.dumps(report))
        result = self.run_batch()
        self.assertEqual(result["status"], "needs-coverage")
        self.assertGreater(result["coverage"]["splits"]["train"]["pairedWindows"], 0)
        self.assertEqual(result["coverage"]["splits"]["train"]["usableVideoWindows"], 0)

    def test_status_is_read_only_before_and_after_preparation(self):
        self.assertEqual(batch_status(self.manifest, self.output, root=self.root)["status"], "pending")
        self.assertFalse(self.output.exists())
        self.run_batch()
        before = {path: (sha256(path), path.stat().st_mtime_ns) for path in self.output.rglob("*") if path.is_file()}
        self.assertEqual(batch_status(self.manifest, self.output, root=self.root)["status"], "ready")
        self.assertEqual(before, {path: (sha256(path), path.stat().st_mtime_ns) for path in self.output.rglob("*") if path.is_file()})

    def test_accept_score_passes_explicit_authorization_to_canonical_review(self):
        from unittest.mock import Mock

        review = Mock(return_value={"status": "reviewed"})
        with patch.dict("sys.modules", {"scripts.batch_canonical": SimpleNamespace(review_canonical=review)}):
            review_batch(self.manifest, self.output, "piece-0", accept_score=True, reviewer="owner", ranges=["0:1"], root=self.root)
        self.assertTrue(review.call_args.kwargs["accept"])
        self.assertEqual(review.call_args.kwargs["root"], self.root)
        self.assertEqual(review.call_args.kwargs["ranges"], ["0:1"])

    def test_train_pipeline_pauses_before_any_optimizer_until_preparation_is_ready(self):
        from unittest.mock import Mock

        train = Mock()
        with patch("scripts.paired_preparation.run_batch", return_value={"status": "needs-review", "actions": []}):
            result = train_batch(self.manifest, self.output, root=self.root, train_command=train)
        self.assertFalse(result["trainingStarted"])
        train.assert_not_called()

    def test_explicit_train_pipeline_runs_one_joint_model_and_reuses_completed_result(self):
        arguments = prepare_training_data.parser().parse_args(["batch-train", "--manifest", str(self.manifest), "--output-directory", str(self.output)])
        self.assertIsNone(arguments.max_hours)
        prepared = self.run_batch()
        calls, checkpoints = [], {}

        def train(command):
            calls.append(command)
            config_path = Path(command[command.index("--config") + 1])
            run = Path(command[command.index("--run-dir") + 1])
            run.mkdir(parents=True)
            config = read_json(config_path)
            checkpoint = run / "best-events.pt"
            checkpoint.write_bytes(("synthetic-" + run.name).encode())
            (run / "summary.json").write_text(json.dumps({"optimizer_updates": 2, "training_steps_processed": 2}))
            identity = {"manifest_sha256": sha256(self.release), "model": config["model"], "features": config["features"]}
            identity["video"] = {"config": config["video"]["model"]}
            checkpoints[str(checkpoint)] = {"global_step": 2, "identity": identity}
            return 0

        with patch("scripts.paired_preparation.run_batch", return_value=prepared), patch("scripts.transcriber_runtime.load_checkpoint", side_effect=lambda path: checkpoints[str(path)]):
            result = train_batch(self.manifest, self.output, root=self.root, train_command=train, device="cpu", cpu_threads=12)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("--audio-checkpoint", calls[0])
            self.assertNotIn("--initialize-from", calls[0])
            self.assertNotIn("--resume", calls[0])
            config = read_json(Path(calls[0][calls[0].index("--config") + 1]))
            self.assertNotIn("freeze_audio", config["video"]["model"])
            self.assertEqual(config["video"]["model"]["architecture_version"], 6)
            self.assertEqual(
                config["video"]["model"]["feature_group_version"],
                "anatomy-fretboard-groups-v2",
            )
            self.assertEqual(config["video"]["model"]["structured_dim"], 233)
            self.assertEqual(config["data"]["num_threads"], 12)
            self.assertNotIn("--max-hours", calls[0])
            self.assertIsNone(config["training"]["max_seconds"])
            self.assertIsNone(result["invocationMaxHours"])
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["settings"]["trainingMode"], "joint")
            self.assertEqual(result["optimizerUpdatesThisInvocation"], 2)
            again = train_batch(self.manifest, self.output, root=self.root, train_command=train, device="cpu", cpu_threads=12)
            self.assertEqual(len(calls), 1)
            self.assertEqual(again["checkpoint"], result["checkpoint"])
            with self.assertRaisesRegex(ValueError, "settings changed"):
                train_batch(self.manifest, self.output, root=self.root, train_command=train, epochs=4, device="cpu")

    def test_fresh_training_attempt_preserves_failed_directory_without_checkpoint(self):
        prepared = self.run_batch()
        attempt = self.output / "training" / "joint-000"
        attempt.mkdir(parents=True)
        marker = attempt / "interrupted.log"
        marker.write_text("preserve")
        calls = []

        def fail(command):
            calls.append(command)
            return 1

        with patch("scripts.paired_preparation.run_batch", return_value=prepared):
            with self.assertRaisesRegex(ValueError, "training failed"):
                train_batch(self.manifest, self.output, root=self.root, train_command=fail)
        self.assertEqual(marker.read_text(), "preserve")
        self.assertTrue(calls[0][calls[0].index("--run-dir") + 1].endswith("joint-001"))
        self.assertEqual(read_json(self.output / "training-state.json")["status"], "training-failed")

    def test_interrupted_training_persists_failure_and_releases_lock(self):
        from unittest.mock import Mock

        prepared = self.run_batch()
        with patch("scripts.paired_preparation.run_batch", return_value=prepared):
            with self.assertRaises(KeyboardInterrupt):
                train_batch(self.manifest, self.output, root=self.root, train_command=Mock(side_effect=KeyboardInterrupt))
        state = read_json(self.output / "training-state.json")
        self.assertEqual(state["status"], "training-failed")
        self.assertTrue(state["runDirectory"].endswith("joint-000"))
        self.assertFalse((self.output / ".batch.lock").exists())

    def test_joint_training_requires_index_to_match_selected_records(self):
        from unittest.mock import Mock

        prepared = self.run_batch()
        self.document["records"] = self.document["records"][:1]
        self.save()
        train = Mock()
        with patch("scripts.paired_preparation.run_batch", return_value=prepared):
            with self.assertRaisesRegex(ValueError, "exactly this batch"):
                train_batch(self.manifest, self.output, root=self.root, train_command=train)
        train.assert_not_called()

    def test_old_staged_training_state_is_not_resumed_as_joint_training(self):
        from unittest.mock import Mock

        prepared = self.run_batch()
        state_path = self.output / "training-state.json"
        state_path.write_text(json.dumps({"kind": "paired-batch-training", "settings": {}}))
        train = Mock()
        with patch("scripts.paired_preparation.run_batch", return_value=prepared), self.assertRaisesRegex(ValueError, "inputs/settings changed"):
            train_batch(self.manifest, self.output, root=self.root, train_command=train)
        train.assert_not_called()

    def test_wallclock_pause_preserves_latest_without_requiring_best_checkpoint(self):
        prepared = self.run_batch()
        calls, checkpoints = [], {}
        def train(command):
            calls.append(command)
            run = Path(command[command.index("--run-dir") + 1])
            run.mkdir(exist_ok=True, parents=True)
            config = read_json(Path(command[command.index("--config") + 1]))
            latest = run / "latest.pt"
            latest.write_bytes(b"synthetic paused checkpoint")
            checkpoints[str(latest)] = {
                "global_step": 1, "identity": {"manifest_sha256": sha256(self.release),
                                               "video": {"config": config["video"]["model"]}},
            }
            (run / "summary.json").write_text(json.dumps({
                "stopped_by": "max_seconds", "optimizer_updates": 1, "validation_pending": True,
            }))
            return 0
        with patch("scripts.paired_preparation.run_batch", return_value=prepared), patch("scripts.transcriber_runtime.load_checkpoint", side_effect=lambda path: checkpoints[str(path)]):
            paused = train_batch(self.manifest, self.output, root=self.root, train_command=train, max_hours=.5)
            self.assertEqual(paused["status"], "paused")
            self.assertTrue(paused["validationPending"])
            self.assertFalse((Path(paused["runDirectory"]) / "best-events.pt").exists())
            again = train_batch(self.manifest, self.output, root=self.root, train_command=train, max_hours=1.)
            self.assertEqual(again["status"], "paused")
            self.assertIn("--resume", calls[1])
            self.assertEqual(calls[1][calls[1].index("--max-hours") + 1], "1.0")
            self.assertEqual(read_json(self.output / "training-state.json")["status"], "paused")


if __name__ == "__main__":
    unittest.main()
