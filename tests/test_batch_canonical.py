from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
import xml.etree.ElementTree as ET

from scripts import batch_canonical, prepare_training_data as preparation
from scripts.dataset_io import ROOT, read_json, sha256
from scripts.dataset_release import validate_release
from tests.test_gp_normalization import archive_bytes
from tests.test_prepare_training_data import fresh_audio, fresh_score


class BatchCanonicalTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / f".batch-canonical-tests-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / ".gitignore").write_text("*\n", encoding="ascii")
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.workspace = self.root / "workspace"
        self.batch = {
            "workspace": str(self.workspace), "releaseVersion": "synthetic-v1",
            "acceptConventions": True, "ffmpegDirectory": None,
            "records": [self.record("score-A", "train", 0), self.record("score-B", "validation", 2, suffix=".wav")],
        }

    def record(self, identifier, split, shift, *, suffix=".flac", unknown=False):
        gp, audio = self.inputs / f"{identifier}.gp", self.inputs / f"{identifier}{suffix}"
        gp.write_bytes(archive_bytes(fresh_score(identifier, shift, unknown=unknown)))
        fresh_audio(audio, shift)
        video = self.inputs / f"{identifier}.mp4"
        video.write_bytes(b"synthetic video unused by the canonical bridge")
        return {
            "id": identifier, "groupId": f"group-{identifier}", "split": split,
            "gp": str(gp), "audio": str(audio), "video": str(video), "title": identifier,
        }

    def snapshot(self, directory=None):
        return {
            str(path.relative_to(self.root)): (sha256(path), path.stat().st_mtime_ns)
            for path in (directory or self.root).rglob("*") if path.is_file()
        }

    def approve(self, record, **options):
        return batch_canonical.review_canonical(
            self.batch, record["id"], reviewer="synthetic-owner", accept=True,
            ranges=["0.4:16.4"], anchors=["1=0.4", "end=16.4"],
            acknowledge_uncertainty=True, **options,
        )

    def freeze(self):
        batch_canonical.ensure_canonical(self.batch)
        for record in self.batch["records"]:
            self.approve(record)
        return batch_canonical.finalize_canonical(self.batch, "synthetic-owner")

    def test_fresh_import_review_release_and_repeat_preserve_trimmed_bytes(self):
        originals = {record[field]: sha256(record[field]) for record in self.batch["records"] for field in ("gp", "audio")}
        with patch.object(preparation, "release_dataset", side_effect=AssertionError("Batch-run cannot release")):
            pending = batch_canonical.ensure_canonical(self.batch)
            self.assertEqual(pending["status"], "needs-review")
            self.assertNotIn("manifestPath", pending)
            self.assertEqual(len(pending["actions"]), 2)
            for action in pending["actions"]:
                directory = self.workspace / "pairs" / action["id"]
                candidate = read_json(directory / "alignment.json")
                mapping = candidate["denseMapping"]
                self.assertEqual(action["suggestedRanges"], [[mapping[0]["clipSeconds"], mapping[-1]["clipSeconds"]]] if mapping else [])
                self.assertTrue(Path(action["notationPath"]).is_file())
                self.assertTrue(Path(action["reportPath"]).is_file())
                self.assertFalse((directory / "review.json").exists())
                self.assertIn("rules.json", action["reviewGuidance"])
            prepared = self.snapshot()
            self.assertEqual(batch_canonical.ensure_canonical(self.batch), pending)
            self.assertEqual(prepared, self.snapshot())
            for record in self.batch["records"]:
                report = self.approve(record)
                self.assertEqual(report["approval"]["split"], record["split"])
                self.assertTrue(all(report["approval"][field] for field in preparation.CONFIRMATIONS.values()))
                self.assertNotIn("percussionAnnotationsComplete", report["approval"])
            awaiting = batch_canonical.ensure_canonical(self.batch)
            self.assertEqual(awaiting["status"], "needs-review")
            self.assertEqual([action["action"] for action in awaiting["actions"]], ["finalize-release"])
            self.assertIn("--authorize-release", awaiting["actions"][0]["command"])
            self.assertIn("REVIEWER", awaiting["actions"][0]["command"])
            self.assertEqual(
                [(row["id"], row["split"], row["approvedClipRanges"]) for row in awaiting["actions"][0]["selectedScope"]],
                [("score-A", "train", [[.4, 16.4]]), ("score-B", "validation", [[.4, 16.4]])],
            )
            self.assertFalse((self.workspace / "releases").exists())
        result = batch_canonical.finalize_canonical(self.batch, "synthetic-owner")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["actions"], [])
        manifest, records, _ = validate_release(result["manifestPath"])
        self.assertNotIn("trainingExecution", manifest)
        self.assertNotIn("trainingExecution", manifest["releaseAuthorization"])
        self.assertTrue(all("trainingExecution" not in payload for _, payload in records))
        self.assertEqual(manifest["releaseAuthorization"]["reviewer"], "synthetic-owner")
        self.assertEqual([entry["id"] for entry, _ in records], ["score-A", "score-B"])
        self.assertEqual(manifest["validationGroups"], ["group-score-B"])
        self.assertEqual(Path(result["records"][0]["audio"]).read_bytes(), Path(self.batch["records"][0]["audio"]).read_bytes())
        for row, source in zip(result["records"], self.batch["records"]):
            self.assertEqual(row["sourceAudio"], source["audio"])
            self.assertEqual(row["gp"], source["gp"])
            self.assertEqual(read_json(row["targets"])["sourceAudioSha256"], originals[source["audio"]])
            self.assertEqual(Path(row["audio"]).suffix, ".flac")
        frozen = self.snapshot()
        self.assertEqual(batch_canonical.ensure_canonical(self.batch), result)
        self.assertEqual(batch_canonical.finalize_canonical(self.batch, "synthetic-owner"), result)
        self.assertEqual(frozen, self.snapshot())
        self.assertEqual(originals, {path: sha256(path) for path in originals})

    def test_trimmed_video_without_audio_reaches_frozen_release_and_joint_training(self):
        import numpy as np

        from scripts.dataset_io import publish_json
        from scripts.paired_preparation import batch_status, finalize_batch, review_batch, run_batch
        from tests.test_local_media import make_video
        from tests.test_paired_video import schema5_bundle_fixture

        for index, record in enumerate(self.batch["records"]):
            score = fresh_score(record["id"], index * 2)
            ET.SubElement(score.find("./Notes/Note[@id='1']"), "Vibrato").text = "Slight"
            Path(record["gp"]).write_bytes(archive_bytes(score))
            video = Path(record["video"]).with_suffix(".mkv")
            make_video(video, Path(record.pop("audio")), duration=19)
            record.update(video=str(video), pluckingScreenSide="left")
        manifest = self.root / "batch.json"
        fretboard_model = self.root / "models" / "fretboard.pt"
        fretboard_model.parent.mkdir(parents=True, exist_ok=True)
        fretboard_model.write_bytes(b"synthetic fretboard detector")
        document = {
            **self.batch, "schemaVersion": 1, "kind": "paired-preparation-batch",
            "handModel": "models\\hand_landmarker.task", "videoPython": sys.executable,
            "fretboardModel": str(fretboard_model),
        }
        publish_json(manifest, document)
        output = self.root / "runs" / "video-evidence" / "batches" / "fresh"
        originals = {record[field]: (sha256(record[field]), Path(record[field]).stat().st_mtime_ns)
                     for record in self.batch["records"] for field in ("gp", "video")}
        self.assertEqual(batch_status(manifest, output, root=self.root)["status"], "pending")
        self.assertFalse(output.exists())
        bundles = self.root / "bundles"
        bundles.mkdir()
        calls = []

        def worker(command, **kwargs):
            request = read_json(command[command.index("--request") + 1])
            calls.append(request)
            receipt = next(read_json(path) for path in (output / "source-cache").glob("*\\receipt.json")
                           if read_json(path)["videoSha256"] == sha256(request["video"]))
            self.assertEqual(receipt["videoSha256"], sha256(request["video"]))
            self.assertEqual(receipt["audioSha256"], sha256(request["audio"]))
            self.assertEqual(receipt["stream"]["firstDecodedPts"], 1250)
            self.assertEqual(receipt["stream"]["timeBase"], [1, 1000])
            self.assertIsNone(request["poseModel"])
            self.assertEqual(set(request), {"schemaVersion", "kind", "id", "video", "audio", "outputDirectory", "pluckingScreenSide", "reuse", "handModel", "poseModel", "fretboardModel", "fretboardDevice", "reviewMode"})
            bundle = bundles / request["id"] / "inputs.json"
            if not bundle.exists():
                bundle, report, arrays = schema5_bundle_fixture(bundles, Path(request["audio"]), request["id"], pts=range(1650, 17250, 40))
                report["inputPaths"]["video"] = request["video"]
                alignment = Path(report["inputPaths"]["alignment"])
                value = read_json(alignment)
                value.update(videoSha256=sha256(request["video"]), videoStartSecondsForTrimmedAudioZero=1.25)
                publish_json(alignment, value)
                arrays["audio_seconds"] -= 1.25
                arrays["structured"][..., :98] = 0
                arrays["structured_available"][..., :98] = False
                np.savez_compressed(bundle.with_name("inputs.npz"), **arrays)
                report.update(videoSha256=sha256(request["video"]), arraysSha256=sha256(bundle.with_name("inputs.npz")))
                fretboard = Path(report["inputPaths"]["fretboard"])
                fretboard_report = read_json(fretboard)
                fretboard_report["videoSha256"] = sha256(request["video"])
                fretboard_report["shotsSha256"] = report["inputSha256"]["shots"]
                publish_json(fretboard, fretboard_report)
                report["inputSha256"].update(video=sha256(request["video"]), alignment=sha256(alignment))
                report["inputSha256"]["fretboard"] = sha256(fretboard)
                report["clock"]["offsetSamples"] = -10000
                publish_json(bundle, report)
            publish_json(command[command.index("--output") + 1], {
                "schemaVersion": 1, "kind": "paired-video-preparation-result", "id": request["id"], "status": "ready",
                "inputSha256": {key: sha256(request[key]) for key in ("video", "audio")},
                "artifacts": {"bundle": str(bundle)}, "actions": [], "stageSummary": [],
            })
            return SimpleNamespace(returncode=0)

        pending = run_batch(manifest, output, root=self.root, runner=worker, progress=lambda message: None)
        self.assertEqual(pending["status"], "needs-review")
        self.assertEqual(calls, [])
        self.assertEqual(len(list((output / "source-cache").glob("*\\receipt.json"))), 2)
        for record in self.batch["records"]:
            review_batch(
                manifest, output, record["id"], reviewer="reviewer", accept_score=True,
                ranges=["0.4:16.4"], anchors=["1=0.4", "end=16.4"], acknowledge_uncertainty=True, root=self.root,
            )
        frozen = finalize_batch(manifest, output, "reviewer", root=self.root)
        self.assertEqual(frozen["status"], "ready")
        with patch("scripts.local_media.run_media", side_effect=AssertionError("Do not re-extract unchanged video")):
            ready = run_batch(manifest, output, root=self.root, runner=worker, progress=lambda message: None)
            repeated = run_batch(manifest, output, root=self.root, runner=worker, progress=lambda message: None)
            self.assertEqual(batch_status(manifest, output, root=self.root)["status"], "ready")
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["indexPath"], repeated["indexPath"])
        self.assertFalse(ready["trainingPerformed"])
        self.assertEqual(read_json(ready["indexPath"])["schemaVersion"], 5)
        self.assertTrue(all({"gp", "video", "audio", "audioOrigin"} == set(row) for row in ready["identity"]["sources"].values()))
        for values in ready["coverage"]["splits"].values():
            self.assertEqual(values["visualCoverage"]["framesWithGuitarGeometry"], 0)
            self.assertEqual(values["visualCoverage"]["framesWithCoarseContext"], 0)
            self.assertGreater(values["usableVideoWindows"], 0)
        training = ROOT / "runs" / f"synthetic-joint-{uuid4().hex}"
        training.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, training)
        config_path = training / "config.json"

        def transcriber(*arguments):
            result = subprocess.run([sys.executable, "-B", "-m", "scripts.transcriber", *map(str, arguments)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        transcriber("config", "--manifest", frozen["manifestPath"], "--video-index", ready["indexPath"], "--output", config_path)
        config = read_json(config_path)
        self.assertEqual(config["video"]["model"]["structured_dim"], 233)
        self.assertNotIn("freeze_audio", config["video"]["model"])
        config["data"].update(manifest=frozen["manifestPath"], batch_size=1, num_threads=1)
        config["features"].update(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
        config["model"].update(n_mels=16, hidden_size=8, recurrent_layers=1, dropout=0)
        config["video"]["model"].update(hidden_size=8, modality_dropout=0)
        config["training"].update(device="cpu", epochs=1, max_steps=1)
        publish_json(config_path, config)
        preflight = self.root / "runs" / "preflight.json"
        transcriber("preflight", "--config", config_path, "--data-root", self.root, "--forward", "--output", preflight)
        report = read_json(preflight)
        self.assertFalse(report["trainingRun"])
        self.assertEqual(report["initialization"], "joint-audio-numeric-video-from-scratch")
        self.assertTrue(all(row["windows"] > 0 and row["independent_hand_observation_view_frames"] > 0 for row in report["splits"].values()))
        run = self.root / "runs" / "joint"
        transcriber("train", "--config", config_path, "--data-root", self.root, "--run-dir", run)
        from scripts.transcriber_runtime import load_checkpoint

        checkpoint = load_checkpoint(run / "latest.pt")
        self.assertEqual(checkpoint["global_step"], 1)
        self.assertEqual(checkpoint["identity"]["initialization"]["kind"], "joint-audio-numeric-video-from-scratch")
        self.assertFalse(checkpoint["identity"]["initialization"]["audioParametersFrozen"])
        self.assertTrue(checkpoint["optimizer_state"]["state"])
        self.assertTrue((run / "best-events.pt").is_file())
        self.assertEqual(read_json(run / "summary.json")["trainingMode"], "joint-audio-numeric-video-from-scratch")
        self.assertEqual(originals, {path: (sha256(path), Path(path).stat().st_mtime_ns) for path in originals})

    def test_frozen_subset_never_reads_or_regenerates_preparation(self):
        self.batch["records"].append(self.record("score-C", "train", 4))
        ready = self.freeze()
        frozen_batch = deepcopy(self.batch)
        frozen_batch.pop("releaseVersion")
        frozen_batch["releaseManifest"] = ready["manifestPath"]
        frozen_batch["records"] = [self.batch["records"][1], self.batch["records"][0]]
        shutil.rmtree(self.workspace / "pairs")
        snapshot = self.snapshot()
        with patch.object(preparation, "load_current", side_effect=AssertionError("Frozen preparation is not current-code-bound")), \
                patch.object(preparation, "prepare_pair", side_effect=AssertionError("No frozen regeneration")), \
                patch.object(preparation, "initialize", side_effect=AssertionError("No frozen workspace mutation")), \
                patch.object(preparation, "release_dataset", side_effect=AssertionError("No frozen republishing")):
            result = batch_canonical.ensure_canonical(frozen_batch)
            self.assertEqual(result["status"], "ready")
            self.assertEqual([(row["id"], row["split"]) for row in result["records"]], [("score-B", "validation"), ("score-A", "train")])
            self.assertEqual(result["manifestPath"], ready["manifestPath"])
            self.assertEqual(batch_canonical.finalize_canonical(frozen_batch, "synthetic-owner"), result)
        self.assertEqual(snapshot, self.snapshot())
        with self.assertRaisesRegex(ValueError, "Frozen releases"):
            batch_canonical.review_canonical(frozen_batch, "score-A")

    def test_frozen_sources_groups_and_splits_must_match(self):
        ready = self.freeze()
        frozen = deepcopy(self.batch)
        frozen.pop("releaseVersion")
        frozen["releaseManifest"] = ready["manifestPath"]
        for field, value in (("groupId", "different-group"), ("split", "validation"), ("id", "absent")):
            changed = deepcopy(frozen)
            changed["records"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differs|not in"):
                batch_canonical.ensure_canonical(changed)
        release_before = self.snapshot(Path(ready["manifestPath"]).parent)
        for index, field in ((0, "gp"), (0, "audio"), (1, "audio")):
            source = Path(frozen["records"][index][field])
            original = source.read_bytes()
            source.write_bytes(original + b"changed")
            try:
                with self.subTest(index=index, field=field), self.assertRaisesRegex(ValueError, "differs"):
                    batch_canonical.ensure_canonical(frozen)
            finally:
                source.write_bytes(original)
        self.assertEqual(release_before, self.snapshot(Path(ready["manifestPath"]).parent))

    def test_existing_version_requires_exact_selected_scope(self):
        self.freeze()
        subset = deepcopy(self.batch)
        subset["records"] = subset["records"][:1]
        with self.assertRaisesRegex(ValueError, "different selected scope"):
            batch_canonical.ensure_canonical(subset)

    def test_changed_external_sources_never_overwrite_imported_sources(self):
        batch_canonical.ensure_canonical(self.batch)
        before = self.snapshot(self.workspace)
        for field in ("gp", "audio"):
            source = Path(self.batch["records"][0][field])
            original = source.read_bytes()
            source.write_bytes(original + b"changed")
            try:
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differs from the external source"):
                    batch_canonical.ensure_canonical(self.batch)
                self.assertEqual(before, self.snapshot(self.workspace))
            finally:
                source.write_bytes(original)

    def test_stale_preparation_requires_deliberate_invalidation(self):
        batch_canonical.ensure_canonical(self.batch)
        rules_path = self.workspace / "pairs" / "score-A" / "rules.json"
        rules = read_json(rules_path)
        rules["confirmFixedTuningCapoText"] = True
        preparation.write_json(rules_path, rules)
        before = self.snapshot(self.workspace)
        with patch.object(preparation, "prepare_pair", side_effect=AssertionError("Stale assets cannot be rebuilt")):
            result = batch_canonical.ensure_canonical(self.batch)
        action = next(action for action in result["actions"] if action["id"] == "score-A")
        self.assertEqual(action["action"], "invalidate-canonical")
        self.assertIn("invalidate", action["command"])
        self.assertEqual(before, self.snapshot(self.workspace))
        pair = preparation.selection(self.workspace, ["score-A"])[0]
        preparation.invalidate(self.workspace, pair, "synthetic rules review")
        with patch.object(preparation, "prepare_pair", side_effect=AssertionError("An invalidated preparation still requires explicit reprepare")):
            result = batch_canonical.ensure_canonical(self.batch)
        action = next(action for action in result["actions"] if action["id"] == "score-A")
        self.assertEqual(action["action"], "prepare-canonical")
        self.assertIn("prepare", action["command"])
        preparation.prepare_pair(self.workspace, pair)
        self.assertEqual(read_json(rules_path), rules)
        result = batch_canonical.ensure_canonical(self.batch)
        self.assertTrue(all(action["action"] == "review-canonical" for action in result["actions"]))

    def test_corrupt_preparation_is_reported_without_overwrite(self):
        batch_canonical.ensure_canonical(self.batch)
        state = self.workspace / "pairs" / "score-A" / "preparation.json"
        state.write_text("{invalid", encoding="ascii")
        before = self.snapshot(self.workspace)
        result = batch_canonical.ensure_canonical(self.batch)
        action = next(action for action in result["actions"] if action["id"] == "score-A")
        self.assertEqual(action["action"], "invalidate-canonical")
        self.assertEqual(before, self.snapshot(self.workspace))

    def test_registered_group_change_is_not_silently_adopted(self):
        batch_canonical.ensure_canonical(self.batch)
        before = self.snapshot(self.workspace)
        self.batch["records"][0]["groupId"] = "changed-group"
        with self.assertRaisesRegex(ValueError, "registered group differs"):
            batch_canonical.ensure_canonical(self.batch)
        self.assertEqual(before, self.snapshot(self.workspace))

    def test_unreviewed_gp_structure_surfaces_existing_cli_remedy(self):
        gp = Path(self.batch["records"][0]["gp"])
        score = fresh_score("unsupported tuning")
        tuning = score.find("./Tracks/Track/Staves/Staff/Properties/Property[@name='Tuning']/Pitches")
        self.assertIsNotNone(tuning)
        tuning.text = "40 45 50 55 59"
        gp.write_bytes(archive_bytes(score))
        before = sha256(gp)
        result = batch_canonical.ensure_canonical(self.batch)
        action = next(action for action in result["actions"] if action["id"] == "score-A")
        self.assertEqual(action["action"], "review-gp-structure")
        self.assertIn("Unresolved GP inspection", action["reason"])
        self.assertIn("prepare", action["command"])
        self.assertEqual(before, sha256(gp))
        self.assertEqual(before, sha256(self.workspace / "pairs" / "score-A" / "raw.gp"))

    def test_capo_text_requires_explicit_rule_without_inferred_metadata(self):
        gp = Path(self.batch["records"][0]["gp"])
        score = fresh_score("fixed tuning and capo")
        ET.SubElement(score.find("./Beats/Beat"), "FreeText").text = "Fixed tuning and full capo"
        gp.write_bytes(archive_bytes(score))
        original = sha256(gp)
        result = batch_canonical.ensure_canonical(self.batch)
        action = next(action for action in result["actions"] if action["id"] == "score-A")
        self.assertEqual(action["action"], "review-capo-tuning-text")
        self.assertIn("confirmFixedTuningCapoText", action["reason"])
        self.assertEqual(action["path"], action["rulesPath"])
        rules_path = Path(action["rulesPath"])
        rules = read_json(rules_path)
        self.assertFalse(rules["confirmFixedTuningCapoText"])
        self.assertFalse((rules_path.parent / "preparation.json").exists())
        rules["confirmFixedTuningCapoText"] = True
        preparation.write_json(rules_path, rules)
        batch_canonical.ensure_canonical(self.batch)
        self.assertTrue((rules_path.parent / "preparation.json").is_file())
        self.assertEqual(read_json(rules_path), rules)
        self.assertEqual(sha256(gp), original)

    def test_stale_inactive_partial_capo_metadata_is_not_auto_accepted(self):
        gp = Path(self.batch["records"][0]["gp"])
        score = fresh_score("explicit full capo")
        score.find(".//Staff/Properties/Property[@name='PartialCapoFret']/Fret").text = "4"
        gp.write_bytes(archive_bytes(score))
        result = batch_canonical.ensure_canonical(self.batch)
        action = next(action for action in result["actions"] if action["id"] == "score-A")
        self.assertEqual(action["action"], "review-full-capo-metadata")
        self.assertIn("all six partial-capo flags zero", action["reason"])
        self.assertFalse(read_json(action["rulesPath"])["confirmFullCapoMetadata"])

    def test_source_legend_uses_existing_rules_editor_without_guessing(self):
        gp = Path(self.batch["records"][0]["gp"])
        gp.write_bytes(archive_bytes(fresh_score("source legend", legend=True)))
        result = batch_canonical.ensure_canonical(self.batch)
        action = next(action for action in result["actions"] if action["id"] == "score-A")
        self.assertEqual(action["action"], "review-notation-rules")
        self.assertEqual(action["path"], action["rulesPath"])
        self.assertIn("Never infer an unreviewed symbol meaning", action["reason"])
        self.assertEqual(read_json(action["rulesPath"])["rules"], [])
        self.assertTrue(any(beat["referenceOnly"] for beat in read_json(action["notationPath"])["beats"]))

    def test_run_never_accepts_conventions_or_review_decisions_by_default(self):
        self.batch.pop("acceptConventions")
        result = batch_canonical.ensure_canonical(self.batch)
        self.assertEqual(result["status"], "needs-review")
        for record in self.batch["records"]:
            directory = self.workspace / "pairs" / record["id"]
            self.assertFalse(read_json(directory / "rules.json")["acceptConventions"])
            self.assertFalse((directory / "preparation.json").exists())
        self.assertTrue(all("fingerstyle-v1" in action["reason"] for action in result["actions"]))
        self.assertTrue(all(action["action"] == "review-conventions" for action in result["actions"]))
        self.assertTrue(all("--accept-conventions" in action["command"] for action in result["actions"]))
        self.batch["acceptConventions"] = True
        batch_canonical.ensure_canonical(self.batch)
        before = self.snapshot()
        report = batch_canonical.review_canonical(self.batch, "score-A")
        self.assertFalse(any(report["approval"].get(field) for field in preparation.CONFIRMATIONS.values()))
        self.assertEqual(before, self.snapshot())
        for kwargs in ({"ranges": ["0.4:16.4"]}, {"anchors": ["1=0.4"]}, {"percussion_complete": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "explicit accept"):
                batch_canonical.review_canonical(self.batch, "score-A", reviewer="synthetic-owner", **kwargs)
        for kwargs in ({"accept": True}, {"accept": True, "reviewer": "synthetic-owner"}, {"accept": True, "ranges": ["0.4:16.4"]}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "reviewer and nonempty"):
                batch_canonical.review_canonical(self.batch, "score-A", **kwargs)
        with self.assertRaisesRegex(ValueError, "reviewed split differs"):
            batch_canonical.finalize_canonical(self.batch, "synthetic-owner")
        self.assertFalse((self.workspace / "releases").exists())

    def test_uncertainty_and_percussion_are_explicit_separate_decisions(self):
        self.batch["records"][0] = self.record("score-A", "train", 0, unknown=True)
        batch_canonical.ensure_canonical(self.batch)
        batch_canonical.review_canonical(
            self.batch, "score-A", reviewer="synthetic-owner", accept=True,
            ranges=["0.4:16.4"], anchors=["1=0.4", "end=16.4"],
        )
        result = batch_canonical.ensure_canonical(self.batch)
        action = next(action for action in result["actions"] if action.get("id") == "score-A")
        self.assertIn("uncertainty", action["reason"])
        report = self.approve(self.batch["records"][0], percussion_complete=True)
        self.assertTrue(report["approval"]["uncertaintyAcknowledged"])
        self.assertTrue(report["approval"]["percussionAnnotationsComplete"])

    def test_unregistered_workspace_and_unbound_artifacts_are_not_adopted(self):
        self.workspace.mkdir()
        marker = self.workspace / "normalized.gp"
        marker.write_bytes(b"unowned")
        with self.assertRaisesRegex(ValueError, "existing preparation/assets are never adopted"):
            batch_canonical.ensure_canonical(self.batch)
        marker.unlink()
        self.batch["acceptConventions"] = False
        batch_canonical.ensure_canonical(self.batch)
        artifact = self.workspace / "pairs" / "score-A" / "normalized.gp"
        artifact.write_bytes(b"unbound derivative")
        self.batch["acceptConventions"] = True
        result = batch_canonical.ensure_canonical(self.batch)
        action = next(action for action in result["actions"] if action["id"] == "score-A")
        self.assertIn("not adopted or overwritten", action["reason"])
        self.assertEqual(artifact.read_bytes(), b"unbound derivative")

    def test_conflicting_groups_duplicates_and_aliases_are_rejected_before_import(self):
        cases = []
        duplicate = deepcopy(self.batch)
        duplicate["records"][1]["id"] = duplicate["records"][0]["id"].upper()
        cases.append((duplicate, "unique"))
        group = deepcopy(self.batch)
        group["records"][1]["groupId"] = group["records"][0]["groupId"]
        cases.append((group, "cannot cross"))
        copied = deepcopy(self.batch)
        copied["records"][1]["gp"] = copied["records"][0]["gp"]
        cases.append((copied, "conflicting groups"))
        alias = deepcopy(self.batch)
        alias["records"][0]["gp"] = str(self.inputs / ".." / "inputs" / "score-A.gp")
        cases.append((alias, "without aliases"))
        for batch, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                batch_canonical.ensure_canonical(batch)
            self.assertFalse(self.workspace.exists())


if __name__ == "__main__":
    unittest.main()
