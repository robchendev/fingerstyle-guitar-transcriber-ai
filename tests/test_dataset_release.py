from copy import deepcopy
import hashlib
from io import StringIO
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from scripts import transcriber
from scripts.dataset_io import ROOT, read_json, sha256, publish_json
from scripts.dataset_release import _validate_payload, candidate_digest, release_scope, validate_mapping, validate_release, validation_groups
from scripts.percussion_supervision import percussion_annotation_coverage
from scripts.training_windows import projected_targets, targets_in_window
from scripts.score_alignment import ScoreClock
from scripts.transcriber_audio import FeatureConfig
from scripts.transcriber_data import TrainingDataset, collate_windows
from scripts.transcriber_model import ModelConfig
from tests.test_score_alignment import clock_fixture


def synthetic_release(root, *, plateau=False, percussion_complete=None, unresolved_percussion=None, validation_groups=("piece-1",), frozen_scalar=False, voice_policy="native-multivoice"):
    entries, records = [], []
    for index, split in enumerate(("train",) + ("validation",) * len(validation_groups)):
        identifier = f"piece-{index}"
        rate = 8000
        audio = root / "audio" / f"{identifier}.flac"
        audio.parent.mkdir(parents=True, exist_ok=True)
        samples = .1 * np.sin(2 * np.pi * (220 + index * 110) * np.arange(rate * 6) / rate)
        sf.write(audio, np.column_stack((samples, -samples)), rate, subtype="PCM_24")
        labels, normalization = clock_fixture()
        labels.update(timeUnit="quarter-note", scoreTimingResolved=True)
        labels["provenance"] = {"sourceGpPath": "unavailable\\source.gp", "eventPath": "unavailable\\events.json"}
        normalization["rawCanonicalPath"] = "unavailable\\canonical.json"
        labels["conditioning"]["instrument"] = {"openStringMidi": [40, 45, 50, 55, 59, 64], "capoFret": 0}
        labels["conditioning"]["providedTiming"]["sourceTimeSignatureChanges"] = []
        labels["targets"] = {
            "notes": [{"id": "n", "voiceIndex": 0, "string": 6, "fret": 0, "soundingPitchMidi": 40, "onsetQuarter": [1, 1], "notatedDurationQuarter": [2, 1], "isAttack": True, "sourceSegments": [{"graceMode": None, "harmonic": None}], "labelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": True}}],
            "gestures": [{"id": "g", "voiceIndex": 0, "technique": "wrist_thump", "onsetQuarter": [1, 1], "scoreOnsetKnown": True, "graceMode": None}],
        }
        if percussion_complete is not None:
            labels["review"] = {"notationSymbols": [], "unresolvedGestures": [deepcopy(unresolved_percussion)] if unresolved_percussion is not None else []}
        candidate = {"denseMapping": [{"clipSeconds": float(t), "referenceSeconds": float(t), "scoreQuarter": float(t)} for t in (0, 3, 6)]}
        if plateau:
            labels["targets"]["notes"].append({**deepcopy(labels["targets"]["notes"][0]), "id": "n2", "onsetQuarter": [2, 1], "fret": 2, "soundingPitchMidi": 42})
            candidate["denseMapping"] = [{"clipSeconds": clip, "referenceSeconds": reference, "scoreQuarter": reference} for clip, reference in ((0., 0.), (1., 1.), (1., 2.), (6., 6.))]
        clock = ScoreClock(labels, normalization)
        notes, gestures = projected_targets(labels, candidate, clock)
        approval = {
            "authorizedUse": True, "recordingAndTargetPitchConfirmed": True, "notationReviewed": True,
            "approveExperimentalRangesAndSplit": True, "groupingConfirmed": True,
            "groupId": identifier, "split": split,
            "sourceGpSha256": hashlib.sha256(identifier.encode()).hexdigest(), "audioSha256": sha256(audio),
            "candidateSha256": candidate_digest(candidate), "approvedClipRanges": [[0., 6.]],
            **({"voiceSupervisionPolicy": voice_policy} if voice_policy is not None and not frozen_scalar else {}),
        }
        if plateau:
            approval["uncertaintyAcknowledged"] = True
        if percussion_complete is not None:
            approval.update(percussionAnnotationsComplete=percussion_complete, reviewer="Synthetic source reviewer")
        coverage = percussion_annotation_coverage(labels, candidate, normalization) if percussion_complete is True else None
        payload = {
            "schemaVersion": 1, "kind": "local-training-targets", "id": identifier,
            "canonical": labels, "normalization": normalization, "candidate": candidate, "approval": approval,
            "downbeatConditioning": {
                "pickupPresent": False, "scoreQuarter": [0, 1], "clipSeconds": 0.,
                "source": "normalized-gp-pickup-and-reviewed-score-audio-alignment",
            },
            "windows": [{"windowId": f"{identifier}:0-{rate * 6}", "startSample": 0, "stopSampleExclusive": rate * 6, "targets": targets_in_window(notes, gestures, 0, rate * 6, rate, percussion_coverage=coverage)}],
        }
        entries.append({
            "id": identifier, "groupId": identifier, "split": split,
            "audioPath": f"audio\\{identifier}.flac", "audioSha256": sha256(audio),
            "sampleRate": rate, "channels": 2, "sampleCount": rate * 6,
            "targetsPath": f"targets\\{identifier}.json",
        })
        records.append((entries[-1], payload))
    group_fields = {"validationGroup": validation_groups[0]} if frozen_scalar else {"validationGroups": list(validation_groups)}
    authorization = {
        "schemaVersion": 1, "kind": "local-release-authorization", "reviewer": "Synthetic reviewer",
        "version": "v1", **group_fields, "authorizedUse": True,
        "approveExperimentalRangesAndSplit": True, "groupingConfirmed": True,
        "distributionAuthorized": False, "trainingExecution": "explicit-command",
        "selectedScope": release_scope(records),
    }
    authorization["sha256"] = candidate_digest(authorization)
    for entry, payload in records:
        payload["approval"].update(releaseReviewer=authorization["reviewer"], releaseAuthorizationSha256=authorization["sha256"])
        target = root / "targets" / f"{entry['id']}.json"
        publish_json(target, payload)
        entry["targetsSha256"] = sha256(target)
    manifest = root / "manifest.json"
    publish_json(manifest, {"schemaVersion": 1, "kind": "local-training-dataset", "trainingReady": True, "visibility": "private", "distributionAuthorized": False, "entries": entries, "counts": {"windowsBySplit": {"train": 1, "validation": len(validation_groups)}}, "version": "v1", **group_fields, "trainingExecution": "explicit-command", "releaseAuthorization": authorization})
    return manifest


class DatasetReleaseTests(unittest.TestCase):
    def test_approved_sample_boundaries_use_the_proposed_integer_sample_ranges(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            path = synthetic_release(Path(directory).resolve())
            _, records, _ = validate_release(path)
            entry, payload = records[0]
            payload["approval"]["approvedClipRanges"] = [[.35000000000000003, 5.9]]
            start, stop = 2800, 47200
            clock = ScoreClock(payload["canonical"], payload["normalization"])
            notes, gestures = projected_targets(payload["canonical"], payload["candidate"], clock)
            payload["windows"] = [{
                "windowId": "sample-boundary", "startSample": start, "stopSampleExclusive": stop,
                "targets": targets_in_window(notes, gestures, start, stop, entry["sampleRate"]),
            }]
            _validate_payload(entry, payload)
            payload["windows"][0]["startSample"] -= 1
            with self.assertRaisesRegex(ValueError, "unapproved"):
                _validate_payload(entry, payload)

    def test_validation_group_documents_preserve_order_and_reject_invalid_or_ambiguous_ids(self):
        document = {"validationGroups": ["piece-2", "piece-1"]}
        self.assertEqual(validation_groups(document), ("piece-2", "piece-1"))
        self.assertEqual(document, {"validationGroups": ["piece-2", "piece-1"]})
        self.assertEqual(validation_groups({"validationGroup": "piece-1"}), ("piece-1",))
        for document in (
            None, {}, {"validationGroup": "piece-1", "validationGroups": ["piece-1"]},
            *({"validationGroups": groups} for groups in (None, "piece-1", (), {}, [], [""], [" "], [None], [12], [["nested"]], ["piece-1", "piece-1"])),
            *({"validationGroup": group} for group in (None, "", " ", 12, ["piece-1"])),
        ):
            with self.subTest(document=document), self.assertRaises(ValueError):
                validation_groups(document)

    def test_frozen_scalar_schema_one_release_keeps_exact_bytes_and_projection(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            manifest_path = synthetic_release(root, frozen_scalar=True)
            before = {path: sha256(path) for path in root.rglob("*") if path.is_file()}
            with patch("scripts.dataset_release.percussion_annotation_coverage", side_effect=AssertionError("No completeness approval")):
                manifest, records, _ = validate_release(manifest_path)
            self.assertEqual(before, {path: sha256(path) for path in before})
            self.assertEqual(manifest["schemaVersion"], 1)
            self.assertEqual(manifest["validationGroup"], "piece-1")
            self.assertEqual(manifest["releaseAuthorization"]["validationGroup"], "piece-1")
            self.assertNotIn("validationGroups", manifest)
            self.assertNotIn("validationGroups", manifest["releaseAuthorization"])
            self.assertTrue(all("percussionAnnotationsComplete" not in scope for scope in manifest["releaseAuthorization"]["selectedScope"]))
            for _, payload in records:
                targets = payload["windows"][0]["targets"]
                self.assertNotIn("percussionAnnotationCoverage", targets)
                self.assertFalse(targets["negativePercussionSupervision"])

    def test_multiple_validation_groups_preserve_order_in_the_existing_loader(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            manifest_path = synthetic_release(root, validation_groups=["piece-2", "piece-1"])
            manifest, records, _ = validate_release(manifest_path)
            self.assertEqual(validation_groups(manifest), ("piece-2", "piece-1"))
            self.assertEqual(validation_groups(manifest["releaseAuthorization"]), ("piece-2", "piece-1"))
            self.assertEqual([entry["split"] for entry, _ in records], ["train", "validation", "validation"])
            features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
            dataset = TrainingDataset(manifest_path, "validation", features, ModelConfig(n_mels=16), root=root)
            self.assertEqual(len(dataset), 2)
            self.assertEqual([dataset[index]["metadata"]["windowId"].split(":")[0] for index in range(len(dataset))], ["piece-1", "piece-2"])

    def test_confirmed_frozen_scalar_release_loads_without_rewriting_assets_or_supervision(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            manifest_path = synthetic_release(root, frozen_scalar=True, percussion_complete=True)
            before = {path: sha256(path) for path in root.rglob("*") if path.is_file()}
            manifest, records, _ = validate_release(manifest_path)
            self.assertEqual(manifest["validationGroup"], "piece-1")
            self.assertNotIn("validationGroups", manifest)
            self.assertTrue(all(scope["percussionAnnotationsComplete"] for scope in manifest["releaseAuthorization"]["selectedScope"]))
            for _, payload in records:
                targets = payload["windows"][0]["targets"]
                self.assertEqual(targets["percussionAnnotationCoverage"], [[0., 6.]])
                self.assertTrue(targets["negativePercussionSupervision"])
            features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
            dataset = TrainingDataset(manifest_path, "validation", features, ModelConfig(n_mels=16), root=root)
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertEqual(item["metadata"]["windowId"], "piece-1:0-48000")
            self.assertEqual(item["metadata"]["voiceSupervisionPolicy"], "native-multivoice")
            self.assertTrue(item["masks"]["voice"].any())
            self.assertTrue(item["masks"]["note_onset"].any())
            self.assertTrue(item["masks"]["pitch"].any())
            self.assertEqual(before, {path: sha256(path) for path in root.rglob("*") if path.is_file()})

    def test_manifest_and_authorization_reject_dual_fields_and_mismatched_group_lists(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve(), validation_groups=["piece-2", "piece-1"])
            original = read_json(manifest_path)
            for name in ("manifest", "authorization"):
                for fields, error in (
                    ({"validationGroup": "piece-1"}, "exactly one"),
                    ({"validationGroups": ["piece-1", "piece-2"]}, "different release or validation group list"),
                    ({"validationGroups": ["piece-1"]}, "different release or validation group list"),
                    ({"validationGroups": ["piece-2", "piece-2"]}, "must be unique"),
                    ({"validationGroups": []}, "nonempty list"),
                ):
                    manifest = deepcopy(original)
                    document = manifest if name == "manifest" else manifest["releaseAuthorization"]
                    document.update(fields)
                    publish_json(manifest_path, manifest)
                    with self.subTest(document=name, fields=fields), self.assertRaisesRegex(ValueError, error):
                        validate_release(manifest_path)

    def test_authorization_binds_exact_group_order_scope_and_per_recording_digest(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve(), validation_groups=["piece-2", "piece-1"])
            original = read_json(manifest_path)
            for change, error in (
                ("groups-with-old-hash", "selected dataset scope"),
                ("scope-with-new-hash", "selected dataset scope"),
                ("groups-with-new-hash", "recording approval differs"),
            ):
                manifest = deepcopy(original)
                authorization = manifest["releaseAuthorization"]
                if change == "scope-with-new-hash":
                    authorization["selectedScope"].reverse()
                else:
                    manifest["validationGroups"].reverse()
                    authorization["validationGroups"].reverse()
                digest = candidate_digest({key: value for key, value in authorization.items() if key != "sha256"})
                self.assertNotEqual(digest, original["releaseAuthorization"]["sha256"])
                if change != "groups-with-old-hash":
                    authorization["sha256"] = digest
                publish_json(manifest_path, manifest)
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, error):
                    validate_release(manifest_path)

    def test_training_execution_prose_does_not_gate_existing_frozen_releases(self):
        for marker in (None, "owner-only", "explicit-command"):
            with self.subTest(marker=marker), TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory).resolve()
                manifest_path = synthetic_release(root)
                manifest = read_json(manifest_path)
                authorization = manifest["releaseAuthorization"]
                for document in (manifest, authorization):
                    if marker is None:
                        document.pop("trainingExecution", None)
                    else:
                        document["trainingExecution"] = marker
                authorization["sha256"] = candidate_digest({key: value for key, value in authorization.items() if key != "sha256"})
                for entry in manifest["entries"]:
                    target = root / entry["targetsPath"]
                    payload = read_json(target)
                    payload["approval"]["releaseAuthorizationSha256"] = authorization["sha256"]
                    publish_json(target, payload)
                    entry["targetsSha256"] = sha256(target)
                publish_json(manifest_path, manifest)
                self.assertEqual(len(validate_release(manifest_path)[1]), 2)

    def test_authorized_validation_groups_cannot_be_absent_from_selected_recordings(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve(), validation_groups=["piece-2", "piece-1"])
            manifest = read_json(manifest_path)
            authorization = manifest["releaseAuthorization"]
            for document in (manifest, authorization):
                document["validationGroups"].append("absent-group")
            authorization["sha256"] = candidate_digest({key: value for key, value in authorization.items() if key != "sha256"})
            publish_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "present in the selected recordings"):
                validate_release(manifest_path)

    def test_multiple_validation_groups_do_not_allow_group_audio_or_gp_leakage(self):
        for duplicate, error in (("group", "related recording group"), ("audio", "Identical audio"), ("score", "Identical GP scores")):
            with self.subTest(duplicate=duplicate), TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory).resolve()
                manifest_path = synthetic_release(root, validation_groups=["piece-2", "piece-1"])
                manifest = read_json(manifest_path)
                train, validation = manifest["entries"][0], manifest["entries"][2]
                target = root / "targets" / "piece-2.json"
                payload = read_json(target)
                if duplicate == "group":
                    validation["groupId"] = train["groupId"]
                elif duplicate == "audio":
                    shutil.copyfile(root / "audio" / "piece-0.flac", root / "audio" / "piece-2.flac")
                    validation["audioSha256"] = train["audioSha256"]
                    payload["approval"]["audioSha256"] = train["audioSha256"]
                else:
                    payload["approval"]["sourceGpSha256"] = read_json(root / "targets" / "piece-0.json")["approval"]["sourceGpSha256"]
                publish_json(target, payload)
                validation["targetsSha256"] = sha256(target)
                publish_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, error):
                    validate_release(manifest_path)

    def test_multiple_validation_groups_still_require_both_nonempty_splits(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve(), validation_groups=["piece-2", "piece-1"])
            original = read_json(manifest_path)
            for split in ("train", "validation"):
                manifest = deepcopy(original)
                manifest["entries"] = [entry for entry in manifest["entries"] if entry["split"] == split]
                manifest["counts"]["windowsBySplit"] = {name: len(manifest["entries"]) if name == split else 0 for name in ("train", "validation")}
                publish_json(manifest_path, manifest)
                with self.subTest(remaining_split=split), self.assertRaisesRegex(ValueError, "split counts are empty"):
                    validate_release(manifest_path)

    def test_explicit_completeness_is_optional_and_bound_in_release_scope(self):
        for complete in (False, True):
            with self.subTest(complete=complete), TemporaryDirectory(dir=ROOT) as directory:
                manifest_path = synthetic_release(Path(directory).resolve(), percussion_complete=complete)
                manifest, records, _ = validate_release(manifest_path)
                self.assertEqual([scope["percussionAnnotationsComplete"] for scope in manifest["releaseAuthorization"]["selectedScope"]], [complete, complete])
                for _, payload in records:
                    targets = payload["windows"][0]["targets"]
                    self.assertEqual(targets["negativePercussionSupervision"], complete)
                    self.assertEqual(targets.get("percussionAnnotationCoverage"), [[0., 6.]] if complete else None)
                scope = release_scope(records)
                records[0][1]["approval"]["percussionAnnotationsComplete"] = not complete
                self.assertNotEqual(release_scope(records), scope)

    def test_confirmed_uncertainty_preserves_temporal_holes_or_empty_coverage(self):
        for unresolved, expected in (
            ({"onsetQuarter": [3, 1], "scoreOnsetKnown": True, "reason": "unknown_symbol"}, [[0., 2.9], [3.1, 6.]]),
            ({"reason": "unbounded_unknown_timing"}, []),
        ):
            with self.subTest(unresolved=unresolved), TemporaryDirectory(dir=ROOT) as directory:
                manifest_path = synthetic_release(Path(directory).resolve(), percussion_complete=True, unresolved_percussion=unresolved)
                _, records, _ = validate_release(manifest_path)
                for _, payload in records:
                    targets = payload["windows"][0]["targets"]
                    self.assertEqual(targets["percussionAnnotationCoverage"], expected)
                    self.assertEqual(targets["negativePercussionSupervision"], bool(expected))
                    self.assertTrue(targets["gestures"][0]["supervisionMask"]["gesture"])

    def test_rehashed_coverage_cannot_expand_censored_passages(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(
                Path(directory).resolve(), percussion_complete=True,
                unresolved_percussion={"onsetQuarter": [3, 1], "scoreOnsetKnown": True},
            )
            manifest = read_json(manifest_path)
            target = manifest_path.parent / "targets" / "piece-0.json"
            original = read_json(target)
            for coverage, negative in (([[0., 6.]], True), ([[0., 2.9], [3.1, 6.]], False), ([], True)):
                payload = deepcopy(original)
                payload["windows"][0]["targets"].update(percussionAnnotationCoverage=coverage, negativePercussionSupervision=negative)
                publish_json(target, payload)
                manifest["entries"][0]["targetsSha256"] = sha256(target)
                publish_json(manifest_path, manifest)
                with self.subTest(coverage=coverage, negative=negative), self.assertRaisesRegex(ValueError, "canonical projection"):
                    validate_release(manifest_path)

    def test_coverage_and_negative_flags_require_explicit_true_and_source_reviewer(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve())
            manifest = read_json(manifest_path)
            target = manifest_path.parent / "targets" / "piece-0.json"
            original = read_json(target)
            for approval in ({}, {"percussionAnnotationsComplete": False}, {"percussionAnnotationsComplete": 1}, {"percussionAnnotationsComplete": "true"}, {"percussionAnnotationsComplete": True}, {"percussionAnnotationsComplete": True, "reviewer": " "}):
                payload = deepcopy(original)
                payload["approval"].update(approval)
                payload["windows"][0]["targets"].update(percussionAnnotationCoverage=[[0., 6.]], negativePercussionSupervision=True)
                publish_json(target, payload)
                manifest["entries"][0]["targetsSha256"] = sha256(target)
                publish_json(manifest_path, manifest)
                with self.subTest(approval=approval), self.assertRaises(ValueError):
                    validate_release(manifest_path)

    def test_completeness_scope_cannot_be_omitted_even_after_authorization_rehash(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve(), percussion_complete=True)
            manifest = read_json(manifest_path)
            authorization = manifest["releaseAuthorization"]
            del authorization["selectedScope"][0]["percussionAnnotationsComplete"]
            authorization["sha256"] = candidate_digest({key: value for key, value in authorization.items() if key != "sha256"})
            publish_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "selected dataset scope"):
                validate_release(manifest_path)

    def test_mapping_bounds_finiteness_clock_and_reference_order_are_required(self):
        clock = ScoreClock(*clock_fixture())
        mapping = [{"clipSeconds": clip, "referenceSeconds": reference, "scoreQuarter": reference} for clip, reference in ((0., 0.), (1., 1.), (1., 2.), (6., 6.))]
        validate_mapping(mapping, clock, 6.)
        for index, field, value in ((1, "clipSeconds", float("nan")), (1, "referenceSeconds", float("inf")), (0, "clipSeconds", -1.), (3, "clipSeconds", 7.), (3, "referenceSeconds", 7.), (2, "clipSeconds", .9), (2, "referenceSeconds", 1.), (1, "scoreQuarter", 1.000001)):
            changed = deepcopy(mapping)
            changed[index][field] = value
            with self.subTest(index=index, field=field, value=value), self.assertRaises(ValueError):
                validate_mapping(changed, clock, 6.)

    def test_plateau_preserves_projection_and_exact_collision_masks(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            manifest_path = synthetic_release(root, plateau=True)
            manifest, records, _ = validate_release(manifest_path)
            payload = records[0][1]
            self.assertEqual([note["proposedOnsetClipSeconds"] for note in payload["windows"][0]["targets"]["notes"]], [1., 1.])
            features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
            item = TrainingDataset(manifest_path, "train", features, ModelConfig(n_mels=16), root=root)[0]
            self.assertEqual(item["metadata"]["stringFrameCollisionsMasked"], 1)
            expected = torch.zeros((300, 6))
            expected[50, 0] = 1
            torch.testing.assert_close(item["targets"]["note_onset"], expected)
            mask = torch.zeros((300, 6), dtype=torch.bool)
            mask[25:274] = True
            mask[47:54, 0] = False
            mask[50, 0] = True
            torch.testing.assert_close(item["masks"]["note_onset"], mask)
            for name in ("fret", "pitch", "voice", "duration_log", "harmonic", "harmonic_kind", "harmonic_node"):
                torch.testing.assert_close(item["masks"][name], torch.zeros((300, 6), dtype=torch.bool))
            target = root / "targets" / "piece-0.json"
            payload["approval"]["uncertaintyAcknowledged"] = False
            publish_json(target, payload)
            manifest["entries"][0]["targetsSha256"] = sha256(target)
            publish_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "plateaus"):
                validate_release(manifest_path)

    def test_single_loader_rejects_retired_schema_without_source_resolution(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            manifest_path = root / "manifest.json"
            publish_json(manifest_path, {"kind": "released-experimental-pilot-dataset", "proposalPath": "must-not-read.json"})
            with self.assertRaisesRegex(ValueError, "training release"):
                TrainingDataset(manifest_path, "train", FeatureConfig(), ModelConfig(), root=root)
            with self.assertRaisesRegex(ValueError, "training release"):
                transcriber.make_dataset(transcriber.default_config(), FeatureConfig(), ModelConfig(), "train", root, manifest_path)

    def test_portable_release_loads_after_original_workspace_is_removed(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory).resolve()
            source = root / "old-workspace" / "release"
            manifest = synthetic_release(source)
            digest = sha256(manifest)
            moved = root / "relocated"
            shutil.copytree(source, moved)
            shutil.rmtree(source.parent)
            features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
            model = ModelConfig(n_mels=16)
            dataset = TrainingDataset(moved / "manifest.json", "train", features, model, root=moved)
            self.assertEqual(dataset.manifest_sha256, digest)
            item = dataset[0]
            self.assertEqual(item["features"].shape, (300, 16))
            self.assertEqual(item["conditioning"].shape, (300, 12))
            self.assertTrue(torch.isfinite(item["features"]).all())
            self.assertGreater(float(item["features"].std()), .1)
            self.assertEqual(int(item["masks"]["percussion"].sum()), 1)
            self.assertEqual(int(item["targets"]["note_onset"].sum()), 1)
            self.assertEqual(collate_windows([item])["lengths"].tolist(), [300])
            with patch("scripts.transcriber_runtime.run_training", side_effect=AssertionError("No training in preflight")), patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer in preflight")), patch("sys.stdout", new=StringIO()):
                self.assertEqual(transcriber.main(["preflight", "--data-root", str(moved), "--manifest", "manifest.json", "--forward"]), 0)
            report = read_json(moved / "runs" / "preflight.json")
            self.assertFalse(report["trainingRun"])
            self.assertEqual(report["weights"], "untrained-in-memory-only")
            self.assertEqual(report["splits"]["validation"]["forwardShapes"]["note_onset_logits"], [1, 300, 6])

    def test_explicit_approval_and_connected_split_guards(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve())
            manifest = read_json(manifest_path)
            for changed in (
                {**manifest, "distributionAuthorized": True},
                {**manifest, "counts": {"windowsBySplit": {"train": 2, "validation": 1}}},
                {**manifest, "entries": [manifest["entries"][0], {**manifest["entries"][1], "groupId": "piece-0"}]},
                {**manifest, "entries": [manifest["entries"][0], {**manifest["entries"][1], "split": "test"}]},
                {**manifest, "releaseAuthorization": {**manifest["releaseAuthorization"], "selectedScope": []}},
                {**manifest, "validationGroups": ["piece-0"]},
            ):
                publish_json(manifest_path, changed)
                with self.assertRaises(ValueError):
                    validate_release(manifest_path)
            publish_json(manifest_path, manifest)
            target = manifest_path.parent / "targets" / "piece-0.json"
            payload = read_json(target)
            payload["approval"]["authorizedUse"] = False
            publish_json(target, payload)
            manifest["entries"][0]["targetsSha256"] = sha256(target)
            publish_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "approval"):
                validate_release(manifest_path)

    def test_rehashed_mutations_cannot_weaken_masks_or_change_approved_timing(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve())
            original_manifest = read_json(manifest_path)
            target = manifest_path.parent / "targets" / "piece-0.json"
            original_payload = read_json(target)

            def change_mask(payload):
                payload["windows"][0]["targets"]["negativePercussionSupervision"] = True

            def change_candidate(payload):
                payload["candidate"]["denseMapping"][1]["clipSeconds"] = 3.5

            def change_range(payload):
                payload["approval"]["approvedClipRanges"] = [[1., 6.]]

            def change_quarter(payload):
                payload["candidate"]["denseMapping"][1]["scoreQuarter"] = 4.
                payload["approval"]["candidateSha256"] = candidate_digest(payload["candidate"])

            for mutation in (change_mask, change_candidate, change_range, change_quarter):
                payload, manifest = deepcopy(original_payload), deepcopy(original_manifest)
                mutation(payload)
                publish_json(target, payload)
                manifest["entries"][0]["targetsSha256"] = sha256(target)
                publish_json(manifest_path, manifest)
                with self.subTest(mutation=mutation.__name__), self.assertRaises(ValueError):
                    validate_release(manifest_path)

    def test_changed_bytes_and_escaping_paths_fail(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            manifest_path = synthetic_release(Path(directory).resolve())
            manifest = read_json(manifest_path)
            target = manifest_path.parent / "targets" / "piece-0.json"
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_release(manifest_path)
            manifest["entries"][0]["targetsSha256"] = sha256(target)
            manifest["entries"][0]["targetsPath"] = "targets\\..\\manifest.json"
            publish_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "stay under"):
                validate_release(manifest_path)


if __name__ == "__main__":
    unittest.main()
