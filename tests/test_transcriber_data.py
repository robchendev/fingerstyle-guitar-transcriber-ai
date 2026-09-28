from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.transcriber_data import EpochShuffleSampler, collate_windows, encode_targets, training_conditioning


VOCABULARY = SimpleNamespace(PERCUSSION_TYPES=("wrist_thump", "thumb_slap", "percussive_hit"), HARMONIC_TYPES=("Natural", "Artificial", "Tap", "Pinch"), HARMONIC_FRETS=(5, 7, 9, 12, 19, 24))
MODEL = SimpleNamespace(max_fret=36, max_voices=4)


class TargetEncodingTests(unittest.TestCase):
    def test_joint_selection_keeps_prepared_tracking_gaps_and_masks_unprepared_targets(self):
        from scripts import transcriber
        from scripts.dataset_io import ROOT, publish_json, sha256
        from scripts.paired_video import build_index
        from scripts.transcriber_audio import FeatureConfig, HarnessError
        from scripts.transcriber_model import ModelConfig
        from scripts.transcriber_data import TrainingDataset
        from tests.test_dataset_release import synthetic_release
        from tests.test_paired_video import bundle_fixture

        with TemporaryDirectory(prefix=".joint-data-", dir=ROOT) as directory:
            root = Path(directory)
            manifest = synthetic_release(root / "release", percussion_complete=True)
            audio = root / "release" / "audio" / "piece-0.flac"
            bundle, report, arrays = bundle_fixture(root, audio)
            observed_arrays = deepcopy(arrays)
            arrays["structured"].fill(0)
            arrays["structured_available"].fill(False)
            arrays["segment_id"].fill(-1)
            np.savez_compressed(bundle.with_name("inputs.npz"), **arrays)
            report["arraysSha256"] = sha256(bundle.with_name("inputs.npz"))
            publish_json(bundle, report)
            index, _ = build_index(manifest, [bundle], root / "index.json", root=root)
            config = transcriber.default_config()
            config["data"]["manifest"] = str(manifest)
            config["video"] = {"index": str(index), "model": {}}
            features = FeatureConfig(sample_rate=8000, n_fft=512, hop_length=160, n_mels=16, f_max=3000)
            model = ModelConfig(architecture_version=2, n_mels=16, hidden_size=4, recurrent_layers=1)
            raw = TrainingDataset(manifest, "train", features, model, root=root, video_index_path=index)
            selected = transcriber.make_dataset(config, features, model, "train", root)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected.video_paired_window_indices, [0])
            self.assertEqual(selected.video_coverage["preparedAudioFrames"], 8)
            self.assertEqual(selected.video_coverage["audioFramesWithUsableVideo"], 0)
            self.assertEqual(selected.video_coverage["audioFramesWithCoarseContext"], 0)
            self.assertEqual(selected.video_coverage["trackingGapOnlyWindows"], 1)
            item, original = selected[0], raw[0]
            self.assertTrue((item["video"]["frame_indices"] == -1).all())
            self.assertFalse(item["video"]["structured_available"].any())
            for name in item["targets"]:
                torch.testing.assert_close(item["targets"][name], original["targets"][name])
            self.assertTrue(item["masks"]["note_onset"][50, 0])
            self.assertTrue(item["masks"]["percussion"][50, 0])
            self.assertFalse(item["masks"]["note_onset"][:50].any())
            self.assertFalse(item["masks"]["note_onset"][58:].any())
            from scripts.technique_supervision import TECHNIQUE_TYPES
            from scripts.transcriber_data import PAIRED_TECHNIQUE_TARGET_POLICY

            ablation = TrainingDataset(manifest, "train", features, model, root=root)
            axis = TECHNIQUE_TYPES.index("rasgueado")
            self.assertTrue(ablation[0]["masks"]["technique"][:, axis].any())
            ablation.paired_technique_target_policy = PAIRED_TECHNIQUE_TARGET_POLICY
            self.assertFalse(ablation[0]["masks"]["technique"][:, axis].any())
            self.assertNotIn("video", ablation[0])
            with self.assertRaisesRegex(HarnessError, "No source-bound prepared paired windows"):
                transcriber.make_dataset(config, features, model, "validation", root)

            from scripts.dataset_release import validate_release
            from scripts.score_alignment import ScoreClock
            from scripts.training_windows import projected_targets, targets_in_window

            manifest_data, records, bindings = validate_release(manifest)
            _, payload = records[0]
            labels = payload["canonical"]
            notes, gestures = projected_targets(labels, payload["candidate"], ScoreClock(labels, payload["normalization"]))
            payload["windows"] = [{
                "windowId": f"synthetic-{start}", "startSample": start, "stopSampleExclusive": start + 16000,
                "targets": targets_in_window(notes, gestures, start, start + 16000, 8000, percussion_coverage=[[0., 6.]]),
            } for start in (0, 16000, 32000)]
            manifest_data["counts"]["windowsBySplit"]["train"] = 3
            np.savez_compressed(bundle.with_name("inputs.npz"), **observed_arrays)
            report["arraysSha256"] = sha256(bundle.with_name("inputs.npz"))
            report["clips"] = [{"startPts": 1000, "endPtsExclusive": 4000}]
            publish_json(bundle, report)
            index, _ = build_index(manifest, [bundle], root / "partly-observed-index.json", root=root)
            config["video"]["index"] = str(index)
            with patch("scripts.transcriber_data.validate_release", return_value=(manifest_data, records, bindings)):
                selected = transcriber.make_dataset(config, features, model, "train", root)
            self.assertEqual(selected.video_paired_window_indices, [0, 1])
            self.assertEqual(selected.video_coverage["excludedWindows"], 1)
            self.assertEqual(selected.video_coverage["preparedAudioFrames"], 150)
            self.assertEqual(selected.video_coverage["windowsWithUsableVideo"], 1)
            self.assertEqual(selected.video_coverage["trackingGapOnlyWindows"], 1)
            self.assertGreater(selected.video_coverage["audioFramesWithUsableVideo"], 0)
            gap = selected[1]
            self.assertTrue((gap["video"]["frame_indices"] == -1).all())
            self.assertTrue(gap["masks"]["note_onset"].any())
            self.assertTrue(gap["masks"]["percussion"].any())

    def test_sample_clock_roundoff_is_not_real_timing_extrapolation(self):
        from scripts.transcriber_audio import HarnessError
        from tests.test_score_alignment import clock_fixture

        labels, normalization = clock_fixture()
        labels["conditioning"]["instrument"] = {"openStringMidi": [40, 45, 50, 55, 59, 64], "capoFret": 0}
        labels["conditioning"]["providedTiming"]["sourceTimeSignatureChanges"] = []
        record = {
            "data": SimpleNamespace(labels=labels, normalization=normalization),
            "candidate": {"denseMapping": [{"clipSeconds": .35000000000000003, "scoreQuarter": 0.}, {"clipSeconds": 6., "scoreQuarter": 6.}]},
            "downbeatConditioning": {
                "pickupPresent": False, "scoreQuarter": [0, 1],
                "clipSeconds": .35000000000000003,
                "source": "normalized-gp-pickup-and-reviewed-score-audio-alignment",
            },
        }
        self.assertEqual(training_conditioning(record, np.array([.35, .37])).shape, (2, 12))
        downbeat = training_conditioning(record, np.array([.35, .37]), 16)
        self.assertEqual(downbeat.shape, (2, 16))
        torch.testing.assert_close(downbeat[0, 12:], torch.tensor([0., 1., 0., 1.]), atol=1e-5, rtol=0)
        with self.assertRaisesRegex(HarnessError, "outside"):
            training_conditioning(record, np.array([.35 - 1 / 48000, .37]))

    def test_confirmed_percussion_negatives_keep_gaps_boundaries_and_close_positives(self):
        gestures = [
            {"sourceGestureId": name, "technique": "wrist_thump", "onsetWindowSeconds": time, "supervisionMask": {"gesture": True, "onset": True}}
            for name, time in (("a", .8), ("b", .84))
        ]
        source = {"targets": {"notes": [], "gestures": [{"id": name, "technique": "wrist_thump"} for name in ("a", "b")]}}
        window = {"targets": {"notes": [], "gestures": gestures, "negativePercussionSupervision": True, "percussionAnnotationCoverage": [[.5, 1.], [1.2, 1.8]]}}
        with patch.dict("sys.modules", {"scripts.transcriber_model": VOCABULARY}):
            targets, masks, _ = encode_targets(window, source, np.arange(100) * .02, MODEL, negative_onsets_allowed=[True] * 6)
        self.assertTrue(masks["percussion"][35].all())
        self.assertFalse(targets["percussion"][35].any())
        self.assertFalse(masks["percussion"][55].any())
        self.assertFalse(masks["percussion"][5].any())
        self.assertFalse(masks["percussion"][90].any())
        self.assertFalse(masks["percussion"][39, 0])
        self.assertTrue(masks["percussion"][40, 0])
        self.assertTrue(masks["percussion"][42, 0])
        self.assertEqual(int(targets["percussion"].sum()), 2)

    def source_note(self, identifier="n", **changes):
        return {"id": identifier, "labelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": True}, "sourceSegments": [{"harmonic": {"type": "Natural", "fret": [12, 1]}}], **changes}

    def projected(self, identifier="n", **changes):
        return {
            "sourceNoteId": identifier, "string": 6, "fret": 12, "soundingPitchMidi": 52, "voiceIndex": 2,
            "onsetWindowSeconds": .8, "notatedDurationQuarter": [2, 1],
            "sourceLabelMask": {"attack": True, "pitch": True, "fingering": True, "notatedDuration": True},
            "supervisionMask": {"onset": True, "pitch": True, "fingering": True, "notatedDuration": True}, **changes,
        }

    def encode(self, notes, source_notes, gestures=(), source_gestures=(),
               voice_supervision_policy="native-multivoice"):
        with patch.dict("sys.modules", {"scripts.transcriber_model": VOCABULARY}):
            return encode_targets(
                {"targets": {"notes": notes, "gestures": list(gestures)}},
                {"targets": {"notes": source_notes, "gestures": list(source_gestures)}},
                np.arange(100) * .02, MODEL,
                negative_onsets_allowed=[False, True, True, True, True, True],
                voice_supervision_policy=voice_supervision_policy,
            )

    def test_voice_harmonic_and_positive_percussion_can_coexist(self):
        gesture = {"sourceGestureId": "g", "technique": "wrist_thump", "onsetWindowSeconds": .8, "supervisionMask": {"gesture": True, "onset": True}}
        targets, masks, collisions = self.encode([self.projected()], [self.source_note()], [gesture], [{"id": "g", "technique": "wrist_thump"}])
        self.assertEqual(targets["voice"][40, 0], 2)
        self.assertEqual(targets["harmonic_kind"][40, 0], 0)
        self.assertEqual(targets["harmonic_node"][40, 0], 3)
        self.assertTrue(masks["harmonic"][40, 0])
        self.assertEqual(targets["percussion"][40, 0], 1)
        self.assertEqual(int(masks["percussion"].sum()), 1)
        self.assertFalse(torch.any(masks["note_onset"][:40, 0]))
        self.assertTrue(masks["note_onset"][40, 0])
        self.assertTrue(masks["note_onset"][60, 1])
        self.assertEqual(collisions, 0)

    def test_flattened_voice_policy_masks_only_voice_supervision(self):
        targets, masks, _ = self.encode(
            [self.projected()], [self.source_note()],
            voice_supervision_policy="flattened-or-unknown",
        )
        self.assertEqual(targets["voice"][40, 0], 0)
        self.assertFalse(masks["voice"][40, 0])
        for name in ("note_onset", "fret", "pitch", "duration_log"):
            self.assertTrue(masks[name][40, 0], name)
        with self.assertRaisesRegex(Exception, "voice supervision policy"):
            self.encode(
                [self.projected()], [self.source_note()],
                voice_supervision_policy="invented",
            )

    def test_collision_masks_categories_but_preserves_attack_presence(self):
        notes = [self.projected("a"), self.projected("b", fret=7)]
        targets, masks, collisions = self.encode(notes, [self.source_note("a"), self.source_note("b")])
        self.assertEqual(collisions, 1)
        self.assertEqual(targets["note_onset"][40, 0], 1)
        self.assertTrue(masks["note_onset"][40, 0])
        for name in ("fret", "pitch", "voice", "duration_log", "harmonic"):
            self.assertFalse(masks[name][40, 0])

    def test_carry_in_and_unknown_onsets_are_not_new_attacks(self):
        note = self.projected(supervisionMask={"onset": False, "pitch": False, "fingering": False, "notatedDuration": False})
        targets, masks, _ = self.encode([note], [self.source_note()])
        self.assertEqual(int(targets["note_onset"].sum()), 0)
        self.assertEqual(int(masks["harmonic"].sum()), 0)
        self.assertEqual(int(masks["pitch"].sum()), 0)

    def test_sampler_is_reproducible_by_epoch_without_consuming_global_rng(self):
        sampler = EpochShuffleSampler(range(12), seed=9)
        state = torch.random.get_rng_state().clone()
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        torch.testing.assert_close(state, torch.random.get_rng_state())

    def test_collation_masks_padding_and_keeps_metadata_out_of_features(self):
        items = []
        for length in (3, 5):
            items.append({"features": torch.ones(length, 16), "conditioning": torch.ones(length, 12), "targets": {"note_onset": torch.ones(length, 6)}, "masks": {"note_onset": torch.ones(length, 6, dtype=torch.bool)}, "metadata": {"private_id": str(length)}})
        batch = collate_windows(items)
        self.assertEqual(batch["features"].shape, (2, 5, 16))
        self.assertFalse(batch["valid_frames"][0, 3])
        self.assertFalse(batch["masks"]["note_onset"][0, 3].any())
        self.assertEqual(batch["metadata"], [{"private_id": "3"}, {"private_id": "5"}])

    def test_video_collation_rejects_old_shapes_dtypes_and_invalid_observations(self):
        from scripts.paired_video import empty_video
        from scripts.transcriber_audio import HarnessError
        from scripts.video_features import STRUCTURED_DIM

        for change in ("dimension", "views", "mask_dtype", "orientation", "wrist", "motion", "index"):
            with self.subTest(change=change):
                video = empty_video(3)
                if change == "dimension":
                    video["structured"] = torch.zeros(1, 4, 98)
                elif change == "views":
                    video["structured"] = torch.zeros(1, 2, STRUCTURED_DIM)
                elif change == "mask_dtype":
                    video["structured_available"] = video["structured_available"].float()
                elif change == "index":
                    video["frame_indices"][0] = 1
                else:
                    video["structured_available"][0, 2, 98:140] = True
                    video["segment_id"][0, 2] = 0
                    if change == "orientation":
                        video["structured_available"][0, 2, 184:186] = True
                    elif change == "wrist":
                        video["structured"][0, 2, 98] = .1
                    else:
                        video["structured_available"][0, 2, 140:184] = True
                item = {"features": torch.zeros(3, 16), "conditioning": torch.zeros(3, 12),
                        "targets": {}, "masks": {}, "metadata": {}, "video": video}
                with self.assertRaises(HarnessError):
                    collate_windows([item])

    def test_coarse_collation_masks_padding_and_retains_hand_only_examples(self):
        from scripts.paired_video import empty_video

        items = []
        for length, coarse in ((3, True), (5, False)):
            video = empty_video(length)
            video["structured_available"][0, 0, 98:140] = True
            video["segment_id"][0, 0] = 0
            video["frame_indices"][:] = 0
            if coarse:
                video["structured"][0, 0, 186:188] = torch.tensor([-9., -2.])
                video["structured_available"][0, 0, 186:188] = True
                video["structured"][0, 0, 190] = 1
                video["structured_available"][0, 0, 190:192] = True
            items.append({
                "features": torch.zeros(length, 16), "conditioning": torch.zeros(length, 12),
                "targets": {"note_onset": torch.ones(length, 6)},
                "masks": {"note_onset": torch.ones(length, 6, dtype=torch.bool)},
                "metadata": {}, "video": video,
            })
        batch = collate_windows(items)
        self.assertEqual(batch["video"]["structured"].shape, (2, 1, 4, 194))
        self.assertTrue(batch["video"]["structured_available"][1, 0, 0, 98:140].all())
        self.assertFalse(batch["video"]["structured_available"][1, ..., 186:].any())
        self.assertFalse(batch["masks"]["note_onset"][0, 3:].any())
        self.assertTrue(batch["masks"]["note_onset"][1].all())
        self.assertEqual(batch["video"]["frame_indices"][0].tolist(), [0, 0, 0, -1, -1])


if __name__ == "__main__":
    unittest.main()
