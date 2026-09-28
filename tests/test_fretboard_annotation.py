import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.fretboard_annotation import (
    HTML,
    KEYPOINTS,
    export_yolo,
    load_predictions,
    mark_prediction_review,
    normalize_annotation,
    normalize_preferences,
    normalize_prediction,
    prefill_predictions,
    refresh_dataset_metadata,
    timestamped_source_url,
    _source_identifier,
    _hand_evidence_times,
    _sample_evidence_times,
    _candidate_times,
    select_diverse_candidates,
    parser,
    validate_geometry,
)


def annotation():
    coordinates = (
        (.30, .20), (.30, .25),
        (.50, .18), (.50, .27),
        (.85, .12), (.85, .33),
        (.40, .225),
    )
    return {
        "points": [
            {"x": x, "y": y, "visibility": 2}
            for x, y in coordinates
        ],
        "complete": True,
        "note": "",
    }


class FretboardAnnotationTests(unittest.TestCase):
    def test_ui_uses_human_landmark_names_and_prominent_save_state(self):
        self.assertIn("Bridge saddle contact — High E / String 1", HTML)
        self.assertIn("12th fret wire — Low E / String 6", HTML)
        self.assertIn("5th fret wire — center", HTML)
        self.assertIn("silver fret wire, not the space between wires", HTML)
        self.assertIn("function timestamp(seconds)", HTML)
        self.assertIn("${timestamp(r.sourceSeconds)}", HTML)
        self.assertIn('id="songLink"', HTML)
        self.assertIn("if(r.sourceUrl)link.href=sourceUrl(r.sourceUrl,r.sourceSeconds)", HTML)
        self.assertIn('else link.removeAttribute("href")', HTML)
        self.assertIn('id="acceptPrediction"', HTML)
        self.assertIn("prediction()?.points", HTML)
        self.assertIn("confidence ${p.confidence.toFixed(3)}", HTML)
        self.assertIn('id="saveState"', HTML)
        self.assertIn('id="progress"', HTML)
        self.assertIn('e.key=="Enter"', HTML)
        self.assertIn('e.key=="ArrowLeft"', HTML)
        self.assertIn('e.key=="ArrowRight"', HTML)
        self.assertIn('$("next").style.visibility=index==state.manifest.records.length-1?"hidden":"visible"', HTML)
        self.assertIn('if(value?.complete)return "complete"', HTML)
        self.assertIn('id="opacity"', HTML)
        self.assertIn('fetch("/api/preferences"', HTML)
        self.assertIn("ctx.globalAlpha=overlayOpacity", HTML)
        self.assertIn('id="dotRadius"', HTML)
        self.assertIn("ctx.arc(X,Y,dotRadius,0", HTML)
        self.assertIn('if(i==selected){ctx.strokeStyle="#000"', HTML)
        self.assertNotIn("dotRadius+(i==selected", HTML)
        self.assertIn("function placeAt(e)", HTML)
        self.assertIn("else if(placing){placeAt(e)}", HTML)
        self.assertNotIn("selectPoint(Math.min(5,selected+1))", HTML)
        self.assertIn("ctx.moveTo(left,cursorY)", HTML)
        self.assertIn("ctx.moveTo(cursorX,top)", HTML)
        self.assertIn('ctx.lineWidth=1/devicePixelRatio;ctx.strokeStyle="#FFF"', HTML)
        self.assertNotIn('ctx.lineWidth=3;ctx.strokeStyle="rgba(0,0,0,.7)"', HTML)
        self.assertIn("background:#fff;color:#111", HTML)
        self.assertNotIn("background:#111;color:#eee", HTML)
        self.assertIn(".pointButton.available{background:#e6f4ea}", HTML)
        self.assertIn(".pointButton.occluded{background:#fff4ce}", HTML)
        self.assertIn(".pointButton.unavailable{background:#fde7e9}", HTML)
        self.assertIn("<strong>Available:</strong>", HTML)
        self.assertIn("<strong>Occluded:</strong>", HTML)
        self.assertIn("<strong>Unavailable:</strong>", HTML)
        self.assertIn("It has no effect on training.", HTML)
        self.assertIn("Available (Q)", HTML)
        self.assertIn("Occluded (W)", HTML)
        self.assertIn("Unavailable (E)", HTML)
        self.assertIn("function setSelectedStatus(value)", HTML)
        self.assertIn("point.visibility=value", HTML)
        self.assertIn('$("clear").classList.toggle("active",mode==0)', HTML)
        self.assertIn("setMode(item().points[selected].visibility)", HTML)
        self.assertIn('if(mode==0){setStatus("Choose Available (Q) or Occluded (W)', HTML)
        self.assertIn('e.key.toLowerCase()=="q"', HTML)
        self.assertIn('e.key.toLowerCase()=="w"', HTML)
        self.assertIn('e.key.toLowerCase()=="e"', HTML)
        self.assertIn('id="captureArrows"', HTML)
        self.assertIn('&&captureArrows){e.preventDefault();e.stopPropagation()', HTML)
        self.assertIn("document.activeElement.blur()", HTML)
        self.assertNotIn("localStorage", HTML)
        self.assertNotIn("`${i+1}: ${n}`", HTML)

    def test_preferences_are_validated_for_local_file_storage(self):
        expected = {
            "schemaVersion": 1,
            "kind": "fretboard-annotation-preferences",
            "overlayOpacity": .45,
            "captureArrowKeys": True,
            "dotRadius": 7.,
        }
        self.assertEqual(normalize_preferences(expected), expected)
        for changed in (
            {**expected, "overlayOpacity": 0},
            {**expected, "captureArrowKeys": "yes"},
            {**expected, "dotRadius": 21},
            {**expected, "extra": True},
        ):
            with self.assertRaises(ValueError):
                normalize_preferences(changed)

    def test_diversity_selection_groups_sources_and_avoids_duplicates(self):
        candidates = []
        for video, offset in (("a", 0.), ("b", .4)):
            for index in range(4):
                feature = np.zeros(8, np.float32)
                feature[index + (0 if video == "a" else 4)] = 1
                candidates.append({
                    "id": f"{video}-{index}", "videoSha256": video,
                    "seconds": float(index), "feature": feature,
                    "quality": 1 - offset / 2,
                })
        selected = select_diverse_candidates(
            candidates, 6, minimum_per_video=1, maximum_per_video=3,
        )
        self.assertEqual(len(selected), 6)
        self.assertEqual({row["videoSha256"] for row in selected}, {"a", "b"})
        self.assertEqual(
            {video: sum(row["videoSha256"] == video for row in selected) for video in ("a", "b")},
            {"a": 3, "b": 3},
        )
        self.assertEqual(len({row["id"] for row in selected}), len(selected))

    def test_diversity_selection_can_extend_an_initial_selection(self):
        candidates = []
        for index in range(4):
            feature = np.zeros(4, np.float32)
            feature[index] = 1
            candidates.append({
                "id": f"a-{index}", "videoSha256": "a",
                "seconds": float(index), "feature": feature, "quality": 1.,
            })
        selected = select_diverse_candidates(
            candidates, 3, minimum_per_video=0, maximum_per_video=4,
            initial=[candidates[0]],
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({row["id"] for row in selected}), 3)

    def test_candidate_times_use_matching_shot_midpoints(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shots.json").write_text(json.dumps({
                "videoSha256": "a" * 64,
                "timeBase": [1, 1000],
                "shots": [
                    {"startPts": 0, "endPtsExclusive": 1000},
                    {"startPts": 1000, "endPtsExclusive": 3000},
                ],
            }))
            self.assertEqual(
                _candidate_times("a" * 64, 10., 4, root),
                [.5, 2.],
            )
            fallback = _candidate_times("b" * 64, 10., 2, root)
            self.assertEqual(len(fallback), 2)
            self.assertTrue(all(0 < value < 10 for value in fallback))

    def test_hand_evidence_sampling_prefers_two_hands_and_bounds_negatives(self):
        times = np.arange(10, dtype=np.float64) / 10
        counts = np.asarray([0, 1, 2, 2, 0, 1, 2, 2, 0, 1], np.int8)
        playing = _sample_evidence_times(times, counts, 3, hand_count=2)
        one_hand = _sample_evidence_times(times, counts, 3, hand_count=1)
        negatives = _sample_evidence_times(times, counts, 2, hand_count=0)
        self.assertEqual(len(playing), 3)
        self.assertTrue(all(count == 2 for _, count in playing))
        self.assertEqual(len(one_hand), 3)
        self.assertTrue(all(count == 1 for _, count in one_hand))
        self.assertEqual(len(negatives), 2)
        self.assertTrue(all(count == 0 for _, count in negatives))

    def test_hand_evidence_archive_uses_source_pts_and_counts(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "hands.npz"
            np.savez(path, pts=np.asarray([100, 200], np.int64), hand_count=np.asarray([1, 2], np.int8))
            times, counts = _hand_evidence_times({
                "arrays": path, "timeBase": [1, 1000], "frameCount": 2,
            })
            np.testing.assert_allclose(times, [.1, .2])
            np.testing.assert_array_equal(counts, [1, 2])

    def test_timestamped_source_url_preserves_existing_query(self):
        self.assertEqual(
            timestamped_source_url("https://youtu.be/example", 12.6),
            "https://youtu.be/example?t=13s",
        )
        self.assertEqual(
            timestamped_source_url("https://youtube.com/watch?v=x", 12),
            "https://youtube.com/watch?v=x&t=12s",
        )

    def test_source_identifier_uses_container_for_generic_source_filename(self):
        self.assertEqual(_source_identifier(Path("sources") / "giorno-demo" / "source.mkv"), "giorno-demo")
        self.assertEqual(_source_identifier(Path("sources") / "tab-0001" / "source.mkv"), "tab-0001")

    def test_refresh_dataset_metadata_repairs_generic_source_records(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({
                "kind": "fretboard-keypoint-annotation-dataset",
                "records": [{
                    "id": "frame",
                    "sourceVideo": str(Path("sources") / "giorno-demo" / "source.mkv"),
                    "sourceId": "source",
                    "sourceTitle": "source",
                    "sourceUrl": None,
                }],
            }))
            metadata = {
                "giorno-demo": {
                    "title": "Giorno's Theme",
                    "sourceUrl": "https://youtu.be/example",
                },
            }
            with patch("scripts.fretboard_annotation._source_metadata", return_value=metadata):
                manifest = refresh_dataset_metadata(root)
            record = manifest["records"][0]
            self.assertEqual(record["sourceId"], "giorno-demo")
            self.assertEqual(record["sourceTitle"], "Giorno's Theme")
            self.assertEqual(record["sourceUrl"], "https://youtu.be/example")

    def test_prediction_review_contract_tracks_pending_and_corrected_points(self):
        value = normalize_prediction({
            "points": annotation()["points"],
            "confidence": .75,
            "reason": "low-confidence",
            "review": "pending",
        })
        self.assertEqual(value["review"], "pending")
        self.assertEqual(value["confidence"], .75)
        with self.assertRaises(ValueError):
            normalize_prediction({**value, "review": "perfect"})

    def test_prediction_review_changes_only_when_frame_is_complete(self):
        prediction = {
            "points": annotation()["points"],
            "confidence": .75,
            "reason": "model-proposal",
            "review": "pending",
        }
        incomplete = annotation()
        incomplete["complete"] = False
        incomplete["points"][0] = {"x": .31, "y": .20, "visibility": 2}
        self.assertEqual(mark_prediction_review(prediction, incomplete)["review"], "pending")
        accepted = annotation()
        self.assertEqual(mark_prediction_review(prediction, accepted)["review"], "accepted")
        corrected = annotation()
        corrected["points"][0] = {"x": .31, "y": .20, "visibility": 2}
        self.assertEqual(mark_prediction_review(prediction, corrected)["review"], "corrected")

    def test_prediction_file_validates_provenance_and_items(self):
        document = {
            "schemaVersion": 1,
            "kind": "fretboard-keypoint-predictions",
            "keypoints": list(KEYPOINTS),
            "modelSha256": "a" * 64,
            "modelName": "best.pt",
            "runtime": {},
            "items": {
                "frame": {
                    "points": annotation()["points"],
                    "confidence": .75,
                    "reason": "model-proposal",
                    "review": "pending",
                },
            },
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            path.write_text(json.dumps(document))
            self.assertEqual(load_predictions(path, model_sha256="a" * 64)["items"]["frame"]["review"], "pending")
            with self.assertRaisesRegex(ValueError, "different model"):
                load_predictions(path, model_sha256="b" * 64)
            document["items"]["other"] = document["items"]["frame"]
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "outside this dataset"):
                load_predictions(path, record_ids={"frame"})
            path.write_text("{")
            with self.assertRaisesRegex(ValueError, "Invalid prediction review file"):
                load_predictions(path)

    def test_prefill_command_parses_model_runtime_options(self):
        args = parser().parse_args([
            "prefill", "--dataset", "dataset", "--model", "best.pt",
            "--device", "0", "--image-size", "3840", "--confidence", ".05",
        ])
        self.assertEqual(args.command, "prefill")
        self.assertEqual(args.device, "0")
        self.assertEqual(args.image_size, 3840)
        self.assertEqual(args.confidence, .05)

    def test_prefill_persists_resumable_model_proposals(self):
        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Scores:
            def argmax(self):
                return Scalar(0)

            def __getitem__(self, _index):
                return Scalar(.8)

        class Boxes:
            conf = Scores()

            def __len__(self):
                return 1

        class Tensor:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class KeypointData:
            def __getitem__(self, _index):
                coordinates = np.asarray([
                    [30., 20., .9], [30., 25., .9],
                    [50., 18., .9], [50., 27., .9],
                    [85., 12., .9], [85., 33., .9],
                    [40., 22.5, .9],
                ])
                return Tensor(coordinates)

        class FakeYolo:
            calls = 0

            def __init__(self, _path):
                self.model = SimpleNamespace(kpt_shape=[7, 3])

            def predict(self, *_args, **_kwargs):
                FakeYolo.calls += 1
                return [SimpleNamespace(
                    boxes=Boxes(),
                    keypoints=SimpleNamespace(data=KeypointData()),
                )]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            cv2.imwrite(str(root / "images" / "frame.jpg"), np.zeros((100, 100, 3), np.uint8))
            (root / "model.pt").write_bytes(b"model")
            (root / "manifest.json").write_text(json.dumps({
                "kind": "fretboard-keypoint-annotation-dataset",
                "keypoints": list(KEYPOINTS),
                "records": [{"id": "frame", "image": "images/frame.jpg"}],
            }))
            (root / "annotations.json").write_text(json.dumps({
                "kind": "fretboard-keypoint-annotations",
                "keypoints": list(KEYPOINTS),
                "items": {},
            }))
            modules = {
                "torch": SimpleNamespace(__version__="test"),
                "ultralytics": SimpleNamespace(__version__="test", YOLO=FakeYolo),
            }
            with patch.dict(sys.modules, modules):
                predictions = prefill_predictions(
                    root, root / "model.pt", image_size=640,
                )
                resumed = prefill_predictions(
                    root, root / "model.pt", image_size=640,
                )
            self.assertEqual(FakeYolo.calls, 1)
            self.assertEqual(predictions, resumed)
            self.assertEqual(predictions["items"]["frame"]["reason"], "model-proposal")
            self.assertEqual(predictions["items"]["frame"]["review"], "pending")
            self.assertEqual(len(predictions["items"]["frame"]["points"]), 7)

    def test_seven_point_contract_normalizes_complete_geometry(self):
        value = normalize_annotation(annotation())
        self.assertEqual(len(value["points"]), 7)
        self.assertTrue(value["complete"])
        validate_geometry(value["points"])

    def test_complete_geometry_allows_unavailable_out_of_frame_anchors(self):
        value = annotation()
        value["points"][4:6] = [
            {"x": None, "y": None, "visibility": 0},
            {"x": None, "y": None, "visibility": 0},
        ]
        normalized = normalize_annotation(value)
        self.assertTrue(normalized["complete"])
        self.assertEqual([point["visibility"] for point in normalized["points"]], [2, 2, 2, 2, 0, 0, 2])

    def test_complete_geometry_rejects_flipped_and_degenerate_points(self):
        value = annotation()
        value["points"][2], value["points"][3] = value["points"][3], value["points"][2]
        with self.assertRaisesRegex(ValueError, "ordering"):
            normalize_annotation(value)
        value = annotation()
        value["points"][1] = dict(value["points"][0])
        with self.assertRaisesRegex(ValueError, "too close"):
            normalize_annotation(value)

    def test_yolo_export_uses_seven_keypoints_and_preserves_incomplete_items(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "train").mkdir(parents=True)
            (root / "images" / "train" / "frame.jpg").write_bytes(b"jpeg")
            (root / "manifest.json").write_text(json.dumps({
                "schemaVersion": 1,
                "kind": "fretboard-keypoint-annotation-dataset",
                "keypoints": list(KEYPOINTS),
                "records": [{"id": "frame", "split": "train", "image": "images/train/frame.jpg"}],
            }))
            (root / "annotations.json").write_text(json.dumps({
                "schemaVersion": 1,
                "kind": "fretboard-keypoint-annotations",
                "keypoints": list(KEYPOINTS),
                "items": {"frame": annotation()},
            }))
            self.assertEqual(export_yolo(root), {"train": 1, "validation": 0, "test": 0})
            fields = (root / "labels" / "train" / "frame.txt").read_text().split()
            self.assertEqual(len(fields), 5 + 7 * 3)
            yaml = (root / "data.yaml").read_text()
            self.assertIn("kpt_shape: [7, 3]", yaml)
            self.assertIn("flip_idx: [0, 1, 2, 3, 4, 5, 6]", yaml)


if __name__ == "__main__":
    unittest.main()
