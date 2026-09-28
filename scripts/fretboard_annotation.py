"""Sample native video frames and annotate fretboard scale keypoints locally."""

from __future__ import annotations

import argparse
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import threading
import webbrowser

import cv2
import numpy as np


KEYPOINTS = (
    "nutString6", "nutString1",
    "fret12String6", "fret12String1",
    "bridgeString6", "bridgeString1",
    "fret5Center",
)
SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(path, value):
    path = Path(path)
    pending = path.with_name("." + path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def _split(digest):
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _videos(paths, directory):
    values = [Path(path).resolve() for path in paths]
    if directory is not None:
        values.extend(
            path.resolve()
            for path in Path(directory).rglob("*")
            if path.suffix.lower() in {".mkv", ".mp4", ".mov", ".webm"}
        )
    values = sorted(set(values))
    if not values or any(not path.is_file() for path in values):
        raise ValueError("Supply at least one existing video.")
    return values


def _source_metadata(pairs_path="data/pairs.json", source_map_path="runs/video-evidence/source-map-v2.json",
                     sources_directory="runs/video-evidence/sources"):
    result = {}
    catalog_path = Path(pairs_path)
    if catalog_path.is_file():
        try:
            document = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Invalid title catalog: {catalog_path}: {error}") from error
        rows = document.get("pairs")
        if not isinstance(rows, list):
            raise ValueError("Title catalog must contain a pairs list.")
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("title"), str):
                result.setdefault(row["id"], {})["title"] = row["title"]
    source_map = Path(source_map_path)
    if source_map.is_file():
        try:
            document = json.loads(source_map.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Invalid source map: {source_map}: {error}") from error
        for row in document.get("pairs", ()):
            identifier = row.get("pairId") if isinstance(row, dict) else None
            if isinstance(identifier, str):
                if isinstance(row.get("title"), str):
                    result.setdefault(identifier, {}).setdefault("title", row["title"])
                if isinstance(row.get("sourceUrl"), str):
                    result.setdefault(identifier, {})["sourceUrl"] = row["sourceUrl"]
    sources = Path(sources_directory)
    if sources.is_dir():
        for receipt_path in sorted(sources.glob("*/receipt.json")):
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError(f"Invalid source receipt: {receipt_path}: {error}") from error
            identifier, metadata = receipt.get("pairId"), receipt.get("metadata")
            if not isinstance(identifier, str) or not isinstance(metadata, dict):
                continue
            if isinstance(metadata.get("title"), str):
                result.setdefault(identifier, {})["title"] = metadata["title"]
            if isinstance(metadata.get("sourceUrl"), str):
                result.setdefault(identifier, {})["sourceUrl"] = metadata["sourceUrl"]
    return result


def timestamped_source_url(url, seconds):
    if not isinstance(url, str) or not url:
        return None
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={max(0, round(float(seconds)))}s"


def normalize_prediction(value):
    if not isinstance(value, dict) or set(value) != {
        "points", "confidence", "reason", "review",
    }:
        raise ValueError("Invalid prediction item.")
    points = value["points"]
    if not isinstance(points, list) or len(points) != len(KEYPOINTS):
        raise ValueError("Prediction uses a different keypoint contract.")
    points = [_point(point) for point in points]
    confidence = value["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
        or not isinstance(value["reason"], str)
        or value["review"] not in ("pending", "accepted", "corrected")
    ):
        raise ValueError("Invalid prediction confidence, reason, or review.")
    return {
        "points": points, "confidence": float(confidence),
        "reason": value["reason"][:200], "review": value["review"],
    }


def load_predictions(path, *, model_sha256=None, record_ids=None):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid prediction review file: {path}: {error}") from error
    if (
        value.get("schemaVersion") != SCHEMA_VERSION
        or value.get("kind") != "fretboard-keypoint-predictions"
        or value.get("keypoints") != list(KEYPOINTS)
        or not isinstance(value.get("modelSha256"), str)
        or len(value["modelSha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["modelSha256"])
        or not isinstance(value.get("modelName"), str)
        or not value["modelName"]
        or not isinstance(value.get("runtime"), dict)
        or not isinstance(value.get("items"), dict)
    ):
        raise ValueError("Invalid prediction review file.")
    if model_sha256 is not None and value["modelSha256"] != model_sha256:
        raise ValueError("Existing predictions use a different model; choose a new dataset copy.")
    items = value["items"]
    if any(not isinstance(identifier, str) for identifier in items):
        raise ValueError("Prediction frame identifiers must be text.")
    if record_ids is not None and not set(items) <= set(record_ids):
        raise ValueError("Predictions contain frames outside this dataset.")
    value["items"] = {
        identifier: normalize_prediction(prediction)
        for identifier, prediction in items.items()
    }
    return value


def mark_prediction_review(prediction, annotation):
    prediction = normalize_prediction(prediction)
    annotation = normalize_annotation(annotation)
    if not annotation["complete"]:
        prediction["review"] = "pending"
    else:
        prediction["review"] = (
            "accepted" if annotation["points"] == prediction["points"]
            else "corrected"
        )
    return prediction


def prefill_predictions(dataset, model_path, *, device="cpu", image_size=1920,
                        confidence=.01, incomplete_only=True):
    dataset, model_path = Path(dataset).resolve(), Path(model_path).resolve()
    manifest_path = dataset / "manifest.json"
    annotations_path = dataset / "annotations.json"
    if not manifest_path.is_file() or not annotations_path.is_file() or not model_path.is_file():
        raise ValueError("Prefill requires a dataset and local detector model.")
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install requirements-fretboard.txt before prediction prefill.") from error
    if type(image_size) is not int or image_size < 32:
        raise ValueError("Prediction image size must be an integer of at least 32.")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("Prediction confidence must be in [0, 1].")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    if (
        manifest.get("kind") != "fretboard-keypoint-annotation-dataset"
        or manifest.get("keypoints") != list(KEYPOINTS)
        or not isinstance(manifest.get("records"), list)
        or annotations.get("kind") != "fretboard-keypoint-annotations"
        or annotations.get("keypoints") != list(KEYPOINTS)
        or not isinstance(annotations.get("items"), dict)
    ):
        raise ValueError("Dataset keypoints differ from the current seven-point contract.")
    model = YOLO(str(model_path))
    if list(getattr(model.model, "kpt_shape", ())) != [len(KEYPOINTS), 3]:
        raise ValueError("Detector model must emit seven keypoints with visibility.")
    predictions_path = dataset / "predictions.json"
    model_sha256 = _sha256(model_path)
    if predictions_path.exists():
        predictions = load_predictions(
            predictions_path, model_sha256=model_sha256,
            record_ids={record["id"] for record in manifest["records"]},
        )
    else:
        predictions = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "fretboard-keypoint-predictions",
            "keypoints": list(KEYPOINTS),
            "modelSha256": model_sha256,
            "modelName": model_path.name,
            "runtime": {
                "python": sys.version, "platform": platform.platform(),
                "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "device": device, "imageSize": image_size, "confidence": confidence,
            },
            "items": {},
        }
    records = manifest["records"]
    for index, record in enumerate(records, 1):
        if incomplete_only and annotations["items"].get(record["id"], {}).get("complete"):
            continue
        if record["id"] in predictions["items"]:
            continue
        image = cv2.imread(str(dataset / record["image"]))
        if image is None:
            raise ValueError(f"Could not read annotation image: {record['image']}")
        result = model.predict(
            image, verbose=False, conf=confidence,
            imgsz=min(image_size, max(image.shape[:2])), device=device,
        )[0]
        points = [{"x": None, "y": None, "visibility": 0} for _ in KEYPOINTS]
        score = 0.
        available = 0
        if result.boxes is not None and result.keypoints is not None and len(result.boxes):
            detection = int(result.boxes.conf.argmax().item())
            score = float(result.boxes.conf[detection].item())
            data = result.keypoints.data[detection].detach().cpu().numpy()
            for point_index, row in enumerate(data):
                if len(row) >= 3 and row[2] >= .25 and 0 <= row[0] < image.shape[1] and 0 <= row[1] < image.shape[0]:
                    points[point_index] = {
                        "x": float(row[0] / image.shape[1]),
                        "y": float(row[1] / image.shape[0]),
                        "visibility": 2,
                    }
                    available += 1
        reason = "miss" if not available else "partial-keypoints" if available < len(KEYPOINTS) else "low-confidence" if score < .5 else "model-proposal"
        predictions["items"][record["id"]] = {
            "points": points, "confidence": score, "reason": reason, "review": "pending",
        }
        _publish(predictions_path, predictions)
        print(f"Prediction prefill: {index}/{len(records)} | {reason} | {record['id']}", flush=True)
    return predictions


def _source_titles(catalog_path):
    if catalog_path is None or not Path(catalog_path).is_file():
        return {}
    try:
        document = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid title catalog: {catalog_path}: {error}") from error
    rows = document.get("pairs")
    if not isinstance(rows, list):
        raise ValueError("Title catalog must contain a pairs list.")
    return {
        row["id"]: row["title"]
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("id"), str)
        and isinstance(row.get("title"), str)
    }


def _source_identifier(video):
    parts = Path(video).parts
    matched = next((part for part in reversed(parts) if part.startswith(("tab-", "rc-tab-"))), None)
    if matched is not None:
        return matched
    path = Path(video)
    return path.parent.name if path.stem.lower() == "source" else path.stem


def refresh_dataset_metadata(dataset):
    dataset = Path(dataset).resolve()
    manifest_path = dataset / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Annotation dataset requires manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("kind") != "fretboard-keypoint-annotation-dataset"
        or not isinstance(manifest.get("records"), list)
    ):
        raise ValueError("Invalid annotation dataset manifest.")
    metadata = _source_metadata()
    for record in manifest["records"]:
        identifier = _source_identifier(record["sourceVideo"])
        source = metadata.get(identifier, {})
        record["sourceId"] = identifier
        record["sourceTitle"] = source.get("title", identifier)
        record["sourceUrl"] = source.get("sourceUrl")
    _publish(manifest_path, manifest)
    return manifest


def _frame_feature(frame):
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).ravel()
    histogram /= max(float(histogram.sum()), 1.)
    appearance = cv2.resize(gray, (16, 9), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    feature = np.concatenate((histogram.astype(np.float32), appearance))
    feature /= max(float(np.linalg.norm(feature)), 1e-8)
    return feature, float(cv2.Laplacian(gray, cv2.CV_32F).var()), float(gray.mean())


def select_diverse_candidates(candidates, target, *, minimum_per_video=1, maximum_per_video=12,
                              initial=()):
    if type(target) is not int or target < 1:
        raise ValueError("target must be a positive integer.")
    if type(minimum_per_video) is not int or type(maximum_per_video) is not int or not 0 <= minimum_per_video <= maximum_per_video:
        raise ValueError("Invalid per-video selection bounds.")
    if not candidates:
        raise ValueError("No usable frame candidates were found.")
    by_video = {}
    for candidate in candidates:
        feature = np.asarray(candidate["feature"], dtype=np.float32)
        if feature.ndim != 1 or not np.isfinite(feature).all():
            raise ValueError("Candidate features must be finite vectors.")
        by_video.setdefault(candidate["videoSha256"], []).append(candidate)
    target = min(target, len(candidates))
    selected = [dict(row) for row in initial]
    counts = {key: 0 for key in by_video}
    for row in selected:
        counts[row["videoSha256"]] = counts.get(row["videoSha256"], 0) + 1
    if len(selected) >= target:
        return selected[:target]
    selected_ids = {row["id"] for row in selected}
    for digest, rows in sorted(by_video.items()):
        needed = max(0, minimum_per_video - counts.get(digest, 0))
        ordered = sorted(rows, key=lambda row: (-row["quality"], row["seconds"]))
        for row in (value for value in ordered if value["id"] not in selected_ids):
            if not needed:
                break
            selected.append({**row, "selectionReason": "per-video-representative"})
            selected_ids.add(row["id"])
            counts[digest] += 1
            needed -= 1
            if len(selected) == target:
                return selected
    while len(selected) < target:
        best = None
        for candidate in candidates:
            digest = candidate["videoSha256"]
            if candidate["id"] in selected_ids or counts[digest] >= maximum_per_video:
                continue
            distance = min(
                (1 - float(np.dot(candidate["feature"], chosen["feature"])) for chosen in selected),
                default=1.,
            )
            score = distance + .05 * candidate["quality"]
            key = score, candidate["quality"], candidate["id"]
            if best is None or key > best[0]:
                best = key, candidate
        if best is None:
            break
        row = {**best[1], "selectionReason": "visual-diversity"}
        selected.append(row)
        selected_ids.add(row["id"])
        counts[row["videoSha256"]] += 1
    return selected


def _hand_evidence_catalog(directory):
    if directory is None:
        return {}
    catalog = {}
    for report_path in sorted(Path(directory).rglob("hands.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            digest = report.get("videoSha256")
            arrays_path = report_path.with_name(report.get("arrays", ""))
            time_base = report.get("timeBase")
            count = report.get("frameCount")
            if (
                report.get("kind") != "fingerstyle-hand-observations"
                or not isinstance(digest, str)
                or not arrays_path.is_file()
                or not isinstance(time_base, list) or len(time_base) != 2
                or any(type(value) is not int or value <= 0 for value in time_base)
                or type(count) is not int or count <= 0
            ):
                continue
            previous = catalog.get(digest)
            if previous is None or count > previous["frameCount"]:
                catalog[digest] = {
                    "report": report_path, "arrays": arrays_path,
                    "timeBase": time_base, "frameCount": count,
                }
        except (OSError, ValueError, TypeError):
            continue
    return catalog


def _hand_evidence_times(entry):
    if entry is None:
        return None
    try:
        with np.load(entry["arrays"], allow_pickle=False) as source:
            pts = source["pts"]
            counts = source["hand_count"]
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"Invalid hand evidence archive: {entry['arrays']}: {error}") from error
    if (
        pts.dtype != np.int64 or counts.dtype != np.int8
        or pts.shape != (entry["frameCount"],) or counts.shape != pts.shape
        or (np.diff(pts) <= 0).any() or not np.isin(counts, (0, 1, 2)).all()
    ):
        raise ValueError(f"Invalid hand evidence timeline: {entry['arrays']}")
    scale = entry["timeBase"][0] / entry["timeBase"][1]
    return pts.astype(np.float64) * scale, counts


def _sample_evidence_times(times, counts, count, *, hand_count):
    selected = np.flatnonzero(counts == hand_count)
    if not len(selected):
        return []
    indices = np.linspace(0, len(selected) - 1, min(count, len(selected))).round().astype(int)
    return [(float(times[selected[index]]), int(counts[selected[index]])) for index in indices]


def _candidate_times(video_digest, duration, count, shots_directory):
    if shots_directory is not None:
        for path in sorted(Path(shots_directory).rglob("shots.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if document.get("videoSha256") != video_digest:
                continue
            rows = document.get("shots")
            time_base = document.get("timeBase")
            if (
                isinstance(rows, list) and rows
                and isinstance(time_base, list) and len(time_base) == 2
                and all(type(value) is int and value > 0 for value in time_base)
            ):
                scale = time_base[0] / time_base[1]
                values = [
                    (row["startPts"] + row["endPtsExclusive"]) * scale / 2
                    for row in rows
                    if isinstance(row, dict)
                    and type(row.get("startPts")) is int
                    and type(row.get("endPtsExclusive")) is int
                    and row["startPts"] < row["endPtsExclusive"]
                ]
                if values:
                    if len(values) <= count:
                        return values
                    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
                    return [values[index] for index in indices]
    return [
        duration * (.01 + .98 * (ordinal + .5) / count)
        for ordinal in range(count)
    ]


def select_dataset(output, videos, *, target_frames=1000, candidates_per_video=24,
                   minimum_per_video=1, maximum_per_video=12, jpeg_quality=95,
                   shots_directory=None, hands_directory=None, negative_fraction=.05,
                   one_hand_fraction=.30,
                   cache_directory=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite annotation dataset: {output}")
    if type(candidates_per_video) is not int or candidates_per_video < 2:
        raise ValueError("candidates_per_video must be an integer >= 2.")
    if (
        isinstance(negative_fraction, bool)
        or not isinstance(negative_fraction, (int, float))
        or not math.isfinite(negative_fraction)
        or not 0 <= negative_fraction < .5
    ):
        raise ValueError("negative_fraction must be from zero through less than one half.")
    if (
        isinstance(one_hand_fraction, bool)
        or not isinstance(one_hand_fraction, (int, float))
        or not math.isfinite(one_hand_fraction)
        or not 0 < one_hand_fraction < 1 - negative_fraction
    ):
        raise ValueError("one_hand_fraction must be positive and leave room for two-hand frames.")
    hand_catalog = _hand_evidence_catalog(hands_directory)
    metadata = _source_metadata()
    cache_root = None if cache_directory is None else Path(cache_directory).resolve()
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    candidates = []
    for video_index, video in enumerate(videos, 1):
        digest = _sha256(video)
        cache_path = None if cache_root is None else cache_root / f"{digest}.npz"
        cached_candidates = []
        cache_complete = False
        if cache_path is not None and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    if (
                        int(cached["candidates_per_video"]) != candidates_per_video
                        or float(cached["negative_fraction"]) != float(negative_fraction)
                    ):
                        raise ValueError("Cached selector settings differ.")
                    for ordinal in range(len(cached["seconds"])):
                        cached_candidates.append({
                            "id": f"{digest[:12]}-{int(cached['ordinals'][ordinal]):04d}",
                            "video": video, "videoSha256": digest,
                            "seconds": float(cached["seconds"][ordinal]),
                            "width": int(cached["widths"][ordinal]),
                            "height": int(cached["heights"][ordinal]),
                            "feature": cached["features"][ordinal],
                            "quality": float(cached["qualities"][ordinal]),
                            "handCount": None if cached["hand_counts"][ordinal] < 0 else int(cached["hand_counts"][ordinal]),
                        })
                    cache_complete = (
                        "one_hand_fraction" in cached.files
                        and float(cached["one_hand_fraction"]) == float(one_hand_fraction)
                    )
                if cache_complete:
                    candidates.extend(cached_candidates)
                    print(f"Selection candidates: {video_index}/{len(videos)} cached | {video.name}", flush=True)
                    continue
            except (OSError, ValueError, KeyError):
                cache_path.unlink(missing_ok=True)
                cached_candidates = []
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
                raise ValueError(f"Video has no reliable duration: {video}")
            duration = frame_count / fps
            hand_evidence = _hand_evidence_times(hand_catalog.get(digest))
            if hands_directory is not None and hand_evidence is None:
                continue
            if hand_evidence is None:
                times = [(value, None) for value in _candidate_times(digest, duration, candidates_per_video, shots_directory)]
            else:
                evidence_times, hand_counts = hand_evidence
                negative_slots = max(1, round(candidates_per_video * negative_fraction))
                positive_slots = candidates_per_video - negative_slots
                one_slots = max(1, round(candidates_per_video * one_hand_fraction))
                two_slots = max(1, positive_slots - one_slots)
                two_hands = _sample_evidence_times(
                    evidence_times, hand_counts, two_slots, hand_count=2,
                )
                one_hand = _sample_evidence_times(
                    evidence_times, hand_counts, one_slots, hand_count=1,
                )
                unused = positive_slots - len(two_hands) - len(one_hand)
                if unused > 0:
                    extra_count = 1 if len(two_hands) < two_slots else 2
                    existing = {seconds for seconds, _ in (*two_hands, *one_hand)}
                    extras = [
                        value for value in _sample_evidence_times(
                            evidence_times, hand_counts, positive_slots, hand_count=extra_count,
                        )
                        if value[0] not in existing
                    ][:unused]
                else:
                    extras = []
                negatives = _sample_evidence_times(
                    evidence_times, hand_counts,
                    negative_slots, hand_count=0,
                )
                times = two_hands + one_hand + extras + negatives
                if cached_candidates:
                    existing = {(round(row["seconds"], 6), row["handCount"]) for row in cached_candidates}
                    times = [
                        value for value in one_hand
                        if (round(value[0], 6), value[1]) not in existing
                    ]
            video_candidates = list(cached_candidates)
            for ordinal, (seconds, hand_count) in enumerate(times):
                capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                ok, frame = capture.read()
                if not ok:
                    continue
                feature, sharpness, brightness = _frame_feature(frame)
                if sharpness < 8 or not 8 <= brightness <= 247:
                    continue
                quality = min(1., sharpness / 250) * min(1., brightness / 50, (255 - brightness) / 50)
                if hand_count is not None:
                    quality *= 1 + .2 * hand_count
                video_candidates.append({
                    "id": f"{digest[:12]}-{len(video_candidates):04d}",
                    "video": video,
                    "videoSha256": digest,
                    "seconds": seconds,
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "feature": feature,
                    "quality": quality,
                    "handCount": hand_count,
                })
            candidates.extend(video_candidates)
            if cache_path is not None:
                pending = cache_path.with_name("." + cache_path.name + ".pending")
                with pending.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        candidates_per_video=np.asarray(candidates_per_video, np.int32),
                        negative_fraction=np.asarray(negative_fraction, np.float64),
                        one_hand_fraction=np.asarray(one_hand_fraction, np.float64),
                        ordinals=np.asarray([
                            int(row["id"].rsplit("-", 1)[1]) for row in video_candidates
                        ], np.int32),
                        seconds=np.asarray([row["seconds"] for row in video_candidates], np.float64),
                        widths=np.asarray([row["width"] for row in video_candidates], np.int32),
                        heights=np.asarray([row["height"] for row in video_candidates], np.int32),
                        features=np.asarray([row["feature"] for row in video_candidates], np.float32),
                        qualities=np.asarray([row["quality"] for row in video_candidates], np.float32),
                        hand_counts=np.asarray([
                            -1 if row["handCount"] is None else row["handCount"] for row in video_candidates
                        ], np.int8),
                    )
                pending.replace(cache_path)
        finally:
            capture.release()
        print(f"Selection candidates: {video_index}/{len(videos)} processed | {video.name}", flush=True)
    known = hands_directory is not None
    positives = [row for row in candidates if row["handCount"] is None or row["handCount"] > 0]
    one_hand = [row for row in candidates if row["handCount"] == 1]
    negatives = [row for row in candidates if row["handCount"] == 0]
    negative_target = min(len(negatives), round(target_frames * negative_fraction)) if known else 0
    positive_target = target_frames - negative_target
    represented_videos = len({row["videoSha256"] for row in positives})
    selected = select_diverse_candidates(
        positives, min(positive_target, represented_videos),
        minimum_per_video=minimum_per_video, maximum_per_video=1,
    )
    one_hand_target = min(len(one_hand), round(target_frames * one_hand_fraction))
    current_one_hand = sum(row["handCount"] == 1 for row in selected)
    one_hand_total = min(
        positive_target,
        len(selected) + max(0, one_hand_target - current_one_hand),
    )
    selected = select_diverse_candidates(
        one_hand, one_hand_total,
        minimum_per_video=0, maximum_per_video=maximum_per_video,
        initial=selected,
    )
    selected = select_diverse_candidates(
        positives, positive_target, minimum_per_video=0,
        maximum_per_video=maximum_per_video, initial=selected,
    )
    if negative_target:
        selected.extend(select_diverse_candidates(
            negatives, negative_target,
            minimum_per_video=0, maximum_per_video=1,
        ))
    selected.sort(key=lambda row: (row["videoSha256"], row["seconds"], row["id"]))
    output.mkdir(parents=True)
    try:
        records = []
        rows_by_video = {}
        for row in selected:
            rows_by_video.setdefault(row["video"], []).append(row)
        exported = 0
        for video_index, (video, rows) in enumerate(sorted(rows_by_video.items(), key=lambda item: str(item[0])), 1):
            capture = cv2.VideoCapture(str(video))
            try:
                for row in sorted(rows, key=lambda value: value["seconds"]):
                    seconds = row["seconds"]
                    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                    ok, frame = capture.read()
                    if not ok:
                        capture.release()
                        capture = cv2.VideoCapture(str(video))
                        capture.set(cv2.CAP_PROP_POS_MSEC, max(0., seconds - .05) * 1000)
                        ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Could not decode selected frame: {video} at {seconds:.3f}s.")
                    split = _split(row["videoSha256"])
                    relative = Path("images") / split / f"{row['id']}.jpg"
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                        raise OSError(f"Could not save selected frame: {destination}")
                    records.append({
                        "id": row["id"], "split": split, "image": relative.as_posix(),
                        "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                        "sourceVideo": str(video), "sourceVideoSha256": row["videoSha256"],
                        "sourceId": _source_identifier(video),
                        "sourceTitle": metadata.get(_source_identifier(video), {}).get("title", _source_identifier(video)),
                        "sourceUrl": metadata.get(_source_identifier(video), {}).get("sourceUrl"),
                        "sourceSeconds": seconds, "selectionReason": row["selectionReason"],
                        "selectionQuality": row["quality"], "handCount": row["handCount"],
                    })
                    exported += 1
            finally:
                capture.release()
            print(
                f"Selection export: {video_index}/{len(rows_by_video)} videos | "
                f"{exported}/{len(selected)} frames | {video.name}",
                flush=True,
            )
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "fretboard-keypoint-annotation-dataset",
            "keypoints": list(KEYPOINTS),
            "coordinateSpace": "normalized_full_frame",
            "nativeResolutionPreserved": True,
            "selection": {
                "method": "global-farthest-cosine-v1",
                "targetFrames": target_frames,
                "candidatesPerVideo": candidates_per_video,
                "minimumPerVideo": minimum_per_video,
                "maximumPerVideo": maximum_per_video,
                "shotReports": shots_directory is not None,
                "handEvidence": hands_directory is not None,
                "negativeFraction": float(negative_fraction),
                "oneHandFraction": float(one_hand_fraction),
            },
            "records": records,
        }
        _publish(output / "manifest.json", manifest)
        _publish(output / "annotations.json", {
            "schemaVersion": SCHEMA_VERSION, "kind": "fretboard-keypoint-annotations",
            "keypoints": list(KEYPOINTS), "items": {},
        })
        _publish(output / "preferences.json", {
            "schemaVersion": SCHEMA_VERSION, "kind": "fretboard-annotation-preferences",
            "overlayOpacity": 1., "captureArrowKeys": True, "dotRadius": 6.,
        })
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def initialize_dataset(output, videos, *, frames_per_video=5, jpeg_quality=95):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite annotation dataset: {output}")
    if type(frames_per_video) is not int or frames_per_video < 1:
        raise ValueError("frames_per_video must be a positive integer.")
    if type(jpeg_quality) is not int or not 80 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 80 and 100.")
    output.mkdir(parents=True)
    records = []
    try:
        metadata = _source_metadata()
        for video in videos:
            digest = _sha256(video)
            split = _split(digest)
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise ValueError(f"Could not open video: {video}")
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if not math.isfinite(fps) or fps <= 0 or count <= 0:
                    raise ValueError(f"Video has no reliable duration: {video}")
                duration = count / fps
                for sample in range(frames_per_video):
                    fraction = (sample + .5) / frames_per_video
                    seconds = duration * (.025 + .95 * fraction)
                    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Could not decode {video} at {seconds:.3f}s.")
                    identifier = f"{digest[:12]}-{sample:03d}"
                    relative = Path("images") / split / f"{identifier}.jpg"
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(
                        str(destination), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                    ):
                        raise OSError(f"Could not save sampled frame: {destination}")
                    records.append({
                        "id": identifier,
                        "split": split,
                        "image": relative.as_posix(),
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                        "sourceVideo": str(video),
                        "sourceId": _source_identifier(video),
                        "sourceTitle": metadata.get(_source_identifier(video), {}).get("title", _source_identifier(video)),
                        "sourceUrl": metadata.get(_source_identifier(video), {}).get("sourceUrl"),
                        "sourceVideoSha256": digest,
                        "sourceSeconds": seconds,
                    })
            finally:
                capture.release()
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "fretboard-keypoint-annotation-dataset",
            "keypoints": list(KEYPOINTS),
            "coordinateSpace": "normalized_full_frame",
            "nativeResolutionPreserved": True,
            "records": records,
        }
        annotations = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "fretboard-keypoint-annotations",
            "keypoints": list(KEYPOINTS),
            "items": {},
        }
        preferences = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "fretboard-annotation-preferences",
            "overlayOpacity": 1.,
            "captureArrowKeys": True,
            "dotRadius": 6.,
        }
        _publish(output / "manifest.json", manifest)
        _publish(output / "annotations.json", annotations)
        _publish(output / "preferences.json", preferences)
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def _point(value):
    if not isinstance(value, dict) or set(value) != {"x", "y", "visibility"}:
        raise ValueError("Each keypoint requires x, y, and visibility.")
    visibility = value["visibility"]
    if visibility not in (0, 1, 2):
        raise ValueError("Keypoint visibility must be 0, 1, or 2.")
    if visibility == 0:
        if value["x"] is not None or value["y"] is not None:
            raise ValueError("Unavailable keypoints must have null coordinates.")
        return {"x": None, "y": None, "visibility": 0}
    x, y = value["x"], value["y"]
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or not 0 <= item <= 1 for item in (x, y)):
        raise ValueError("Visible keypoint coordinates must be finite values in [0, 1].")
    return {"x": float(x), "y": float(y), "visibility": visibility}


def normalize_annotation(value):
    if not isinstance(value, dict) or set(value) != {"points", "complete", "note"}:
        raise ValueError("Annotation requires points, complete, and note.")
    points = value["points"]
    if not isinstance(points, list) or len(points) != len(KEYPOINTS):
        raise ValueError(f"Annotation requires {len(KEYPOINTS)} ordered keypoints.")
    points = [_point(point) for point in points]
    if type(value["complete"]) is not bool or not isinstance(value["note"], str):
        raise ValueError("Annotation complete must be boolean and note must be text.")
    if value["complete"]:
        validate_geometry(points)
    return {"points": points, "complete": value["complete"], "note": value["note"][:1000]}


def normalize_preferences(value):
    if not isinstance(value, dict):
        raise ValueError("Invalid annotation preferences.")
    allowed = {
        "schemaVersion", "kind", "overlayOpacity", "captureArrowKeys",
        "disableArrowNavigation", "dotRadius",
    }
    if not {"schemaVersion", "kind", "overlayOpacity"} <= set(value) or set(value) - allowed:
        raise ValueError("Invalid annotation preferences.")
    legacy = "captureArrowKeys" not in value
    opacity = value["overlayOpacity"]
    capture = True if legacy else value["captureArrowKeys"]
    radius = value.get("dotRadius", 6.)
    if (
        value["schemaVersion"] != SCHEMA_VERSION
        or value["kind"] != "fretboard-annotation-preferences"
        or isinstance(opacity, bool)
        or not isinstance(opacity, (int, float))
        or not math.isfinite(opacity)
        or not .1 <= opacity <= 1
        or (legacy and type(value["disableArrowNavigation"]) is not bool)
        or type(capture) is not bool
        or isinstance(radius, bool)
        or not isinstance(radius, (int, float))
        or not math.isfinite(radius)
        or not 2 <= radius <= 20
    ):
        raise ValueError("Invalid annotation preferences.")
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "fretboard-annotation-preferences",
        "overlayOpacity": float(opacity),
        "captureArrowKeys": capture,
        "dotRadius": float(radius),
    }


def validate_geometry(points):
    available_pairs = [
        (index // 2, points[index], points[index + 1])
        for index in range(0, 6, 2)
        if points[index]["visibility"] and points[index + 1]["visibility"]
    ]
    vectors = []
    centers = []
    anchors = []
    for anchor, left, right in available_pairs:
        vector = (right["x"] - left["x"], right["y"] - left["y"])
        if math.hypot(*vector) < .002:
            raise ValueError("Outer string points at one anchor are too close.")
        vectors.append(vector)
        centers.append(((left["x"] + right["x"]) / 2, (left["y"] + right["y"]) / 2))
        anchors.append(anchor)
    if len(vectors) > 1 and any(
        vector[0] * vectors[0][0] + vector[1] * vectors[0][1] <= 0
        for vector in vectors[1:]
    ):
        raise ValueError("String 6 and string 1 ordering flips between anchors.")
    for left, right in zip(centers, centers[1:]):
        if math.dist(left, right) < .01:
            raise ValueError("Available anchor centers must be distinct.")
    for index in range(0, 6, 2):
        left, right = points[index:index + 2]
        if bool(left["visibility"]) != bool(right["visibility"]):
            # One-sided visibility is valid supervision, but cannot define a
            # cross-board anchor for geometry preview or ordering checks.
            continue


def export_yolo(dataset):
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    annotations = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    items = annotations["items"]
    counts = {split: 0 for split in SPLITS}
    for record in manifest["records"]:
        value = items.get(record["id"])
        if value is None or not value.get("complete"):
            continue
        value = normalize_annotation(value)
        points = value["points"]
        visible = [point for point in points if point["visibility"]]
        label = dataset / "labels" / record["split"] / f"{record['id']}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        if not visible:
            label.write_text("", encoding="ascii")
            counts[record["split"]] += 1
            continue
        left = max(0., min(point["x"] for point in visible) - .02)
        top = max(0., min(point["y"] for point in visible) - .02)
        right = min(1., max(point["x"] for point in visible) + .02)
        bottom = min(1., max(point["y"] for point in visible) + .02)
        fields = ["0", f"{(left + right) / 2:.8f}", f"{(top + bottom) / 2:.8f}", f"{right - left:.8f}", f"{bottom - top:.8f}"]
        for point in points:
            fields.extend((
                "0" if point["x"] is None else f"{point['x']:.8f}",
                "0" if point["y"] is None else f"{point['y']:.8f}",
                str(point["visibility"]),
            ))
        label.write_text(" ".join(fields) + "\n", encoding="ascii")
        counts[record["split"]] += 1
    yaml = "\n".join((
        f"path: {dataset.as_posix()}",
        "train: images/train",
        "val: images/validation",
        "test: images/test",
        "kpt_shape: [7, 3]",
        "flip_idx: [0, 1, 2, 3, 4, 5, 6]",
        "names:",
        "  0: fretboard",
        "",
    ))
    (dataset / "data.yaml").write_text(yaml, encoding="utf-8")
    return counts


HTML = r"""<!doctype html>
<meta charset="utf-8"><title>Fretboard annotation</title>
<style>
*{box-sizing:border-box} html,body{margin:0;height:100%;background:#eee;color:#111;font:14px Arial,sans-serif}
button,input,textarea{font:inherit} button{border:1px solid #999;border-radius:2px;background:#f4f4f4;color:#111;padding:7px 10px;cursor:pointer} button:hover{background:#e7e7e7}
#top{height:52px;display:flex;align-items:center;gap:8px;padding:0 10px;background:#fff;border-bottom:1px solid #aaa}
#counter{font-weight:bold} #songLink{font-weight:bold;color:#0645ad} #saveState{margin-left:auto;font-weight:bold;color:#176b2c}
#progressWrap{position:absolute;top:52px;left:0;right:0;height:25px;padding:4px 6px;background:#ddd;border-bottom:1px solid #aaa;overflow-x:auto;overflow-y:hidden}
#progress{height:17px;display:flex;min-width:max-content}.progressFrame{min-width:4px;flex:1 0 4px;height:17px;padding:0;border:0;border-right:1px solid #111;border-radius:0}
.progressFrame.complete{background:#3fb950}.progressFrame.started{background:#d29922}.progressFrame.untouched{background:#f85149}.progressFrame.current{outline:2px solid white;outline-offset:-2px;position:relative;z-index:1}
.progressFrame.predicted{background:#8b5cf6}
#side{position:absolute;top:77px;bottom:0;left:0;width:320px;padding:12px;background:#fff;border-right:1px solid #aaa;overflow:auto}
#wrap{position:absolute;top:77px;bottom:0;left:320px;right:0;overflow:hidden;background:#ccc}
canvas{width:100%;height:100%;cursor:crosshair}
h2{font-size:15px;margin:0 0 7px}.help{color:#555;line-height:1.35;margin:0 0 10px}
#pointButtons{display:grid;gap:4px}.pointButton{text-align:left;padding:7px 8px;border-left:5px solid var(--color);background:#fff}
.pointButton.available{background:#e6f4ea}.pointButton.occluded{background:#fff4ce}.pointButton.unavailable{background:#fde7e9}
.pointButton.active{outline:2px solid #111;outline-offset:-2px}.pointButton small{display:block;color:#444;margin-top:2px}
.section{margin-top:14px;padding-top:11px;border-top:1px solid #ccc}.modes{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.modes button.active,#clear.active{outline:2px solid #2563eb;background:#dbeafe}#clear{width:100%;margin-top:5px}
.complete{display:flex;gap:7px;align-items:flex-start;font-weight:bold;line-height:1.3}.complete input{margin-top:2px}
#note{width:100%;height:65px;resize:vertical;background:#fff;color:#111;border:1px solid #999;border-radius:0;padding:6px}
.rangeRow{display:flex;align-items:center;gap:10px}.rangeRow input{flex:1}.rangeRow output{min-width:42px;text-align:right;font-variant-numeric:tabular-nums}
</style>
<div id="top"><button id="prev">Previous ←</button><button id="next">Next →</button><a id="songLink" target="_blank" rel="noopener noreferrer"></a><span id="counter"></span><span id="saveState">Ready</span></div>
<div id="progressWrap"><div id="progress"></div></div>
<aside id="side">
 <h2>Points</h2>
 <p class="help">Select a point, then click or drag it into position. “Fret” means the silver fret wire, not the space between wires. Use physical strings 6 and 1 regardless of tuning. Place nut and bridge points at the string contact points.</p>
 <div id="pointButtons"></div>
 <div id="predictionPanel" class="section" hidden><h2>Model prediction</h2><p id="predictionInfo" class="help"></p><button id="acceptPrediction">Accept prediction and complete</button></div>
 <div class="section"><h2>Point status</h2><p class="help"><strong>Available:</strong> the exact point is directly visible.<br><strong>Occluded:</strong> blocked, but its exact position is still confidently identifiable.<br><strong>Unavailable:</strong> outside the frame or not confidently identifiable.</p><div class="modes"><button id="visible">Available (Q)</button><button id="occluded">Occluded (W)</button></div><button id="clear">Unavailable (E)</button></div>
 <div class="section"><h2>Overlay opacity</h2><div class="rangeRow"><input id="opacity" type="range" min="10" max="100" step="5" value="100"><output id="opacityValue">100%</output></div></div>
 <div class="section"><h2>Landmark dot radius</h2><div class="rangeRow"><input id="dotRadius" type="range" min="2" max="20" step="1" value="6"><output id="dotRadiusValue">6 px</output></div></div>
 <div class="section"><label class="complete"><input id="captureArrows" type="checkbox" checked><span>Always use ← / → for frame navigation<br><small>Focused checkboxes, sliders, buttons, and text fields will not consume these keys.</small></span></label></div>
 <div class="section"><label class="complete"><input id="complete" type="checkbox"><span>Frame fully reviewed<br><small>Points outside the shot may remain unavailable.</small></span></label></div>
 <div class="section"><h2>Optional note</h2><p class="help">For human review only. It has no effect on training.</p><textarea id="note" placeholder="Describe ambiguity, capo, occlusion, or why points are unavailable."></textarea></div>
 <div class="section help">Enter: complete and advance<br>← / →: previous / next without completing<br>Q / W / E: available / occluded / unavailable<br>Wheel: zoom<br>Middle/right drag: pan<br>1-7: select landmark</div>
</aside>
<div id="wrap"><canvas id="canvas"></canvas></div>
<script>
const names = ["nutString6","nutString1","fret12String6","fret12String1","bridgeString6","bridgeString1","fret5Center"];
const labels = ["Nut contact — Low E / String 6","Nut contact — High E / String 1","12th fret wire — Low E / String 6","12th fret wire — High E / String 1","Bridge saddle contact — Low E / String 6","Bridge saddle contact — High E / String 1","5th fret wire — center"];
const colors = ["#ff5252","#ffca28","#66bb6a","#26c6da","#7e57c2","#ec407a","#1565c0"];
let state, index=0, selected=0, mode=2, image=new Image(), scale=1, ox=0, oy=0, panning=false, placing=false, last, cursorX=0, cursorY=0, cursorInside=false, overlayOpacity=1, dotRadius=6, captureArrows=true;
const $=id=>document.getElementById(id), canvas=$("canvas"), ctx=canvas.getContext("2d");
function blank(){return {points:names.map(()=>({x:null,y:null,visibility:0})),complete:false,note:""}}
function prediction(){return state.predictions?.items?.[state.manifest.records[index].id]||null}
function item(){let id=state.manifest.records[index].id;if(!state.annotations.items[id]){let p=prediction();state.annotations.items[id]=p?{points:structuredClone(p.points),complete:false,note:""}:blank()}return state.annotations.items[id]}
async function load(){state=await (await fetch("/api/state")).json();overlayOpacity=state.preferences.overlayOpacity;dotRadius=state.preferences.dotRadius;captureArrows=state.preferences.captureArrowKeys;$("opacity").value=Math.round(overlayOpacity*100);$("opacityValue").value=`${Math.round(overlayOpacity*100)}%`;$("dotRadius").value=dotRadius;$("dotRadiusValue").value=`${dotRadius} px`;$("captureArrows").checked=captureArrows;show(0)}
function fit(){let w=canvas.clientWidth,h=canvas.clientHeight; canvas.width=w*devicePixelRatio;canvas.height=h*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);scale=Math.min(w/image.width,h/image.height);ox=(w-image.width*scale)/2;oy=(h-image.height*scale)/2;draw()}
function pointStatus(point){return point.visibility==2?"Available":point.visibility==1?"Occluded":"Unavailable"}
function pointStatusClass(point){return point.visibility==2?"available":point.visibility==1?"occluded":"unavailable"}
function annotationState(value,record){if(value?.complete)return "complete";if(value&&(value.note.trim()||value.points.some(point=>point.visibility)))return "started";if(state.predictions?.items?.[record.id])return "predicted";return "untouched"}
function renderProgress(){let host=$("progress");host.innerHTML="";state.manifest.records.forEach((record,i)=>{let value=state.annotations.items[record.id],status=annotationState(value,record),button=document.createElement("button");button.className=`progressFrame ${status}${i==index?" current":""}`;button.title=`Frame ${i+1}: ${status}`;button.setAttribute("aria-label",button.title);button.onclick=()=>show(i);host.appendChild(button)});let current=host.children[index];if(current)current.scrollIntoView({block:"nearest",inline:"center"})}
function renderButtons(){let host=$("pointButtons");host.innerHTML="";item().points.forEach((point,i)=>{let b=document.createElement("button");b.className=`pointButton ${pointStatusClass(point)}${i==selected?" active":""}`;b.style.setProperty("--color",colors[i]);b.innerHTML=`<strong>${i+1}. ${labels[i]}</strong><small>${pointStatus(point)}</small>`;b.onclick=()=>selectPoint(i);host.appendChild(b)})}
function selectPoint(i){selected=i;setMode(item().points[selected].visibility);renderButtons();draw()}
function setMode(value){mode=value;$("visible").classList.toggle("active",mode==2);$("occluded").classList.toggle("active",mode==1);$("clear").classList.toggle("active",mode==0)}
function setSelectedStatus(value){setMode(value);let point=item().points[selected];if(point.x!==null&&point.y!==null){point.visibility=value;renderButtons();draw();save()}else{setStatus(value==2?"Click the image to place this available point":"Click the image to place this occluded point")}}
function setSelectedUnavailable(){setMode(0);item().points[selected]={x:null,y:null,visibility:0};renderButtons();draw();save()}
function timestamp(seconds){let milliseconds=Math.round((seconds%1)*1000),whole=Math.floor(seconds),s=whole%60,m=Math.floor(whole/60)%60,h=Math.floor(whole/3600);return `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}.${String(milliseconds).padStart(3,"0")}`}
function sourceUrl(url,seconds){if(!url)return "";let separator=url.includes("?")?"&":"?";return `${url}${separator}t=${Math.max(0,Math.round(seconds))}s`}
function renderPrediction(){let p=prediction(),panel=$("predictionPanel");panel.hidden=!p;if(p)$("predictionInfo").textContent=`${p.reason} · confidence ${p.confidence.toFixed(3)} · ${p.review}`}
function show(i){index=Math.max(0,Math.min(state.manifest.records.length-1,i));let r=state.manifest.records[index];image.onload=()=>{fit();};image.src="/"+r.image;let link=$("songLink");link.textContent=r.sourceTitle||r.sourceId||"Unknown song";if(r.sourceUrl)link.href=sourceUrl(r.sourceUrl,r.sourceSeconds);else link.removeAttribute("href");link.style.pointerEvents=r.sourceUrl?"auto":"none";$("counter").textContent=`· ${timestamp(r.sourceSeconds)} · Frame ${index+1} of ${state.manifest.records.length} · ${r.split} · ${r.width} × ${r.height}`;$("next").style.visibility=index==state.manifest.records.length-1?"hidden":"visible";renderPrediction();$("complete").checked=item().complete;$("note").value=item().note;renderButtons();renderProgress();setMode(item().points[selected].visibility);setStatus("All changes autosave")}
function draw(){ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);ctx.globalAlpha=1;ctx.drawImage(image,ox,oy,image.width*scale,image.height*scale);let p=item().points,predictionPoints=prediction()?.points;ctx.save();ctx.globalAlpha=overlayOpacity;ctx.lineWidth=2;
 if(predictionPoints){predictionPoints.forEach((x,i)=>{if(!x.visibility)return;let X=ox+x.x*image.width*scale,Y=oy+x.y*image.height*scale;ctx.strokeStyle="#666";ctx.setLineDash([4,3]);ctx.beginPath();ctx.arc(X,Y,dotRadius+3,0,Math.PI*2);ctx.stroke();ctx.setLineDash([])})}
 [[0,2,4],[1,3,5],[0,1],[2,3],[4,5]].forEach(line=>{let q=line.map(i=>p[i]);if(q.every(x=>x.visibility)){ctx.strokeStyle="#00e5ff";ctx.beginPath();q.forEach((x,j)=>{let X=ox+x.x*image.width*scale,Y=oy+x.y*image.height*scale;j?ctx.lineTo(X,Y):ctx.moveTo(X,Y)});ctx.stroke()}});
 p.forEach((x,i)=>{if(!x.visibility)return;let X=ox+x.x*image.width*scale,Y=oy+x.y*image.height*scale;ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(X,Y,dotRadius,0,Math.PI*2);ctx.fill();if(i==selected){ctx.strokeStyle="#000";ctx.lineWidth=2/devicePixelRatio;ctx.stroke()}ctx.fillStyle="white";ctx.fillText(i+1,X+dotRadius+5,Y-dotRadius-2)});ctx.restore();
 if(cursorInside){let left=Math.max(0,ox),right=Math.min(canvas.clientWidth,ox+image.width*scale),top=Math.max(0,oy),bottom=Math.min(canvas.clientHeight,oy+image.height*scale);if(cursorX>=left&&cursorX<=right&&cursorY>=top&&cursorY<=bottom){ctx.save();ctx.lineWidth=1/devicePixelRatio;ctx.strokeStyle="#FFF";ctx.beginPath();ctx.moveTo(left,cursorY);ctx.lineTo(right,cursorY);ctx.moveTo(cursorX,top);ctx.lineTo(cursorX,bottom);ctx.stroke();ctx.restore()}}}
function setStatus(x,error=false){$("saveState").textContent=x;$("saveState").style.color=error?"#ff7b72":"#7ee787"}
async function save(){let r=state.manifest.records[index], value=item();value.complete=$("complete").checked;value.note=$("note").value;setStatus("Saving…");let response=await fetch("/api/annotation/"+r.id,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(value)});let data=await response.json();if(!response.ok){$("complete").checked=false;value.complete=false;renderProgress();setStatus(data.error,true);return false}state.annotations.items[r.id]=data;let p=prediction();if(p)p.review=!data.complete?"pending":JSON.stringify(data.points)==JSON.stringify(p.points)?"accepted":"corrected";renderPrediction();renderButtons();renderProgress();setStatus("Saved");return true}
async function savePreferences(){state.preferences.overlayOpacity=overlayOpacity;state.preferences.dotRadius=dotRadius;state.preferences.captureArrowKeys=captureArrows;setStatus("Saving preferences…");let response=await fetch("/api/preferences",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(state.preferences)});let data=await response.json();if(!response.ok){setStatus(data.error,true);return}state.preferences=data;setStatus("Preferences saved")}
function placeAt(e){if(mode==0){setStatus("Choose Available (Q) or Occluded (W) before placing this point",true);return false}let x=(e.offsetX-ox)/(image.width*scale),y=(e.offsetY-oy)/(image.height*scale);if(x<0||x>1||y<0||y>1)return false;item().points[selected]={x,y,visibility:mode};renderButtons();draw();return true}
canvas.onmousedown=e=>{if(e.button==1||e.button==2){panning=true;last=[e.clientX,e.clientY];return}if(e.button==0){placing=placeAt(e)}};
canvas.onmousemove=e=>{cursorX=e.offsetX;cursorY=e.offsetY;cursorInside=true;if(panning){ox+=e.clientX-last[0];oy+=e.clientY-last[1];last=[e.clientX,e.clientY]}else if(placing){placeAt(e)}draw()};canvas.onmouseup=e=>{if(e.button==0&&placing){placing=false;save()}panning=false};canvas.onmouseleave=()=>{cursorInside=false;if(placing){placing=false;save()}panning=false;draw()};canvas.onmouseenter=e=>{cursorInside=true;cursorX=e.offsetX;cursorY=e.offsetY;draw()};canvas.oncontextmenu=e=>e.preventDefault();
canvas.onwheel=e=>{e.preventDefault();let factor=e.deltaY<0?1.15:1/1.15,x=e.offsetX,y=e.offsetY;ox=x-(x-ox)*factor;oy=y-(y-oy)*factor;scale*=factor;draw()};
$("prev").onclick=()=>show(index-1);$("next").onclick=()=>show(index+1);$("visible").onclick=()=>setSelectedStatus(2);$("occluded").onclick=()=>setSelectedStatus(1);$("clear").onclick=setSelectedUnavailable;$("complete").onchange=save;$("note").onchange=save;$("opacity").oninput=e=>{overlayOpacity=+e.target.value/100;$("opacityValue").value=`${e.target.value}%`;draw()};$("opacity").onchange=savePreferences;$("dotRadius").oninput=e=>{dotRadius=+e.target.value;$("dotRadiusValue").value=`${e.target.value} px`;draw()};$("dotRadius").onchange=savePreferences;$("captureArrows").onchange=e=>{captureArrows=e.target.checked;savePreferences()};
$("acceptPrediction").onclick=async()=>{$("complete").checked=true;if(await save()&&index<state.manifest.records.length-1)show(index+1)};
window.onresize=fit;window.onkeydown=async e=>{if((e.key=="ArrowLeft"||e.key=="ArrowRight")&&captureArrows){e.preventDefault();e.stopPropagation();if(document.activeElement)document.activeElement.blur();show(index+(e.key=="ArrowLeft"?-1:1));return}if(["INPUT","TEXTAREA"].includes(e.target.tagName))return;if(e.key=="Enter"){e.preventDefault();$("complete").checked=true;if(await save()&&index<state.manifest.records.length-1)show(index+1)}else if(e.key=="["){e.preventDefault();show(index-1)}else if(e.key=="]"){e.preventDefault();show(index+1)}else if(/[1-7]/.test(e.key))selectPoint(+e.key-1);else if(e.key.toLowerCase()=="q")setSelectedStatus(2);else if(e.key.toLowerCase()=="w")setSelectedStatus(1);else if(e.key.toLowerCase()=="e")setSelectedUnavailable()};load();
</script>"""


def serve(dataset, port=8765, *, open_browser=True):
    dataset = Path(dataset).resolve()
    manifest_path, annotations_path = dataset / "manifest.json", dataset / "annotations.json"
    preferences_path = dataset / "preferences.json"
    predictions_path = dataset / "predictions.json"
    if not manifest_path.is_file() or not annotations_path.is_file():
        raise ValueError("Annotation dataset requires manifest.json and annotations.json.")
    if not preferences_path.exists():
        _publish(preferences_path, {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "fretboard-annotation-preferences",
            "overlayOpacity": 1.,
            "captureArrowKeys": True,
            "dotRadius": 6.,
        })
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_ids = {record["id"] for record in manifest.get("records", ())}
    if predictions_path.is_file():
        load_predictions(predictions_path, record_ids=record_ids)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, value, status=HTTPStatus.OK):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                data = HTML.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path == "/api/state":
                try:
                    self._json({
                        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
                        "annotations": json.loads(annotations_path.read_text(encoding="utf-8")),
                        "preferences": normalize_preferences(json.loads(preferences_path.read_text(encoding="utf-8"))),
                        "predictions": load_predictions(predictions_path, record_ids=record_ids) if predictions_path.is_file() else None,
                    })
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            relative = PurePosixPath(self.path.lstrip("/"))
            path = dataset.joinpath(*relative.parts).resolve()
            if not path.is_relative_to(dataset) or not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("content-type", "image/jpeg" if path.suffix.lower() != ".png" else "image/png")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path == "/api/preferences":
                try:
                    length = int(self.headers.get("content-length", "0"))
                    if not 0 < length <= 10_000:
                        raise ValueError("Invalid preferences request size.")
                    value = normalize_preferences(json.loads(self.rfile.read(length)))
                    with lock:
                        _publish(preferences_path, value)
                    self._json(value)
                except (ValueError, json.JSONDecodeError) as error:
                    self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            prefix = "/api/annotation/"
            if not self.path.startswith(prefix):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            identifier = self.path[len(prefix):]
            try:
                length = int(self.headers.get("content-length", "0"))
                if not 0 < length <= 100_000:
                    raise ValueError("Invalid annotation request size.")
                value = normalize_annotation(json.loads(self.rfile.read(length)))
                with lock:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if identifier not in {record["id"] for record in manifest["records"]}:
                        raise ValueError("Unknown frame identifier.")
                    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
                    annotations["items"][identifier] = value
                    _publish(annotations_path, annotations)
                    if predictions_path.is_file():
                        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
                        prediction = predictions.get("items", {}).get(identifier)
                        if prediction is not None:
                            predictions["items"][identifier] = mark_prediction_review(
                                prediction, value,
                            )
                            _publish(predictions_path, predictions)
                self._json(value)
            except (ValueError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Fretboard annotation UI: {url}", flush=True)
    if open_browser:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--output", required=True)
    initialize.add_argument("--video", action="append", default=[])
    initialize.add_argument("--video-directory")
    initialize.add_argument("--frames-per-video", type=int, default=5)
    initialize.add_argument("--jpeg-quality", type=int, default=95)
    selection = commands.add_parser("select")
    selection.add_argument("--output", required=True)
    selection.add_argument("--video", action="append", default=[])
    selection.add_argument("--video-directory")
    selection.add_argument("--target-frames", type=int, default=1000)
    selection.add_argument("--candidates-per-video", type=int, default=24)
    selection.add_argument("--minimum-per-video", type=int, default=1)
    selection.add_argument("--maximum-per-video", type=int, default=12)
    selection.add_argument("--jpeg-quality", type=int, default=95)
    selection.add_argument("--shots-directory")
    selection.add_argument("--hands-directory")
    selection.add_argument("--negative-fraction", type=float, default=.05)
    selection.add_argument("--one-hand-fraction", type=float, default=.30)
    selection.add_argument("--cache-directory", default="data/fretboard-selection-cache")
    ui = commands.add_parser("serve")
    ui.add_argument("--dataset", required=True)
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-browser", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--dataset", required=True)
    prefill = commands.add_parser("prefill")
    prefill.add_argument("--dataset", required=True)
    prefill.add_argument("--model", required=True)
    prefill.add_argument("--device", default="cpu")
    prefill.add_argument("--image-size", type=int, default=1920)
    prefill.add_argument("--confidence", type=float, default=.01)
    refresh = commands.add_parser("refresh-metadata")
    refresh.add_argument("--dataset", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "init":
        manifest = initialize_dataset(
            args.output,
            _videos(args.video, args.video_directory),
            frames_per_video=args.frames_per_video,
            jpeg_quality=args.jpeg_quality,
        )
        print(json.dumps({"frames": len(manifest["records"]), "output": str(Path(args.output).resolve())}))
    elif args.command == "select":
        manifest = select_dataset(
            args.output, _videos(args.video, args.video_directory),
            target_frames=args.target_frames,
            candidates_per_video=args.candidates_per_video,
            minimum_per_video=args.minimum_per_video,
            maximum_per_video=args.maximum_per_video,
            jpeg_quality=args.jpeg_quality,
            shots_directory=args.shots_directory,
            hands_directory=args.hands_directory,
            negative_fraction=args.negative_fraction,
            one_hand_fraction=args.one_hand_fraction,
            cache_directory=args.cache_directory,
        )
        print(json.dumps({"frames": len(manifest["records"]), "output": str(Path(args.output).resolve())}))
    elif args.command == "serve":
        serve(args.dataset, args.port, open_browser=not args.no_browser)
    elif args.command == "export":
        print(json.dumps({"exported": export_yolo(args.dataset)}))
    elif args.command == "prefill":
        predictions = prefill_predictions(
            args.dataset, args.model, device=args.device,
            image_size=args.image_size, confidence=args.confidence,
        )
        print(json.dumps({"predictions": len(predictions["items"])}))
    else:
        manifest = refresh_dataset_metadata(args.dataset)
        print(json.dumps({"records": len(manifest["records"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
