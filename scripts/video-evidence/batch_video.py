"""Resumable, nontraining orchestration of the existing video-evidence stages."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import av
import cv2
import mediapipe
import numpy as np
import scipy

from audio_sync import align_soundtrack, installed_tools as _installed_tools
from core import EvidenceError, PRIVATE_OUTPUT_ROOT, REPOSITORY_ROOT, sha256, video_stream
from geometry import (AUTOMATIC_PREPARATION_METHOD, GEOMETRY_POINTS, STATE_CODES, geometry_row_complete,
                      annotations_ready, automatic_cut_intervals, cut_uncertainty_mask, validate_geometry_annotations)
from hand_motion import load_audio_clock, validate_clips
from hand_roles import (RoleConfig, _read_arrays, assign_hand_roles, load_role_inputs,
                        load_role_observations)
from hand_tracking import track_hands
from paired_inputs import (FEATURE_LAYOUT, INPUT_REPRESENTATION, SCHEMA_VERSION as PAIRED_SCHEMA_VERSION,
                           STRUCTURED_DIM, VIEW_ORDER, FRETBOARD_FEATURE_LAYOUT,
                           FRETBOARD_INPUT_REPRESENTATION, FRETBOARD_SCHEMA_VERSION,
                           FRETBOARD_STRUCTURED_DIM, prepare_paired_inputs, prepare_paired_inputs_v5)
from fretboard_tracking import FretboardDetectorConfig, track_fretboard
from shot_inspector import (_write_thumbnail, frame_timeline_sha256, inspect_shots,
                            prepare_automatic_shots, validate_frame_timeline)


STAGES = ("alignment", "shots", "hands", "annotations", "geometry", "roles", "fretboard", "bundle")
SHOT_STAGES = ("inspection", "automatic-inspection")
KINDS = {
    "alignment": "video-to-trimmed-audio-alignment",
    "shots": "video-shot-inspection", "hands": "fingerstyle-hand-observations",
    "annotations": "guitar-geometry-annotations", "geometry": "guitar-geometry-observations",
    "roles": "guitar-relative-hand-roles", "fretboard": "six-point-fretboard-observations",
    "bundle": "paired-video-inputs",
}
SOURCE_TIME_POLICY = (
    "Original video bytes, integer source PTS and time base are retained. "
    "Only frames inside the aligned already-trimmed audio and requested clips are packaged. "
    "Fresh detection uses the contiguous clip envelope; true shot cuts reset trackers. "
    "Final inputs contain independent hands and masks, never RGB crops; geometry and coarse slots remain unavailable. "
    "No frame/geometry interpolation, target-derived inputs, training or video rewriting."
)


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceError(f"Expected a JSON object: {path}")
    return value


def _path(value, *, file=False, private=False):
    if not isinstance(value, (str, Path)) or not str(value):
        raise EvidenceError("Paths must be nonempty absolute paths.")
    path = Path(value)
    if not path.is_absolute() or path.absolute() != path.resolve():
        raise EvidenceError(f"Path must be absolute and unaliased: {path}")
    if private and not path.is_relative_to(PRIVATE_OUTPUT_ROOT):
        raise EvidenceError(f"Artifact path must be under {PRIVATE_OUTPUT_ROOT}: {path}")
    for part in (path, *path.parents):
        if part.exists() and (part.is_symlink() or getattr(part.lstat(), "st_file_attributes", 0) & 0x400):
            raise EvidenceError(f"Aliased path is forbidden: {part}")
    if path.exists() and path.is_file() and path.stat().st_nlink != 1:
        raise EvidenceError(f"Hardlinked file is forbidden: {path}")
    if file and not path.is_file():
        raise EvidenceError(f"Required regular file is missing: {path}")
    return path


def _atomic(path, value, *, private=True):
    path = _path(path, private=private)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        with pending.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _runtime():
    tools = {}
    for path in _installed_tools():
        result = subprocess.run([str(path), "-version"], capture_output=True, text=True, check=True)
        tools[path.stem] = {"sha256": sha256(path), "version": result.stdout.splitlines()[0]}
    return {
        "python": sys.version,
        "packages": {"av": av.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
                     "mediapipe": mediapipe.__version__, "scipy": scipy.__version__},
        "tools": tools,
        "implementationSha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
                                 if not p.name.startswith("test_")},
    }


def _request(path):
    path = _path(path, file=True)
    document = _json(path)
    allowed = {"schemaVersion", "kind", "id", "video", "audio", "outputDirectory",
               "pluckingScreenSide", "clips", "reuse", "handModel", "poseModel",
               "fretboardModel", "fretboardDevice", "reviewMode"}
    if set(document) - allowed:
        raise EvidenceError(f"Unknown video request fields: {sorted(set(document) - allowed)}")
    if type(document.get("schemaVersion")) is not int or document["schemaVersion"] != 1 or document.get("kind") != "paired-video-preparation-request":
        raise EvidenceError("Unsupported paired-video-preparation-request schema.")
    if not isinstance(document.get("id"), str) or not document["id"].strip():
        raise EvidenceError("Request id must be a nonempty string.")
    result = dict(document)
    for name in ("video", "audio"):
        result[name] = str(_path(document.get(name), file=True))
    output = _path(document.get("outputDirectory"), private=True)
    if not output.is_relative_to(PRIVATE_OUTPUT_ROOT / "batches") or len(output.relative_to(PRIVATE_OUTPUT_ROOT / "batches").parts) < 2:
        raise EvidenceError("Worker output must be runs/video-evidence/batches/<batch>/<id>.")
    result["outputDirectory"] = str(output)
    result.setdefault("pluckingScreenSide", "geometry")
    result.setdefault("reviewMode", "automatic")
    if result["reviewMode"] not in ("automatic", "manual"):
        raise EvidenceError("reviewMode must be automatic or manual.")
    RoleConfig(plucking_screen_side=result["pluckingScreenSide"])
    clips = result.get("clips")
    if clips is not None:
        if not isinstance(clips, list) or not clips or any(
            not isinstance(c, list) or len(c) != 2 or any(type(v) is not int for v in c) or c[0] >= c[1]
            for c in clips
        ):
            raise EvidenceError("clips must contain increasing integer [startPts, endPtsExclusive] pairs.")
        ordered = sorted(clips)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise EvidenceError("Requested clips overlap.")
    result["clips"] = clips
    reuse = result.get("reuse", {})
    if not isinstance(reuse, dict) or set(reuse) - set(STAGES):
        raise EvidenceError("reuse contains unsupported video stages.")
    result["reuse"] = {name: str(_path(value, file=True, private=True)) for name, value in reuse.items()}
    result.setdefault("handModel", str(PRIVATE_OUTPUT_ROOT / "models" / "hand_landmarker.task"))
    result.setdefault("poseModel", None)
    result.setdefault("fretboardModel", None)
    result.setdefault("fretboardDevice", "cpu")
    if not isinstance(result["fretboardDevice"], str) or not result["fretboardDevice"].strip():
        raise EvidenceError("fretboardDevice must be a nonempty Ultralytics device string.")
    for name in ("handModel", "poseModel", "fretboardModel"):
        if result[name] is not None:
            result[name] = str(_path(result[name]))
    inputs = [Path(result[name]) for name in ("video", "audio")]
    other = [Path(p) for p in result["reuse"].values()]
    other.extend(Path(result[n]) for n in ("handModel", "poseModel", "fretboardModel") if result[n])
    if len(set(inputs + other)) != len(inputs + other) or any(p.is_relative_to(output) for p in inputs + other + [path]):
        raise EvidenceError("Input paths must be distinct and outside the worker output directory.")
    return path, result


def _artifacts(path, stage):
    if stage in SHOT_STAGES:
        stage = "shots"
    path = _path(path, file=True, private=True)
    values = {str(path): sha256(path)}
    if stage in ("hands", "geometry", "roles", "fretboard", "bundle"):
        document = _json(path)
        expected = "inputs.npz" if stage == "bundle" else "fretboard.npz" if stage == "fretboard" else f"{stage}.npz"
        key = "arraysPath" if stage == "bundle" else "arrays"
        if document.get(key) != expected:
            raise EvidenceError(f"{stage} must reference canonical local {expected}.")
        arrays = _path(path.with_name(expected), file=True, private=True)
        digest = sha256(arrays)
        if document.get("arraysSha256") is not None and document["arraysSha256"] != digest:
            raise EvidenceError(f"{stage} arrays differ from the report hash.")
        if stage in ("roles", "bundle") and document.get("arraysSha256") != digest:
            raise EvidenceError(f"{stage} requires its array hash.")
        values[str(arrays)] = digest
    if stage == "annotations":
        document = _json(path)
        values.update({str(p): sha256(p) for p in _annotation_image_paths(path, document)})
    if stage == "shots":
        rows = _json(path).get("shots")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise EvidenceError("Shot review rows must be objects.")
        for row in rows:
            if row.get("reviewFrame"):
                if not isinstance(row["reviewFrame"], str):
                    raise EvidenceError("Shot reviewFrame must be a relative path string.")
                relative = Path(row["reviewFrame"])
                if relative.is_absolute():
                    raise EvidenceError("Shot thumbnails must use relative private paths.")
                image = _path(Path(os.path.abspath(path.parent / relative)), file=True, private=True)
                digest = sha256(image)
                if row.get("reviewFrameSha256") is not None and row["reviewFrameSha256"] != digest:
                    raise EvidenceError("Shot thumbnail differs from its source-bound hash.")
                values[str(image)] = digest
    return values


def _verify_hashes(values, description):
    for name, digest in values.items():
        if sha256(_path(name, file=True)) != digest:
            raise EvidenceError(f"Stale {description}: {name} changed. Recompute in a NEW output directory.")


def _validate_observation_archive(stage, path, count):
    shapes = {"pts": (np.dtype("int64"), (count,)), "shot_id": (np.dtype("int32"), (count,))}
    if stage == "hands":
        shapes.update({
            "hand_count": (np.dtype("int8"), (count,)),
            "image_landmarks": (np.dtype("float32"), (count, 2, 21, 3)),
            "handedness": (np.dtype("int8"), (count, 2)),
            "handedness_score": (np.dtype("float32"), (count, 2)),
            "detection_source": (np.dtype("int8"), (count,)),
        })
    else:
        shapes.update({
            "coordinates": (np.dtype("float32"), (count, 6, 2)),
            "confidence": (np.dtype("float32"), (count, 6)),
            "source": (np.dtype("int8"), (count, 6)), "state": (np.dtype("int8"), (count,)),
        })
    arrays = _read_arrays(Path(path).with_name(f"{stage}.npz"), shapes)
    scores = arrays["handedness_score"] if stage == "hands" else arrays["confidence"]
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise EvidenceError(f"{stage} observation confidence must be finite and within [0, 1].")
    if stage == "hands":
        observed = np.isfinite(arrays["image_landmarks"][..., :2]).all(-1).any(-1)
        if not np.array_equal(observed.sum(-1), arrays["hand_count"]):
            raise EvidenceError("Hand counts disagree with observed landmarks.")
        if not np.isin(arrays["handedness"], [-1, 0, 1]).all() or not np.isin(arrays["detection_source"], [0, 1, 2]).all():
            raise EvidenceError("Unknown hand observation encoding.")
    elif (not np.isin(arrays["state"], list(STATE_CODES.values())).all()
          or not np.isin(arrays["source"], [0, 1, 2]).all()
          or not np.isfinite(arrays["coordinates"][scores > 0]).all()):
        raise EvidenceError("Geometry state/source/coordinate evidence is invalid.")
    return arrays


def _validate_report(stage, path, request, sources, paths):
    document = _json(_path(path, file=True, private=True))
    expected_bundle_version = FRETBOARD_SCHEMA_VERSION if request.get("fretboardModel") else PAIRED_SCHEMA_VERSION
    if stage == "bundle" and (document.get("schemaVersion") != expected_bundle_version or "imageSize" in document):
        raise EvidenceError("Older numeric/RGB bundles cannot be reused; rebuild the current geometry-and-hand inputs from cached observations in a NEW output directory.")
    version = expected_bundle_version if stage == "bundle" else 1
    if document.get("kind") != KINDS[stage] or type(document.get("schemaVersion")) is not int or document["schemaVersion"] != version:
        raise EvidenceError(f"Invalid {stage} report schema.")
    _artifacts(path, stage)
    if stage in ("alignment", "shots", "hands", "annotations", "geometry", "fretboard"):
        if document.get("videoSha256") != sources["video"]:
            raise EvidenceError(f"{stage} report does not match video hash.")
    if stage == "alignment":
        if (document.get("trimmedAudioSha256") != sources["audio"] or document.get("rate") != [1, 1]
                or any(type(value) is not int for value in document["rate"])):
            raise EvidenceError("Alignment must match trimmed audio and use the unit-rate clock.")
        offset = document.get("videoStartSecondsForTrimmedAudioZero")
        if type(offset) not in (int, float) or not math.isfinite(offset):
            raise EvidenceError("Alignment offset must be finite.")
        if document.get("status") not in ("ambiguous", "supported"):
            raise EvidenceError("Unsupported alignment status.")
        if document["status"] == "supported":
            load_audio_clock(sources["video"], request["audio"], path)
    if stage == "shots":
        validate_frame_timeline(document)
        rows = document.get("shots")
        if not isinstance(rows, list) or not rows or document.get("shotCount") != len(rows):
            raise EvidenceError("Shot report requires every ordered shot.")
        tb = document.get("timeBase")
        if not isinstance(tb, list) or len(tb) != 2 or any(type(v) is not int or v <= 0 for v in tb):
            raise EvidenceError("Shot time base must be an explicit positive rational.")
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("shotId") != i or any(type(row.get(k)) is not int for k in ("startPts", "endPtsExclusive")) or row["startPts"] >= row["endPtsExclusive"]:
                raise EvidenceError("Shot report has invalid PTS intervals.")
            if i and rows[i - 1]["endPtsExclusive"] != row["startPts"]:
                raise EvidenceError("Shot intervals must be contiguous.")
        if type(document.get("frameCount")) is not int or document["frameCount"] < 1:
            raise EvidenceError("Shot frameCount must be positive.")
        if (any(type(document.get(key)) is not int for key in ("firstPts", "lastPts", "width", "height"))
                or document["firstPts"] != rows[0]["startPts"] or document["lastPts"] < document["firstPts"]
                or document["lastPts"] >= rows[-1]["endPtsExclusive"] or min(document["width"], document["height"]) <= 0):
            raise EvidenceError("Shot endpoints and source dimensions are malformed.")
    if stage in ("hands", "annotations", "geometry", "fretboard"):
        if "shots" not in paths or document.get("shotsSha256") != sha256(paths["shots"]):
            raise EvidenceError(f"{stage} requires its exact source shot report.")
        shots = _json(paths["shots"])
        if stage == "annotations":
            validate_geometry_annotations(document, shots, sources["video"], sha256(paths["shots"]))
            if annotations_ready(document) and not all(geometry_row_complete(row) for row in document["shots"]):
                raise EvidenceError("Completed annotations contain incomplete shot geometry.")
            if any(row["geometry"] is not None or row.get("additionalKeyframes") for row in document["shots"]):
                raise EvidenceError("Geometry preparation is disabled; reused annotations must contain only unavailable coordinates.")
        elif stage in ("hands", "geometry"):
            if any(document.get(key) != shots.get(key) for key in ("timeBase", "frameCount", "shotCount")):
                raise EvidenceError(f"{stage} coverage differs from shots.")
            arrays = _validate_observation_archive(stage, path, shots["frameCount"])
            pts, ids = arrays["pts"], arrays["shot_id"]
            if np.any(np.diff(pts) <= 0):
                raise EvidenceError(f"{stage} PTS are malformed.")
            if pts[0] != shots["firstPts"] or pts[-1] != shots["lastPts"]:
                raise EvidenceError(f"{stage} PTS endpoints differ from shots.")
            for row in shots["shots"]:
                mask = (pts >= row["startPts"]) & (pts < row["endPtsExclusive"])
                if not mask.any() or not np.array_equal(mask, ids == row["shotId"]):
                    raise EvidenceError(f"{stage} PTS/shot IDs disagree.")
            if stage == "geometry":
                if "annotations" not in paths or document.get("annotationsSha256") != sha256(paths["annotations"]):
                    raise EvidenceError("Geometry requires its exact source annotations.")
                if document.get("pointOrder") != list(GEOMETRY_POINTS) or document.get("stateEncoding") != STATE_CODES:
                    raise EvidenceError("Geometry point/state encoding is unsupported.")
                if arrays["confidence"].any() or arrays["source"].any() or np.isfinite(arrays["coordinates"]).any():
                    raise EvidenceError("Geometry preparation is disabled; reused coordinates must remain unavailable.")
        else:
            if document.get("shotsSha256") != sha256(paths["shots"]) or document.get("modelSha256") != sha256(request["fretboardModel"]):
                raise EvidenceError("Fretboard observations require the exact shots and detector model.")
    if stage in ("roles", "bundle"):
        required = {"shots", "hands", "geometry", "annotations"}
        if not required <= paths.keys():
            raise EvidenceError(f"{stage} reuse requires its complete dependency reports.")
        inputs = load_role_inputs(request["video"], paths["shots"], paths["hands"], paths["geometry"], paths["annotations"])
        if stage == "roles":
            load_role_observations(inputs, path)
            if document.get("config", {}).get("plucking_screen_side") != request["pluckingScreenSide"]:
                raise EvidenceError("Reused role orientation differs from requested pluckingScreenSide.")
        else:
            if not {"roles", "alignment"} <= paths.keys():
                raise EvidenceError("Bundle reuse requires alignment and roles.")
            roles, _ = load_role_observations(inputs, paths["roles"])
            _, _, _, audio_hashes = load_audio_clock(sources["video"], request["audio"], paths["alignment"])
            expected = {**inputs["hashes"], **audio_hashes, "roles": sha256(paths["roles"]),
                        "roleArrays": roles["arraysSha256"]}
            if request.get("fretboardModel"):
                if "fretboard" not in paths:
                    raise EvidenceError("Schema-5 bundle reuse requires fretboard observations.")
                fretboard_report = _json(paths["fretboard"])
                expected.update(
                    fretboard=sha256(paths["fretboard"]),
                    fretboardArrays=fretboard_report["arraysSha256"],
                )
            if document.get("inputSha256") != expected or document.get("id") != request["id"]:
                raise EvidenceError("Reused bundle differs from source dependencies or id.")
            representation, dimension, layout = (
                (FRETBOARD_INPUT_REPRESENTATION, FRETBOARD_STRUCTURED_DIM, FRETBOARD_FEATURE_LAYOUT)
                if request.get("fretboardModel")
                else (INPUT_REPRESENTATION, STRUCTURED_DIM, FEATURE_LAYOUT)
            )
            if (document.get("inputRepresentation") != representation
                    or document.get("featureDimension") != dimension
                    or document.get("featureLayout") != layout or document.get("viewOrder") != VIEW_ORDER):
                raise EvidenceError("Bundle must contain the supported numeric guitar-and-hand representation.")
            with np.load(Path(path).with_name("inputs.npz"), allow_pickle=False) as arrays:
                if set(arrays.files) != {"structured", "structured_available", "pts", "audio_seconds", "technique_available", "segment_id"}:
                    raise EvidenceError("Bundle must contain only numeric structure, masks and timestamps; RGB/image arrays are forbidden.")
                for start, stop in ((0, 98), (186, 194)):
                    if arrays["structured_available"][..., start:stop].any() or arrays["structured"][..., start:stop].any():
                        raise EvidenceError("Geometry and coarse input slots must remain zero and unavailable.")
                if request.get("fretboardModel"):
                    if document.get("schemaVersion") != 5 or arrays["structured"].shape[-1] != 233:
                        raise EvidenceError("Fretboard preparation requires schema-5 D233 paired inputs.")
                elif document.get("schemaVersion") != PAIRED_SCHEMA_VERSION:
                    raise EvidenceError("Legacy hand-only preparation requires schema-4 paired inputs.")
            for name, value in document.get("inputPaths", {}).items():
                if name not in expected or sha256(_path(value, file=True)) != expected[name]:
                    raise EvidenceError("Bundle inputPaths do not match its bound sources.")
            if set(document.get("inputPaths", {})) != set(expected):
                raise EvidenceError("Bundle inputPaths are incomplete.")
            if request["clips"] is not None and document.get("clips") != [
                {"startPts": start, "endPtsExclusive": end} for start, end in sorted(request["clips"])
            ]:
                raise EvidenceError("Reused bundle clip scope differs from requested clips.")
    return document


def _preflight(request, sources):
    paths = {}
    for stage in STAGES:
        if stage in request["reuse"]:
            paths[stage] = request["reuse"][stage]
            _validate_report(stage, paths[stage], request, sources, paths)
    if "geometry" in paths and not annotations_ready(_json(paths["annotations"])):
        raise EvidenceError("Reused geometry needs completed, source-bound annotations.")
    if {"hands", "geometry", "annotations", "shots"} <= paths.keys():
        load_role_inputs(request["video"], paths["shots"], paths["hands"], paths["geometry"], paths["annotations"])


@contextmanager
def _lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".worker.lock"
    try:
        stream = path.open("x", encoding="ascii")
    except FileExistsError as error:
        raise EvidenceError(f"Worker locked: {path}. Check its PID and remove ONLY this lock after confirming that worker stopped.") from error
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


class _Worker:
    def __init__(self, request_path, request, *, reset_failed=False):
        self.request_path, self.request = request_path, request
        self.output = Path(request["outputDirectory"])
        self.progress("inputs: verifying source hashes, reusable artifacts and job identity...")
        self.sources = {name: sha256(request[name]) for name in ("video", "audio")}
        _preflight(request, self.sources)
        files = {str(request_path): sha256(request_path)}
        files.update({request[n]: self.sources[n] for n in self.sources})
        for stage, path in request["reuse"].items():
            files.update(_artifacts(path, stage))
        for name in ("handModel", "poseModel", "fretboardModel"):
            if request[name] and Path(request[name]).is_file():
                files[request[name]] = sha256(request[name])
        identity = {"request": request, "files": files, "runtime": _runtime()}
        self.identity = _digest(identity)
        self.initial_files = files
        self.ledger_path = self.output / "status.json"
        if self.ledger_path.exists():
            self.ledger = _json(self.ledger_path)
            if self.ledger.get("identitySha256") != self.identity:
                raise EvidenceError("Stale input/configuration/implementation/runtime identity. Recompute in a NEW output directory.")
        else:
            if any(p.name != ".worker.lock" for p in self.output.iterdir()):
                raise EvidenceError("Output exists without a worker ledger; use a NEW output directory.")
            self.ledger = {"schemaVersion": 1, "kind": "paired-video-stage-ledger",
                           "identitySha256": self.identity, "identity": identity, "stages": {}}
            self.save()
        self.reset_failed = reset_failed
        self.paths, self.summary = {}, []
        self.receipt_path = self.output / "workerreviewreceipt.json"
        self.receipt = _json(self.receipt_path) if self.receipt_path.exists() else {}
        if self.receipt and self.receipt.get("identitySha256") != self.identity:
            raise EvidenceError("Review receipt belongs to a different source/configuration.")
        expected_receipt = self.ledger.get("reviewReceiptSha256")
        if expected_receipt is not None and (not self.receipt_path.is_file() or sha256(self.receipt_path) != expected_receipt):
            raise EvidenceError("Stale review receipt; use a NEW output directory.")
        if self.receipt and expected_receipt is None:
            raise EvidenceError("Unrecorded review receipt; use a NEW output directory.")
        self.progress("inputs: verified.")

    def progress(self, message):
        print(f"{self.request['id']}: {message}", flush=True)

    def save(self):
        _atomic(self.ledger_path, self.ledger)

    def check_sources(self):
        _verify_hashes(self.initial_files, "source/configuration")

    def dependencies(self, names):
        values = {}
        for name in names:
            path = self.paths[name]
            values.update(_artifacts(path, name) if name in (*STAGES, *SHOT_STAGES) else {str(path): sha256(path)})
        return values

    def stage(self, name, operation, *, dependencies=(), reuse=None):
        self.progress(f"{name}: starting; verifying dependencies...")
        dependency_hashes = self.dependencies(dependencies)
        previous = self.ledger["stages"].get(name)
        directory = self.output / name
        if previous and previous["status"] == "complete":
            if previous["dependencies"] != dependency_hashes:
                raise EvidenceError(f"Stale {name} dependencies. Recompute in a NEW output directory.")
            _verify_hashes(previous["artifacts"], f"{name} output")
            path = Path(previous["path"])
            self.paths[name] = path
            self.summary.append({"stage": name, "status": "reused", "provenance": previous["provenance"], "path": str(path)})
            self.progress(f"{name}: reused verified output.")
            return path
        if previous:
            if not self.reset_failed:
                raise EvidenceError(f"Incomplete stage {name}; resume with run --request \"{self.request_path}\" --output <result.json> --reset-failed.")
            if previous.get("directory") != str(directory) or previous.get("provenance") != "processed":
                raise EvidenceError(f"Cannot reset an unowned {name} stage; use a NEW output directory.")
            if directory.exists():
                _path(directory, private=True)
                # Preserve failed evidence; never recursively delete possibly user-created files.
                directory.rename(self.output / f"{name}.failed-{uuid4().hex}")
        elif directory.exists():
            raise EvidenceError(f"Unrecorded partial stage {directory}; use a NEW output directory.")
        entry = {"status": "running", "dependencies": dependency_hashes, "directory": str(directory),
                 "provenance": "imported" if reuse else "processed"}
        self.ledger["stages"][name] = entry
        self.save()
        try:
            self.progress(f"{name}: {'importing cached output' if reuse else 'processing'}...")
            produced = reuse if reuse else operation(directory)
            if isinstance(produced, tuple):
                produced, provenance = produced
                entry["provenance"] = provenance
            path = Path(produced)
            self.paths[name] = path
            if name in (*STAGES, *SHOT_STAGES):
                _validate_report("shots" if name in SHOT_STAGES else name, path, self.request, self.sources, self.paths)
            hashes = _artifacts(path, name) if name in (*STAGES, *SHOT_STAGES) else {str(path): sha256(path)}
            self.check_sources()
            _verify_hashes(dependency_hashes, f"{name} dependency")
            entry.update(status="complete", path=str(path), artifacts=hashes)
            self.save()
        except Exception as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            self.save()
            self.progress(f"{name}: failed: {error}")
            raise
        self.summary.append({"stage": name, "status": "reused" if reuse or entry["provenance"] == "cached" else "completed",
                             "provenance": entry["provenance"], "path": str(path)})
        self.progress(f"{name}: completed ({entry['provenance']}).")
        return path

    def action(self, stage, reason, command, path):
        self.check_sources()
        self.summary.append({"stage": stage, "status": "needs-review", "path": str(path)})
        return self.result("needs-review", [{"stage": stage, "reason": reason, "command": command, "path": str(path)}])

    def command(self, command, *arguments):
        return [sys.executable, str(Path(__file__).resolve()), command, "--request", str(self.request_path), *arguments]

    def result(self, status, actions=()):
        self.check_sources()
        for name, entry in self.ledger["stages"].items():
            if entry["status"] == "complete":
                _verify_hashes(entry["artifacts"], f"{name} output")
        return {"schemaVersion": 1, "kind": "paired-video-preparation-result", "id": self.request["id"],
                "status": status, "inputSha256": self.sources,
                "artifacts": {name: str(path) for name, path in self.paths.items() if name in STAGES},
                "actions": list(actions), "stageSummary": self.summary, "sourceTimePolicy": SOURCE_TIME_POLICY}


def _annotation_image_paths(path, document):
    paths = []
    rows = document.get("shots")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise EvidenceError("Geometry review rows must be objects.")
    for row in rows:
        additional = row.get("additionalKeyframes", [])
        if not isinstance(additional, list) or any(not isinstance(seed, dict) for seed in additional):
            raise EvidenceError("Additional geometry keyframes must be objects.")
        for seed in (row, *additional):
            if not isinstance(seed.get("keyframeImage"), str):
                raise EvidenceError("Geometry keyframeImage must be a relative path string.")
            relative = Path(seed["keyframeImage"])
            if relative.is_absolute() or ".." in relative.parts:
                raise EvidenceError("Geometry review keyframes must be local relative image paths.")
            absolute = Path(os.path.abspath(Path(path).parent / relative))
            paths.append(_path(absolute, file=True, private=True))
    return paths


def _timeline(video, shots_path, directory, hands_path=None):
    shots = _json(shots_path)
    cached_timeline = validate_frame_timeline(shots)
    if cached_timeline is not None:
        pts, end = cached_timeline
        directory.mkdir()
        path = directory / "timeline.json"
        _atomic(path, {"schemaVersion": 1, "kind": "source-video-timeline",
                       "videoSha256": shots["videoSha256"], "timeBase": shots["timeBase"],
                       "pts": pts, "endPtsExclusive": end, "timestampSource": "validated-shot-inspection"})
        return path
    rows = []
    if hands_path is not None:
        with np.load(Path(hands_path).with_name("hands.npz"), allow_pickle=False) as arrays:
            pts = arrays["pts"].tolist()
        end = shots["shots"][-1]["endPtsExclusive"]
        if end != shots["lastPts"] + 1:
            directory.mkdir()
            path = directory / "timeline.json"
            _atomic(path, {"schemaVersion": 1, "kind": "source-video-timeline",
                           "videoSha256": sha256(video), "timeBase": shots["timeBase"],
                           "pts": pts, "endPtsExclusive": end,
                           "timestampSource": "validated-cached-hand-observations"})
            return path
    with av.open(str(video)) as container:
        stream = video_stream(container)
        time_base = Fraction(stream.time_base)
        if [time_base.numerator, time_base.denominator] != shots["timeBase"]:
            raise EvidenceError("Source and shot time bases differ.")
        if hands_path is not None:
            container.seek(shots["lastPts"], stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or rows and frame.pts <= rows[-1][0]:
                raise EvidenceError("Source PTS must be present and strictly increasing.")
            rows.append([int(frame.pts), int(frame.duration or 0)])
        if not rows:
            raise EvidenceError("Source contains no frames.")
        end = rows[-1][0] + rows[-1][1] if rows[-1][1] > 0 else None
        if end is None and stream.duration is not None:
            end = int(stream.start_time or 0) + stream.duration
        if end is None or end <= rows[-1][0]:
            raise EvidenceError("Source has no trustworthy final frame duration; supply explicit clips ending at a decoded PTS.")
    directory.mkdir()
    path = directory / "timeline.json"
    _atomic(path, {"schemaVersion": 1, "kind": "source-video-timeline", "videoSha256": sha256(video),
                   "timeBase": shots["timeBase"], "pts": pts if hands_path is not None else [row[0] for row in rows],
                   "endPtsExclusive": end})
    return path


def _clips(request, sources, alignment, shots_path, timeline_path):
    clock, duration, _, _ = load_audio_clock(sources["video"], request["audio"], alignment)
    shots, timeline = _json(shots_path), _json(timeline_path)
    pts = np.asarray(timeline["pts"], np.int64)
    tb = Fraction(*shots["timeBase"])
    samples = np.asarray([clock.map_pts(int(p), tb) for p in pts])
    length = duration * clock.sample_rate
    valid = (samples >= 0) & (samples < length)
    valid &= (pts >= shots["firstPts"]) & (pts <= shots["lastPts"])
    if not valid.any():
        raise EvidenceError("No source frames lie inside the aligned trimmed audio.")
    selected = pts[valid]
    if request["clips"] is not None:
        clips = [{"label": f"clip-{i + 1}", "startPts": c[0], "endPtsExclusive": c[1]}
                 for i, c in enumerate(sorted(request["clips"]))]
    else:
        last = int(np.flatnonzero(valid)[-1])
        next_pts = int(pts[last + 1]) if last + 1 < len(pts) else timeline["endPtsExclusive"]
        bound = math.floor((length - clock.offset_samples) / (tb * clock.sample_rate))
        end = min(next_pts, bound)
        # A sub-tick audio tail cannot form a nonempty integer-PTS interval for
        # the final frame. Keep the preceding native frames, never extend audio.
        selected = selected[selected < end]
        if len(selected) < 3:
            raise EvidenceError("Aligned audio has fewer than three source frames inside representable PTS bounds.")
        clips = [{"label": "aligned-audio", "startPts": int(selected[0]), "endPtsExclusive": end}]
    bounds = set(int(p) for p in pts) | {timeline["endPtsExclusive"], clips[-1]["endPtsExclusive"] if request["clips"] is None else shots["shots"][-1]["endPtsExclusive"]}
    for clip in clips:
        start, end = clip["startPts"], clip["endPtsExclusive"]
        mask = (pts >= start) & (pts < end)
        if start not in set(pts.tolist()) or end not in bounds or mask.sum() < 3 or not valid[mask].all():
            raise EvidenceError("Each requested clip must contain at least three aligned source frames with exact source PTS bounds.")
        if clock.map_pts(start, tb) < 0 or clock.map_pts(end, tb) > length:
            raise EvidenceError("Requested clip is outside aligned trimmed audio.")
    return clips


def _select_interval(video, original_path, timeline_path, clips, directory):
    original, timeline = _json(original_path), _json(timeline_path)
    start = min(row["startPts"] for row in clips)
    end = max(row["endPtsExclusive"] for row in clips)
    tb = Fraction(*original["timeBase"])
    rows = []
    for old in original["shots"]:
        left, right = max(start, old["startPts"]), min(end, old["endPtsExclusive"])
        # The inspector's historical lastPts+1 is a coverage sentinel, not a duration.
        if old is original["shots"][-1] and old["endPtsExclusive"] == original["lastPts"] + 1:
            right = min(end, timeline["endPtsExclusive"])
        if left >= right:
            continue
        row = {**old, "shotId": len(rows), "sourceShotId": old.get("sourceShotId", old["shotId"]),
               "startPts": left, "endPtsExclusive": right,
               "startSeconds": float(left * tb), "endSecondsExclusive": float(right * tb)}
        if left != old["startPts"]:
            row.update(boundaryType="range_start", boundaryProvenance="aligned_audio_selection", cutScore=None)
        rows.append(row)
    pts = [p for p in timeline["pts"] if start <= p < end]
    if not rows or not pts or rows[0]["startPts"] != pts[0] or rows[-1]["endPtsExclusive"] != end:
        raise EvidenceError("Selected interval is not covered by the inspected shots.")
    directory.mkdir()
    boundaries = {r["startPts"]: r for r in rows}
    with av.open(str(video)) as container:
        stream = video_stream(container)
        container.seek(start, stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts >= end:
                break
            if frame.pts in boundaries:
                row = boundaries[frame.pts]
                row["reviewFrame"] = _write_thumbnail(directory, row["shotId"], frame.pts, frame.to_ndarray(format="rgb24"))
                row["reviewFrameSha256"] = sha256(directory / row["reviewFrame"])
    report = {**original, "shots": rows, "shotCount": len(rows), "frameCount": len(pts),
              "firstPts": pts[0], "lastPts": pts[-1], "sourceInspectionSha256": sha256(original_path),
              "sourceInspectionPath": str(original_path), "requestedClips": request_clips(clips),
              "coveragePolicy": SOURCE_TIME_POLICY}
    if "framePts" in original:
        report.update(framePts=pts, frameEndPtsExclusive=end)
        report["frameTimelineSha256"] = frame_timeline_sha256(report)
    if "automaticReview" in original:
        review = original["automaticReview"]
        report["automaticReview"] = {**review, "uncertainIntervals": [
            {**interval, "startPts": max(start, interval["startPts"]), "endPtsExclusive": min(end, interval["endPtsExclusive"])}
            for interval in review.get("uncertainIntervals", [])
            if interval["startPts"] < end and interval["endPtsExclusive"] > start
        ]}
    for field in ("lowContrastCutCandidates", "dissolveCandidates"):
        report[field] = [row for row in original.get(field, []) if start <= row["pts"] < end]
    path = directory / "shots.json"
    _atomic(path, report)
    return path


def request_clips(clips):
    return [{"startPts": row["startPts"], "endPtsExclusive": row["endPtsExclusive"]} for row in clips]


def _unavailable_annotations(shots_path, directory):
    shots = _json(shots_path)
    directory.mkdir()
    rows = []
    for shot in shots["shots"]:
        image = _path(Path(shots_path).parent / shot["reviewFrame"], file=True, private=True)
        relative = f"shot-{shot['shotId']:04d}.jpg"
        shutil.copyfile(image, directory / relative)
        rows.append({
            "shotId": shot["shotId"], "startPts": shot["startPts"], "endPtsExclusive": shot["endPtsExclusive"],
            "keyframePts": shot["startPts"], "keyframeImage": relative,
            "state": "calibration_unstable", "geometry": None,
            "reviewNote": "Geometry disabled; independent hand evidence only.",
        })
    report = {
        "schemaVersion": 1, "kind": "guitar-geometry-annotations", "visibility": "private",
        "videoSha256": shots["videoSha256"], "shotsSha256": sha256(shots_path),
        "pointOrder": list(GEOMETRY_POINTS), "coordinateSpace": "normalized_full_frame", "shots": rows,
        "reviewComplete": False, "preparationComplete": True, "preparationMethod": AUTOMATIC_PREPARATION_METHOD,
        "geometryPolicy": "disabled; reserved coordinates remain unavailable",
        "proposalSummary": {"autoAcceptedShotIds": [], "unavailableShotIds": [row["shotId"] for row in rows], "reviewRequiredShotIds": []},
        "trainingPerformed": False,
    }
    path = directory / "annotations.json"
    _atomic(path, report)
    return path


def _unavailable_geometry(shots_path, hands_path, annotations_path, directory):
    shots = _json(shots_path)
    with np.load(Path(hands_path).with_name("hands.npz"), allow_pickle=False) as hand:
        pts, ids = hand["pts"].copy(), hand["shot_id"].copy()
    count = len(pts)
    states = np.full(count, STATE_CODES["calibration_unstable"], np.int8)
    states[cut_uncertainty_mask(pts, automatic_cut_intervals(shots))] = STATE_CODES["transition"]
    directory.mkdir()
    arrays = directory / "geometry.npz"
    np.savez_compressed(arrays, pts=pts, shot_id=ids, coordinates=np.full((count, 6, 2), np.nan, np.float32),
                        confidence=np.zeros((count, 6), np.float32), source=np.zeros((count, 6), np.int8), state=states)
    report = {
        "schemaVersion": 1, "kind": "guitar-geometry-observations", "visibility": "private",
        "videoSha256": shots["videoSha256"], "shotsSha256": sha256(shots_path), "annotationsSha256": sha256(annotations_path),
        "timeBase": shots["timeBase"], "frameCount": count, "shotCount": shots["shotCount"],
        "pointOrder": list(GEOMETRY_POINTS), "stateEncoding": STATE_CODES, "arrays": "geometry.npz", "arraysSha256": sha256(arrays),
        "framesWithCoordinateFrame": 0, "coordinateFrameCoverage": 0., "trainingPerformed": False,
        "geometryPolicy": "disabled; unavailable coordinates from native hand timestamps, no tracking or interpolation",
    }
    path = directory / "geometry.json"
    _atomic(path, report)
    return path


def _align(video, audio, directory):
    path, document = align_soundtrack(video, audio, directory / "alignment.json")
    # FFT correlation can exceed its mathematical bound by machine roundoff.
    score = document["correlation"]
    if 1 < abs(score) <= 1 + 1e-12:
        document["correlationBeforeRoundoffClamp"] = score
        document["correlation"] = math.copysign(1., score)
        _atomic(path, document)
    return path


def _cached(worker, stage, operation):
    names = ("video", "audio") if stage == "alignment" else ("video",)
    identity = {"stage": stage, "sources": {n: worker.sources[n] for n in names},
                "configuration": "existing-api-defaults",
                "runtime": worker.ledger["identity"]["runtime"]}
    batches = PRIVATE_OUTPUT_ROOT / "batches"
    batch_root = batches / worker.output.relative_to(batches).parts[0]
    directory = batch_root / ".stage-cache" / stage / _digest(identity)
    with _lock(directory):
        manifest_path, artifact = directory / "cache.json", directory / "artifact"
        if manifest_path.exists():
            entry = _json(manifest_path)
            if entry.get("identity") != identity:
                raise EvidenceError(f"Cache identity mismatch: {manifest_path}")
            if entry.get("status") == "complete":
                _verify_hashes(entry["artifacts"], f"{stage} cache")
                return Path(entry["path"]), "cached"
            if not worker.reset_failed:
                raise EvidenceError(f"Incomplete {stage} cache; use --reset-failed to preserve it and retry.")
            if artifact.exists():
                _path(artifact, private=True)
                artifact.rename(directory / f"artifact.failed-{uuid4().hex}")
        elif any(p.name != ".worker.lock" for p in directory.iterdir()):
            raise EvidenceError(f"Unrecorded cache files at {directory}; use a NEW batch directory.")
        entry = {"schemaVersion": 1, "kind": "paired-video-stage-cache",
                 "identity": identity, "status": "running"}
        _atomic(manifest_path, entry)
        try:
            path = Path(operation(artifact))
            worker.check_sources()
            hashes = {}
            for file in artifact.rglob("*"):
                _path(file, private=True)
                if file.is_file():
                    hashes[str(file)] = sha256(file)
            if str(path) not in hashes:
                raise EvidenceError("Cached stage did not produce its report in the owned cache directory.")
            entry.update(status="complete", path=str(path), artifacts=hashes)
            _atomic(manifest_path, entry)
            return path, "processed"
        except Exception as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            _atomic(manifest_path, entry)
            raise


def _run(worker):
    r, p = worker.request, worker.paths
    reuse = r["reuse"]
    alignment = worker.stage("alignment", lambda _d: _cached(
        worker, "alignment", lambda d: _align(r["video"], r["audio"], d)),
                             reuse=reuse.get("alignment"))
    if "alignment" in worker.receipt:
        entry = worker.receipt["alignment"]
        _verify_hashes(entry["artifacts"], "reviewed alignment")
        alignment = p["alignment"] = Path(entry["path"])
    if _validate_report("alignment", alignment, r, worker.sources, p)["status"] != "supported":
        return worker.action("alignment", "Soundtrack alignment is ambiguous; supply the reviewed source-video second for trimmed-audio zero.",
                             worker.command("review", "--alignment-offset", "<SECONDS>"), alignment)
    inspection = worker.stage("inspection", lambda _d: _cached(
        worker, "inspection", lambda d: inspect_shots(r["video"], d)[0]), reuse=reuse.get("shots"))
    if "shots" in worker.receipt:
        entry = worker.receipt["shots"]
        _verify_hashes(entry["artifacts"], "reviewed shots")
        inspection = p["inspection"] = Path(entry["path"])
    _validate_report("shots", inspection, r, worker.sources, p)
    reviewed_cuts = worker.receipt.get("shots", {}).get("reviewedShotsSha256") == sha256(inspection)
    if r["reviewMode"] == "automatic" and not reviewed_cuts:
        automatic = _json(inspection).get("automaticReview", {})
        if automatic.get("method") != "conservative-cut-boundaries-v1":
            if any(name in reuse for name in ("hands", "geometry", "roles")):
                raise EvidenceError("Unreviewed cached observations cannot be assigned new automatic cuts. Accept their shot boundaries or omit observation reuse entries.")
            inspection = worker.stage("automatic-inspection", lambda d: prepare_automatic_shots(
                r["video"], p["inspection"], d)[0], dependencies=("inspection",))
            p["inspection"] = inspection
    elif not reviewed_cuts:
        return worker.action("shots", "Review camera boundaries before detection; accept or add cuts in source-video seconds. Thumbnails and scores are not approval.",
                             worker.command("review", "--accept-shots"), inspection)
    timeline = worker.stage("timeline", lambda d: _timeline(r["video"], inspection, d, reuse.get("hands")), dependencies=("inspection",))
    clips = _clips(r, worker.sources, alignment, inspection, timeline)
    if "hands" in reuse or "geometry" in reuse:
        if "hands" not in reuse:
            coverage = _json(inspection)
            if coverage["firstPts"] < clips[0]["startPts"] or coverage["lastPts"] >= clips[-1]["endPtsExclusive"]:
                raise EvidenceError("Broad geometry reuse requires its cached hands; omit geometry to prepare only the aligned clip envelope.")
        shots = worker.stage("shots", None, dependencies=("inspection", "alignment", "timeline"), reuse=inspection)
        if "hands" in reuse:
            with np.load(Path(reuse["hands"]).with_name("hands.npz"), allow_pickle=False) as arrays:
                pts = arrays["pts"].copy()
        else:
            pts = np.asarray(_json(timeline)["pts"], np.int64)
        validate_clips(clips, pts, _json(shots), allow_partial_end=r["clips"] is None)
    else:
        shots = worker.stage("shots", lambda d: _select_interval(r["video"], inspection, timeline, clips, d),
                             dependencies=("inspection", "alignment", "timeline"))
    _validate_report("shots", shots, r, worker.sources, p)

    def hands(d):
        _path(r["handModel"], file=True)
        if r["poseModel"]:
            _path(r["poseModel"], file=True)
        return track_hands(r["video"], shots, r["handModel"], d, pose_model_path=r["poseModel"])[0]

    hands_path = worker.stage("hands", hands, dependencies=("shots",), reuse=reuse.get("hands"))
    _validate_report("hands", hands_path, r, worker.sources, p)
    annotations = worker.stage("annotations", lambda d: _unavailable_annotations(shots, d),
                               dependencies=("shots", "hands"), reuse=reuse.get("annotations"))
    geometry = worker.stage("geometry", lambda d: _unavailable_geometry(shots, hands_path, annotations, d),
                            dependencies=("shots", "annotations"), reuse=reuse.get("geometry"))
    _validate_report("geometry", geometry, r, worker.sources, p)
    roles = worker.stage("roles", lambda d: assign_hand_roles(
        r["video"], shots, hands_path, geometry, annotations, d,
        config=RoleConfig(plucking_screen_side=r["pluckingScreenSide"]))[0],
        dependencies=("shots", "hands", "geometry", "annotations"), reuse=reuse.get("roles"))
    _validate_report("roles", roles, r, worker.sources, p)
    fretboard = None
    if r["fretboardModel"]:
        fretboard = worker.stage(
            "fretboard",
            lambda d: track_fretboard(
                r["video"], shots, r["fretboardModel"], d,
                detector_config=FretboardDetectorConfig(device=r["fretboardDevice"]),
            )[0],
            dependencies=("shots",), reuse=reuse.get("fretboard"),
        )
        _validate_report("fretboard", fretboard, r, worker.sources, p)
    if fretboard is not None:
        bundle_operation = lambda d: prepare_paired_inputs_v5(
            r["video"], shots, hands_path, geometry, annotations, roles, fretboard,
            r["audio"], alignment, d, clips, pair_id=r["id"],
        )[0]
    else:
        bundle_operation = lambda d: prepare_paired_inputs(
            r["video"], shots, hands_path, geometry, annotations, roles,
            r["audio"], alignment, d, clips, pair_id=r["id"],
        )[0]
    bundle_dependencies = ("shots", "hands", "geometry", "annotations", "roles", "alignment") + (("fretboard",) if fretboard is not None else ())
    bundle = worker.stage("bundle", bundle_operation,
        dependencies=bundle_dependencies,
        reuse=reuse.get("bundle"))
    bundle_document = _validate_report("bundle", bundle, r, worker.sources, p)
    if bundle_document["clips"] != request_clips(clips):
        raise EvidenceError("Reused bundle clip scope differs from requested clips.")
    result = worker.result("ready")
    result["reviewMode"] = r["reviewMode"]
    return result


def run_request(request_path, *, reset_failed=False):
    """Run until all existing-format artifacts are ready or explicit review is needed."""
    request_path, request = _request(request_path)
    with _lock(Path(request["outputDirectory"])):
        return _run(_Worker(request_path, request, reset_failed=reset_failed))


def review_request(request_path, *, accept_shots=False, add_cuts=(), alignment_offset=None):
    """Record the supplied review decision, then resume the same normalized request."""
    request_path, request = _request(request_path)
    if add_cuts and not accept_shots or not (accept_shots or alignment_offset is not None):
        raise EvidenceError("Review requires --accept-shots (with optional --add-cut), or --alignment-offset.")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in add_cuts):
        raise EvidenceError("Added cut seconds must be finite and positive.")
    if alignment_offset is not None and (type(alignment_offset) not in (int, float) or not math.isfinite(alignment_offset)):
        raise EvidenceError("Alignment offset must be finite.")
    with _lock(Path(request["outputDirectory"])):
        worker = _Worker(request_path, request)
        entries = worker.ledger["stages"]
        if any(name in entries for name in ("hands", "annotations", "geometry", "roles", "bundle")):
            raise EvidenceError("Review would stale downstream observations; use a NEW output directory.")
        receipt = {**worker.receipt, "schemaVersion": 1, "kind": "paired-video-review-receipt",
                   "identitySha256": worker.identity}
        if alignment_offset is not None:
            if "timeline" in entries:
                raise EvidenceError("Alignment review would stale selected intervals; use a NEW output directory.")
            if "alignment" in receipt:
                if _json(receipt["alignment"]["path"])["videoStartSecondsForTrimmedAudioZero"] != alignment_offset:
                    raise EvidenceError("Alignment already reviewed differently; use a NEW output directory.")
            else:
                original = entries.get("alignment")
                if not original or original["status"] != "complete":
                    raise EvidenceError("Run preparation before reviewing alignment.")
                _verify_hashes(original["artifacts"], "alignment")
                document = _json(original["path"])
                path = worker.output / "alignment-reviewed" / "alignment.json"
                if path.exists():
                    raise EvidenceError("Unrecorded alignment review exists; use a NEW output directory.")
                document.update(status="supported", reviewRequired=False,
                                videoStartSecondsForTrimmedAudioZero=alignment_offset,
                                method="manual-timestamp", reviewMethod="explicit-worker-review-command",
                                originalAlignmentSha256=sha256(original["path"]), rate=[1, 1])
                _atomic(path, document)
                receipt["alignment"] = {"path": str(path), "artifacts": _artifacts(path, "alignment")}
        if accept_shots:
            original = entries.get("inspection")
            if not original or original["status"] != "complete":
                raise EvidenceError("Run preparation to inspect shots before accepting boundaries.")
            _verify_hashes(original["artifacts"], "shot inspection")
            if "shots" in receipt:
                if receipt["shots"].get("addedCutSeconds", []) != list(add_cuts):
                    raise EvidenceError("Shots already reviewed differently; use a NEW output directory.")
            else:
                path = Path(original["path"])
                if add_cuts:
                    directory = worker.output / "inspection-reviewed"
                    if directory.exists():
                        raise EvidenceError("Unrecorded reviewed inspection exists; use a NEW output directory.")
                    review_path = worker.output / "shot-boundary-review.json"
                    _atomic(review_path, {"schemaVersion": 1, "kind": "video-shot-boundary-review",
                                         "videoSha256": worker.sources["video"],
                                         "addBoundaries": [{"seconds": v, "type": "hard_cut",
                                                            "note": "Explicit worker review command"} for v in add_cuts]})
                    path, _ = inspect_shots(request["video"], directory, review_path=review_path)
                receipt["shots"] = {"path": str(path), "artifacts": _artifacts(path, "shots"),
                                    "reviewedShotsSha256": sha256(path), "addedCutSeconds": list(add_cuts),
                                    "reviewMethod": "explicit-worker-review-command"}
        worker.check_sources()
        _atomic(worker.receipt_path, receipt)
        worker.ledger["reviewReceiptSha256"] = sha256(worker.receipt_path)
        worker.save()
    return run_request(request_path)


def _result_path(path, result, request_path):
    path = _path(path)
    if not path.is_relative_to(REPOSITORY_ROOT / "runs"):
        raise EvidenceError("Worker result must remain under the project's private runs directory.")
    if path == Path(request_path):
        raise EvidenceError("Result cannot overwrite the request.")
    request = _json(request_path)
    output = Path(request.get("outputDirectory", ""))
    if path.is_relative_to(output) or path in {Path(v) for v in result.get("artifacts", {}).values()}:
        raise EvidenceError("Result must be outside stage output directories and input artifacts.")
    protected = [request.get(n) for n in ("video", "audio", "handModel", "poseModel", "fretboardModel")]
    protected.extend(request.get("reuse", {}).values())
    if path in {Path(p) for p in protected if p}:
        raise EvidenceError("Result cannot overwrite any source.")
    if path.exists():
        old = _json(path)
        if old.get("kind") != "paired-video-preparation-result" or old.get("id") != result["id"]:
            raise EvidenceError("Refusing to overwrite an unrelated result artifact.")
    return path


def _publish_result(path, result, request_path):
    path = _result_path(path, result, request_path)
    _atomic(path, result, private=False)


def main(argv=None):
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--request", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--reset-failed", action="store_true")
    review = commands.add_parser("review")
    review.add_argument("--request", required=True)
    review.add_argument("--output")
    review.add_argument("--accept-shots", action="store_true")
    review.add_argument("--add-cut", type=float, action="append", default=[])
    review.add_argument("--alignment-offset", type=float)
    args = parser.parse_args(argv)
    if args.command == "review" and args.output is None:
        args.output = str(Path(args.request).with_name("result.json"))
    try:
        if args.output:
            _result_path(args.output, {"id": _json(args.request).get("id")}, args.request)
        if args.command == "run":
            result = run_request(args.request, reset_failed=args.reset_failed)
        else:
            result = review_request(args.request, accept_shots=args.accept_shots,
                                    add_cuts=args.add_cut, alignment_offset=args.alignment_offset)
        if args.output:
            _publish_result(args.output, result, args.request)
    except (EvidenceError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Video preparation blocked: {error}", file=sys.stderr)
        # Failure is explicit, never a success-shaped partial bundle.
        if args.output:
            try:
                request = _json(args.request)
                sources = {}
                for name in ("video", "audio"):
                    value = request.get(name)
                    if isinstance(value, str) and Path(value).is_file():
                        sources[name] = sha256(_path(value, file=True))
                    else:
                        sources[name] = None
                result = {"schemaVersion": 1, "kind": "paired-video-preparation-result", "id": request.get("id"),
                          "status": "blocked", "inputSha256": sources, "artifacts": {}, "stageSummary": [],
                          "actions": [{"stage": "preparation", "reason": str(error), "command": [], "path": str(args.request)}],
                          "sourceTimePolicy": SOURCE_TIME_POLICY}
                _publish_result(args.output, result, args.request)
            except (EvidenceError, OSError, ValueError) as publication_error:
                print(f"Could not publish blocked result: {publication_error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
