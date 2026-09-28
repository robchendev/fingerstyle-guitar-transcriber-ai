"""Validate self-contained private training releases without preparation history."""

import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re

import soundfile as sf

from .dataset_io import read_json, sha256
from .percussion_supervision import percussion_annotation_coverage
from .training_windows import projected_targets, range_sample_bounds, targets_in_window
from .score_alignment import ScoreClock


RELEASE_KIND = "local-training-dataset"
VOICE_SUPERVISION_POLICIES = (
    "native-multivoice", "intentional-single-voice", "flattened-or-unknown",
)
_DIGEST = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
_CONFIRMATIONS = ("authorizedUse", "recordingAndTargetPitchConfirmed", "notationReviewed", "approveExperimentalRangesAndSplit", "groupingConfirmed")


def candidate_digest(candidate):
    return hashlib.sha256(json.dumps(candidate, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def validation_groups(document):
    if not isinstance(document, dict) or ("validationGroups" in document) == ("validationGroup" in document):
        raise ValueError("Specify exactly one of validationGroups or historical validationGroup.")
    groups = document["validationGroups"] if "validationGroups" in document else [document["validationGroup"]]
    if not isinstance(groups, list) or not groups or any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("Validation groups must be a nonempty list of nonempty string IDs.")
    if len(groups) != len(set(groups)):
        raise ValueError("Validation group IDs must be unique.")
    return tuple(groups)


def release_scope(records):
    return [{
        "id": entry["id"], "groupId": entry["groupId"], "split": entry["split"],
        "audioSha256": entry["audioSha256"],
        "sourceGpSha256": payload["approval"]["sourceGpSha256"],
        "canonicalSha256": candidate_digest(payload["canonical"]),
        "normalizationSha256": candidate_digest(payload["normalization"]),
        "candidateSha256": payload["approval"]["candidateSha256"],
        "approvedClipRanges": payload["approval"]["approvedClipRanges"],
        "windowsSha256": candidate_digest(payload["windows"]),
        **({"downbeatConditioning": payload["downbeatConditioning"]} if "downbeatConditioning" in payload else {}),
        **({"percussionAnnotationsComplete": payload["approval"]["percussionAnnotationsComplete"]} if "percussionAnnotationsComplete" in payload["approval"] else {}),
        **({"voiceSupervisionPolicy": payload["approval"]["voiceSupervisionPolicy"]} if "voiceSupervisionPolicy" in payload["approval"] else {}),
    } for entry, payload in records]


def _validate_authorization(manifest, records):
    authorization = manifest["releaseAuthorization"]
    if not isinstance(authorization, dict) or type(authorization.get("schemaVersion")) is not int or authorization["schemaVersion"] != 1 or authorization.get("kind") != "local-release-authorization":
        raise ValueError("The complete release requires explicit batch authorization.")
    reviewer = authorization["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip() or any(authorization.get(field) is not True for field in ("authorizedUse", "approveExperimentalRangesAndSplit", "groupingConfirmed")) or authorization.get("distributionAuthorized") is not False:
        raise ValueError("Invalid release reviewer or batch authorization decisions.")
    groups = validation_groups(manifest)
    if authorization["version"] != manifest["version"] or validation_groups(authorization) != groups:
        raise ValueError("Batch authorization references a different release or validation group list.")
    if set(groups) - {entry["groupId"] for entry, _ in records}:
        raise ValueError("Every validation group must be present in the selected recordings.")
    digest = candidate_digest({key: value for key, value in authorization.items() if key != "sha256"})
    if authorization.get("sha256") != digest or candidate_digest(authorization["selectedScope"]) != candidate_digest(release_scope(records)):
        raise ValueError("The selected dataset scope differs from its batch authorization.")
    for entry, payload in records:
        approval = payload["approval"]
        expected_split = "validation" if entry["groupId"] in groups else "train"
        if entry["split"] != expected_split or approval.get("releaseAuthorizationSha256") != digest or approval.get("releaseReviewer") != reviewer:
            raise ValueError("A recording approval differs from the authorized batch split or reviewer.")


def _regular_file(path):
    path = Path(path).absolute()
    if path.resolve() != path or any(part.is_symlink() or getattr(part.stat(), "st_file_attributes", 0) & 0x400 for part in (path, *path.parents)):
        raise ValueError(f"Release paths must not use aliases: {path}")
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"Expected an independent regular release file: {path}")
    return path


def release_path(root, relative, folder):
    if not isinstance(relative, str):
        raise ValueError("Release paths must be relative strings.")
    windows = PureWindowsPath(relative)
    if windows.is_absolute() or windows.drive or windows.root or ".." in windows.parts or ":" in relative or not windows.parts or windows.parts[0] != folder:
        raise ValueError(f"Release path must stay under {folder}: {relative}")
    return _regular_file(Path(root).joinpath(*windows.parts))


def _number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def validate_mapping(mapping, clock, duration):
    if not isinstance(mapping, list) or len(mapping) < 2:
        raise ValueError("Release requires at least two mapped timing coordinates.")
    _number(duration, "Audio duration")
    previous = (-1., -1.)
    for point in mapping:
        clip = _number(point["clipSeconds"], "Mapped clip time")
        reference = _number(point["referenceSeconds"], "Mapped reference time")
        quarter = _number(point["scoreQuarter"], "Mapped score position")
        if not 0 <= clip <= duration or not 0 <= reference <= clock.duration_seconds or clip < previous[0] or reference <= previous[1]:
            raise ValueError("Timing must have nondecreasing audio and strictly increasing reference time within the score/audio.")
        if abs(clock.quarter_at(reference) - quarter) > 1e-7:
            raise ValueError("Mapped score position disagrees with the nominal tempo clock.")
        previous = clip, reference


def mapping_risks(mapping):
    risks = []
    for left, right in zip(mapping, mapping[1:]):
        if left["clipSeconds"] == right["clipSeconds"]:
            if risks and risks[-1]["clipSeconds"] == left["clipSeconds"]:
                risks[-1]["referenceSecondsEnd"] = right["referenceSeconds"]
            else:
                risks.append({"kind": "score-time-plateau", "clipSeconds": left["clipSeconds"], "referenceSecondsStart": left["referenceSeconds"], "referenceSecondsEnd": right["referenceSeconds"]})
    return risks


def _validate_payload(entry, payload):
    if not isinstance(payload, dict) or type(payload.get("schemaVersion")) is not int or payload["schemaVersion"] != 1 or payload.get("kind") != "local-training-targets" or payload.get("id") != entry["id"]:
        raise ValueError("Release targets have the wrong identity or schema.")
    approval = payload["approval"]
    if not isinstance(approval, dict) or any(approval.get(field) is not True for field in _CONFIRMATIONS):
        raise ValueError("Release requires explicit source, notation, range, use and grouping approval.")
    if "percussionAnnotationsComplete" in approval and type(approval["percussionAnnotationsComplete"]) is not bool:
        raise ValueError("Percussion completeness approval must be an explicit boolean.")
    if "voiceSupervisionPolicy" in approval and approval["voiceSupervisionPolicy"] not in VOICE_SUPERVISION_POLICIES:
        raise ValueError("Voice supervision policy is unsupported.")
    if approval.get("percussionAnnotationsComplete") is True and (not isinstance(approval.get("reviewer"), str) or not approval["reviewer"].strip()):
        raise ValueError("Percussion completeness requires a source-bound reviewer.")
    if approval.get("groupId") != entry["groupId"] or approval.get("split") != entry["split"]:
        raise ValueError("Reviewed grouping or split differs from the released recording.")
    _digest(approval["sourceGpSha256"], "Source GP hash")
    if approval["audioSha256"] != entry["audioSha256"] or approval["candidateSha256"] != candidate_digest(payload["candidate"]):
        raise ValueError("Reviewed audio or alignment differs from the released payload.")
    labels, normalization = payload["canonical"], payload["normalization"]
    if labels.get("scoreTimingResolved") is not True or labels.get("timeUnit") != "quarter-note":
        raise ValueError("Release requires resolved normalized score-time labels.")
    for name in ("notes", "gestures"):
        identifiers = [value["id"] for value in labels["targets"][name]]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Canonical target IDs must be unique.")
    clock = ScoreClock(labels, normalization)
    mapping = payload["candidate"]["denseMapping"]
    duration = entry["sampleCount"] / entry["sampleRate"]
    validate_mapping(mapping, clock, duration)
    downbeat = payload.get("downbeatConditioning")
    if downbeat is not None:
        if not isinstance(downbeat, dict) or set(downbeat) != {
            "pickupPresent", "scoreQuarter", "clipSeconds", "source",
        }:
            raise ValueError("Downbeat conditioning has an unsupported contract.")
        if type(downbeat["pickupPresent"]) is not bool or downbeat["source"] != "normalized-gp-pickup-and-reviewed-score-audio-alignment":
            raise ValueError("Downbeat conditioning provenance is invalid.")
        quarter = downbeat["scoreQuarter"]
        if (
            not isinstance(quarter, list) or len(quarter) != 2
            or any(type(value) is not int for value in quarter)
            or quarter[1] <= 0
        ):
            raise ValueError("Downbeat score position must be an exact fraction.")
        expected_quarter = clock.measure_starts[1 if downbeat["pickupPresent"] else 0]
        if quarter[0] / quarter[1] != float(expected_quarter):
            raise ValueError("Downbeat score position differs from the normalized pickup contract.")
        seconds = downbeat["clipSeconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= duration:
            raise ValueError("Downbeat timestamp must lie inside the released audio.")
        left = max(
            index for index, point in enumerate(mapping)
            if point["scoreQuarter"] <= float(expected_quarter)
        )
        right = min(left + 1, len(mapping) - 1)
        a, b = mapping[left], mapping[right]
        expected_seconds = a["clipSeconds"] if a["scoreQuarter"] == b["scoreQuarter"] else (
            a["clipSeconds"]
            + (float(expected_quarter) - a["scoreQuarter"])
            * (b["clipSeconds"] - a["clipSeconds"])
            / (b["scoreQuarter"] - a["scoreQuarter"])
        )
        if not math.isclose(seconds, expected_seconds, abs_tol=1e-8):
            raise ValueError("Downbeat timestamp differs from the reviewed score-audio alignment.")
    if mapping_risks(mapping) and approval.get("uncertaintyAcknowledged") is not True:
        raise ValueError("Timing plateaus require explicit source-bound uncertainty acceptance and approved ranges.")
    ranges = approval["approvedClipRanges"]
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("Explicit approved clip ranges are required.")
    previous_end = -1.
    for left, right in ranges:
        _number(left, "Approved range start")
        _number(right, "Approved range end")
        if not mapping[0]["clipSeconds"] <= left < right <= mapping[-1]["clipSeconds"] or left < previous_end:
            raise ValueError("Approved ranges must be ordered, nonoverlapping and inside the mapping.")
        previous_end = right
    notes, gestures = projected_targets(labels, payload["candidate"], clock)
    percussion_coverage = percussion_annotation_coverage(labels, payload["candidate"], normalization) if approval.get("percussionAnnotationsComplete") is True else None
    windows = payload["windows"]
    if not isinstance(windows, list) or not windows:
        raise ValueError("Each released recording requires nonempty windows.")
    identifiers, bounds = set(), set()
    approved_samples = range_sample_bounds(ranges, entry["sampleRate"])
    for window in windows:
        identifier = window["windowId"]
        start, stop = window["startSample"], window["stopSampleExclusive"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers or type(start) is not int or type(stop) is not int or not 0 <= start < stop <= entry["sampleCount"]:
            raise ValueError("Invalid or duplicate released window.")
        if (start, stop) in bounds:
            raise ValueError("Duplicate released sample bounds.")
        identifiers.add(identifier)
        bounds.add((start, stop))
        if not 2 * entry["sampleRate"] <= stop - start <= 8 * entry["sampleRate"]:
            raise ValueError("Released windows must last between two and eight seconds.")
        if not any(a <= start < stop <= b for a, b in approved_samples):
            raise ValueError("A released window crosses an unapproved interval.")
        expected = targets_in_window(notes, gestures, start, stop, entry["sampleRate"], percussion_coverage=percussion_coverage)
        if candidate_digest(window["targets"]) != candidate_digest(expected):
            raise ValueError("Released targets or uncertainty masks differ from canonical projection.")


def _validate_release(manifest_path):
    path = _regular_file(manifest_path)
    manifest = read_json(path)
    if not isinstance(manifest, dict) or type(manifest.get("schemaVersion")) is not int or manifest["schemaVersion"] != 1 or manifest.get("kind") != RELEASE_KIND or manifest.get("trainingReady") is not True or manifest.get("visibility") != "private" or manifest.get("distributionAuthorized") is not False:
        raise ValueError("Expected a ready, private local training release.")
    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("A training release cannot be empty.")
    guards = {path: sha256(path)}
    groups, recordings, scores, identifiers, used_paths = {}, {}, {}, set(), set()
    counts = {"train": 0, "validation": 0}
    records = []
    for entry in entries:
        identifier, group, split = entry["id"], entry["groupId"], entry["split"]
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier) or identifier.casefold() in identifiers:
            raise ValueError("Recording IDs must be unique safe filenames.")
        identifiers.add(identifier.casefold())
        if not isinstance(group, str) or not group.strip() or split not in counts:
            raise ValueError("Each recording requires an explicit group and train/validation split.")
        if group in groups and groups[group] != split:
            raise ValueError("A related recording group crosses training and validation.")
        groups[group] = split
        for field in ("sampleRate", "channels", "sampleCount"):
            if type(entry[field]) is not int or entry[field] <= 0:
                raise ValueError(f"Audio {field} must be a positive integer.")
        if entry["channels"] > 8 or entry["sampleCount"] / entry["sampleRate"] > 900:
            raise ValueError("Recordings must have at most eight channels and fifteen minutes.")
        audio = release_path(path.parent, entry["audioPath"], "audio")
        target = release_path(path.parent, entry["targetsPath"], "targets")
        if audio.suffix.lower() != ".flac" or target.suffix.lower() != ".json":
            raise ValueError("Release assets must be FLAC audio and JSON targets.")
        for asset, field in ((audio, "audioSha256"), (target, "targetsSha256")):
            if asset in used_paths or sha256(asset) != _digest(entry[field], field):
                raise ValueError("A release asset is reused or its checksum changed.")
            used_paths.add(asset)
            guards[asset] = entry[field]
        audio_hash = entry["audioSha256"]
        if audio_hash in recordings and recordings[audio_hash] != split:
            raise ValueError("Identical audio crosses training and validation.")
        recordings[audio_hash] = split
        with sf.SoundFile(audio) as stream:
            if stream.format != "FLAC" or (stream.samplerate, stream.channels, len(stream)) != (entry["sampleRate"], entry["channels"], entry["sampleCount"]):
                raise ValueError("Release audio properties differ from the manifest.")
        payload = read_json(target)
        _validate_payload(entry, payload)
        score_hash = payload["approval"]["sourceGpSha256"]
        if score_hash in scores and scores[score_hash] != split:
            raise ValueError("Identical GP scores cross training and validation.")
        scores[score_hash] = split
        counts[split] += len(payload["windows"])
        records.append((entry, payload))
    declared_counts = manifest["counts"]["windowsBySplit"]
    if not isinstance(declared_counts, dict) or any(type(value) is not int for value in declared_counts.values()) or not all(counts.values()) or declared_counts != counts:
        raise ValueError("Release split counts are empty or inconsistent.")
    _validate_authorization(manifest, records)
    for asset, expected in guards.items():
        if sha256(asset) != expected:
            raise ValueError("A release asset changed while it was being loaded.")
    return manifest, records, guards


def validate_release(manifest_path):
    try:
        return _validate_release(manifest_path)
    except (KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        raise ValueError(f"Malformed local training release: {error}") from error
