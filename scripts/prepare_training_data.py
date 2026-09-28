"""Prepare local GP/audio/video data without an LLM; only batch-train starts training."""

import argparse
from collections.abc import Sequence
from copy import deepcopy
from io import BytesIO
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from uuid import uuid4

import numpy as np
import soundfile as sf

from .audio_alignment import AlignmentError, align_first_attack, audio_features, reference_features
from .audio_tools import AcquisitionError, executable, probe, run_media
from .canonical_events import canonical_counts, canonicalize, dead_note_marks, fraction, pitched_note_marks
from .dataset_io import ROOT, read_json, sha256
from .dataset_release import candidate_digest, mapping_risks, release_scope, validate_mapping, validate_release, validation_groups as read_validation_groups
from .gp_events import catalog_timing, decode_score, performance_events, playback_order
from .gp_normalization import normalize_gp_bytes
from .inspect_gp_files import GpInspectionError, inspect_gp, read_gp
from .percussion_supervision import percussion_annotation_coverage
from .training_windows import projected_targets, range_sample_bounds, sample_windows, targets_in_window
from .score_alignment import AlignmentInputError, ScoreClock, candidate_mapping, matching_events


CONVENTIONS = {
    "schemaVersion": 1, "uppercaseOIsWristThump": True, "simultaneousTextPriority": "lowest-voice-index",
    "plainXIsThumbSlap": True, "ghostXIsPercussiveHit": True,
    "id": "fingerstyle-v1", "shortTextIsPercussiveHit": False,
}
CONFIRMATIONS = {
    "authorize_use": "authorizedUse", "confirm_pitch": "recordingAndTargetPitchConfirmed",
    "confirm_notation": "notationReviewed", "approve_experimental_ranges": "approveExperimentalRangesAndSplit",
    "confirm_grouping": "groupingConfirmed",
}
RAW_GP_NAME = "raw.gp"
TRIMMED_AUDIO_NAME = "trimmed.flac"
SOURCE_AUDIO_PREFIX = "trimmed-source"
ARTIFACTS = (RAW_GP_NAME, TRIMMED_AUDIO_NAME, "normalized.gp", "events.json", "canonical.json", "normalization.json", "alignment.json", "notation.json")
CODE_FILES = (
    "prepare_training_data.py", "inspect_gp_files.py", "gp_events.py", "canonical_events.py",
    "gp_normalization.py", "settings.py", "audio_alignment.py", "score_alignment.py",
    "training_windows.py", "dataset_release.py", "dataset_io.py", "audio_tools.py", "percussion_supervision.py",
)


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value) or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", value, re.I):
        raise ValueError("IDs/versions need 1-80 filename-safe ASCII letters, digits, '-' or '_'; no Windows device names.")
    return value


def regular_path(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or part.exists() and getattr(part.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Directory aliases and links are not allowed: {part}")
    if path.resolve() != path or path.is_file() and path.stat().st_nlink != 1:
        raise ValueError(f"Expected an independent path without aliases: {path}")
    return path


def workspace_path(path):
    path = regular_path(path)
    if path == ROOT or ROOT.is_relative_to(path):
        raise ValueError("Choose a separate private workspace, not the repository or an ancestor.")
    if any(path.is_relative_to(ROOT / name) for name in ("cache", ".git")) or any(path.is_relative_to(ROOT / "data" / name) for name in ("pairs", "releases", "archive")):
        raise ValueError("Use a workspace root, not generated assets, archived evidence, cache or Git directories.")
    return path


def write_bytes(path, content):
    path = regular_path(path)
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        with staging.open("xb") as stream:
            stream.write(content)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def write_json(path, value):
    write_bytes(path, (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"))


def initialize(workspace):
    workspace = workspace_path(workspace)
    path = workspace / "pairs.json"
    if path.is_file():
        load_pairs(workspace)
        return path
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("init requires an empty directory; existing preparation/assets are never adopted.")
    write_bytes(workspace / ".gitignore", b"*\n")
    write_json(path, {"schemaVersion": 1, "kind": "local-pairs", "pairs": []})
    return path


def load_pairs(workspace):
    workspace = workspace_path(workspace)
    document = read_json(regular_path(workspace / "pairs.json"))
    if not isinstance(document, dict) or document.get("schemaVersion") != 1 or document.get("kind") != "local-pairs" or not isinstance(document.get("pairs"), list):
        raise ValueError("Expected a version-1 local-pairs pairs.json; run init for a new workspace.")
    identifiers = set()
    for pair in document["pairs"]:
        if not isinstance(pair, dict) or set(pair) - {"id", "groupId", "performerId", "title", "gpPath", "audioPath"}:
            raise ValueError("Pairs accept id, groupId, optional performerId/title, gpPath and audioPath only.")
        identifier = safe_id(pair.get("id"))
        if identifier.casefold() in identifiers:
            raise ValueError("Pair IDs must be unique, including on case-insensitive filesystems.")
        identifiers.add(identifier.casefold())
        for field in ("groupId", "gpPath", "audioPath"):
            if not isinstance(pair.get(field), str) or not pair[field].strip():
                raise ValueError(f"{identifier}: {field} must be a nonempty string.")
        if "performerId" in pair and (not isinstance(pair["performerId"], str) or not pair["performerId"].strip()):
            raise ValueError("Optional performerId must be a nonempty, explicitly supplied identity.")
        if "title" in pair and (not isinstance(pair["title"], str) or not pair["title"].strip()):
            raise ValueError("Optional title must be a nonempty string.")
    return document


def selection(workspace, identifiers=None):
    pairs = load_pairs(workspace)["pairs"]
    if identifiers is not None:
        if not identifiers or len(identifiers) != len(set(identifiers)) or set(identifiers) - {pair["id"] for pair in pairs}:
            raise ValueError("Requested IDs must be a unique subset of pairs.json.")
        by_id = {pair["id"]: pair for pair in pairs}
        pairs = [by_id[identifier] for identifier in identifiers]
    if not pairs:
        raise ValueError("No local pairs selected; use add or edit pairs.json.")
    return pairs


def rules_template():
    return {
        "schemaVersion": 1, "acceptConventions": False, "rules": [],
        "confirmFixedTuningCapoText": False, "confirmFullCapoMetadata": False, "fineTarget": None,
    }


def add_pair(workspace, identifier, group, gp, audio, performer=None, *, title=None):
    workspace = workspace_path(workspace)
    document = load_pairs(workspace)
    safe_id(identifier)
    if identifier.casefold() in {pair["id"].casefold() for pair in document["pairs"]}:
        raise ValueError("Pair ID already exists; edit pairs.json then deliberately invalidate to revise it.")
    gp, audio = regular_path(gp), regular_path(audio)
    if not gp.is_file() or not audio.is_file() or gp.suffix.lower() != ".gp" or gp == audio:
        raise ValueError("Select a modern .gp score and a separate already-trimmed local audio file.")
    suffix = audio.suffix.lower()
    if suffix not in {".flac", ".wav", ".aif", ".aiff", ".mp3", ".m4a", ".aac", ".ogg", ".opus"}:
        raise ValueError("Unsupported local source audio format.")
    if any(path.is_relative_to(workspace / name) for path in (gp, audio) for name in ("pairs", "releases", "archive")):
        raise ValueError("Import original external files, not another pair, release or archive's artifacts.")
    directory = regular_path(workspace / "pairs" / identifier)
    if directory.exists():
        raise ValueError("An unregistered pair directory already exists; do not overwrite its files.")
    audio_name = TRIMMED_AUDIO_NAME if suffix == ".flac" else f"{SOURCE_AUDIO_PREFIX}{suffix}"
    pair = {"id": identifier, "groupId": group, "gpPath": f"pairs\\{identifier}\\{RAW_GP_NAME}", "audioPath": f"pairs\\{identifier}\\{audio_name}"}
    if performer is not None:
        pair["performerId"] = performer
    if title is not None:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Optional title must be a nonempty string.")
        pair["title"] = title
    if not isinstance(group, str) or not group.strip() or performer is not None and not performer.strip():
        raise ValueError("Group and supplied performer identities must be nonempty.")
    original = {gp: sha256(gp), audio: sha256(audio)}
    staging = regular_path(workspace / "pairs" / f".importing-{uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        for source, name in ((gp, RAW_GP_NAME), (audio, audio_name)):
            shutil.copyfile(source, staging / name)
            if sha256(staging / name) != original[source] or sha256(source) != original[source]:
                raise ValueError("A source changed during import; no pair was registered.")
        write_json(staging / "rules.json", rules_template())
        staging.rename(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    document["pairs"].append(pair)
    write_json(workspace / "pairs.json", document)
    return pair


def source_paths(workspace, pair):
    directory = regular_path(workspace / "pairs" / safe_id(pair["id"]))
    paths = []
    for field in ("gpPath", "audioPath"):
        path = Path(pair[field])
        if path.is_absolute() or path.drive or ".." in path.parts:
            raise ValueError("Registered inputs must be workspace-relative owned source files.")
        path = workspace / path
        allowed = {RAW_GP_NAME} if field == "gpPath" else {TRIMMED_AUDIO_NAME, *(f"{SOURCE_AUDIO_PREFIX}{suffix}" for suffix in (".wav", ".aif", ".aiff", ".mp3", ".m4a", ".aac", ".ogg", ".opus"))}
        if path.parent != directory or path.name not in allowed:
            raise ValueError(f"Registered inputs must be this pair's own {RAW_GP_NAME} and imported trimmed audio, never normalized or another pair's outputs.")
        path = regular_path(path)
        if not path.is_file():
            raise ValueError(f"{pair['id']}: missing local {field}: {path}")
        paths.append(path)
    if paths[0].suffix.lower() != ".gp" or paths[0] == paths[1]:
        raise ValueError("Select an immutable modern .gp score and a separate already-trimmed audio file.")
    return paths


def input_binding(workspace, pair, rules):
    gp, audio = source_paths(workspace, pair)
    return {
        "pair": pair, "sourceGpSha256": sha256(gp), "sourceAudioSha256": sha256(audio),
        "rulesSha256": candidate_digest(rules),
        "implementation": {name: sha256(Path(__file__).with_name(name)) for name in CODE_FILES},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": package_version("scipy"), "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__},
    }


def load_current(workspace, pair):
    directory = regular_path(workspace / "pairs" / pair["id"])
    state = read_json(regular_path(directory / "preparation.json"))
    if state.get("invalidated"):
        raise ValueError(f"{pair['id']}: invalidated; run prepare before review/release.")
    rules = read_json(regular_path(directory / "rules.json"))
    if input_binding(workspace, pair, rules) != state["inputs"]:
        raise ValueError(f"{pair['id']}: source, grouping, rules or implementation changed; invalidate with a reason, then prepare and review again.")
    for name, digest in state["artifacts"].items():
        if name not in ARTIFACTS or sha256(regular_path(directory / name)) != digest:
            raise ValueError(f"{pair['id']}: preparation artifact changed: {name}; invalidate and prepare again.")
    if set(state["artifacts"]) != set(ARTIFACTS):
        raise ValueError("Preparation is incomplete; invalidate and prepare again.")
    return directory, state


def invalidate(workspace, pair, reason):
    if not reason.strip():
        raise ValueError("Invalidation requires a nonempty reason.")
    directory = regular_path(workspace / "pairs" / pair["id"])
    state_path = directory / "preparation.json"
    if not state_path.exists():
        raise ValueError("No completed preparation to invalidate; fix the inputs and run prepare.")
    state = read_json(regular_path(state_path))
    if state.get("invalidated"):
        raise ValueError("This pair is already invalidated; run prepare.")
    history_path = regular_path(directory / "history.json")
    history = read_json(history_path) if history_path.exists() else []
    review_path = regular_path(directory / "review.json")
    history.append({"reason": reason, "preparation": state, "review": read_json(review_path) if review_path.exists() else None})
    write_json(history_path, history)
    state["invalidated"] = reason
    write_json(state_path, state)
    review_path.unlink(missing_ok=True)


def convert_audio(source, destination, ffmpeg_dir=None):
    """Keep FLAC bytes or encode decoded PCM once, without timeline/DSP filters."""
    suffix = source.suffix.lower()
    if suffix in {".flac", ".wav", ".aif", ".aiff"}:
        with sf.SoundFile(source) as stream:
            if stream.subtype not in {"PCM_U8", "PCM_S8", "PCM_16", "PCM_24"}:
                raise ValueError("FLAC preparation supports integer PCM up to 24 bits; float/32-bit PCM requires an explicit source conversion, not silent quantization.")
            if not 1 <= stream.channels <= 8 or not 0 < len(stream) / stream.samplerate <= 900:
                raise ValueError("Local preparation supports nonempty audio up to fifteen minutes and eight channels.")
            expected = (stream.samplerate, stream.channels, len(stream))
            if suffix == ".flac" and stream.format == "FLAC":
                shutil.copyfile(source, destination)
            else:
                subtype = "PCM_24" if stream.subtype == "PCM_24" else "PCM_16"
                with sf.SoundFile(destination, "w", samplerate=stream.samplerate, channels=stream.channels, format="FLAC", subtype=subtype) as output:
                    for block in stream.blocks(blocksize=65536, dtype="int32", always_2d=True):
                        output.write(block)
    elif suffix in {".mp3", ".m4a", ".aac", ".ogg", ".opus"}:
        ffmpeg, ffprobe = executable("ffmpeg", ffmpeg_dir), executable("ffprobe", ffmpeg_dir)
        original = probe(source, ffprobe)
        if not 1 <= original["channels"] <= 8:
            raise ValueError("Local preparation supports at most eight channels.")
        if original["sampleCount"] is not None and original["sampleCount"] / original["sampleRate"] > 900:
            raise ValueError("Local preparation supports at most fifteen minutes.")
        run_media([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(source),
            "-map", "0:a:0", "-vn", "-map_metadata", "-1", "-c:a", "flac", "-sample_fmt", "s32",
            "-bits_per_raw_sample", "24", str(destination),
        ])
        expected = (original["sampleRate"], original["channels"], None)
    else:
        raise ValueError("Supported local audio: FLAC, integer PCM WAV/AIFF, MP3, M4A, AAC, Ogg or Opus.")
    with sf.SoundFile(destination) as stream:
        actual = (stream.samplerate, stream.channels, len(stream))
        if stream.format != "FLAC" or actual[:2] != expected[:2] or expected[2] is not None and actual[2] != expected[2]:
            raise ValueError("FLAC conversion changed native rate, channels or decoded sample count.")
        if not 0 < len(stream) / stream.samplerate <= 900:
            raise ValueError("Local preparation supports nonempty audio up to fifteen minutes.")
    return {"sampleRate": actual[0], "channels": actual[1], "sampleCount": actual[2], "sha256": sha256(destination)}


def extract_score(path, pair, rules):
    inspection = inspect_gp(path)
    if inspection["trackCount"] != 1 or inspection["tracks"][0]["staffCount"] != 1:
        raise ValueError("Only one six-string guitar track/staff with fixed tuning/full capo is supported.")
    warnings = set(inspection["warnings"])
    staff = inspection["tracks"][0]["staves"][0]
    if rules["confirmFixedTuningCapoText"] is True:
        warnings.discard("capo_or_tuning_text_requires_review")
    if rules.get("confirmFullCapoMetadata") is True and not staff["partialCapoActive"] and staff["partialCapoStringFlags"] == "000000":
        warnings.discard("inconsistent_partial_capo_metadata")
    if warnings:
        raise ValueError(f"Unresolved GP inspection: {sorted(warnings)}. No tuning/capo/meter/tempo defaults are supplied.")
    root = read_gp(path)
    decoded = decode_score(root, staff["openStringMidi"], staff["capoFret"])
    music = [measure for measure in decoded["measures"] if not measure["referenceOnly"]]
    if not music:
        raise ValueError("A preparation requires nonreference musical measures.")
    for field, code in (("fermatas", "fermata_timing_requires_alignment"), ("tripletFeel", "swing_timing_requires_alignment")):
        affected = [measure["index"] for measure in music if measure[field]]
        if affected:
            decoded["issues"].append({"code": code, "measureIndices": affected})
    if any(beat["graceMode"] for beat in decoded["scoreEvents"] if not beat["referenceOnly"]):
        decoded["issues"].append({"code": "grace_timing_requires_alignment"})
    score = {
        "schemaVersion": 1, "catalogId": pair["id"], "sourceGpSha256": sha256(path),
        "timeUnit": "quarter-note", "audioAlignment": None,
        "instrument": {
            "stringOrder": [6, 5, 4, 3, 2, 1], "openStringMidi": staff["openStringMidi"],
            "capoFret": staff["capoFret"], "fretConvention": "capo-relative",
        },
        "providedTiming": catalog_timing(decoded), **decoded,
        "playback": performance_events(decoded, playback_order(decoded["measures"], fine_measure_index=music[-1]["index"] if rules.get("fineTarget") == "last-musical-measure" else None), silence_repeated_entry_ties=True),
    }
    return score, root.find("./MasterTrack/Anacrusis") is not None


def automatic_candidate(labels, normalization, audio_path):
    clock = ScoreClock(labels, normalization)
    try:
        events, omitted = matching_events(labels, clock)
        with sf.SoundFile(audio_path) as stream:
            duration = len(stream) / stream.samplerate
            if not 8000 <= stream.samplerate <= 48000:
                raise AlignmentError("Automatic DSP supports 8000-48000 Hz; keep the native audio and supply manual anchors.")
        samples, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = audio_features(samples, rate)
        reference = reference_features(events, clock.duration_seconds)
        result = align_first_attack(reference, audio, first_reference_seconds=min(event["onset"] for event in events))
        mapping = candidate_mapping(reference, audio, result, clock, allow_audio_prefix=True)
        dense = [{key: float(mapping[key][index]) for key in ("clipSeconds", "referenceSeconds", "scoreQuarter")} for index in range(len(mapping["referenceSeconds"]))]
        validate_mapping(dense, clock, duration)
        return {"denseMapping": dense, "method": "first-attack-dtw", "diagnostics": result["diagnostics"], "timingRisks": mapping_risks(dense), "omittedMatchingCues": omitted, "trainingReady": False}
    except (AlignmentError, AlignmentInputError) as error:
        return {"denseMapping": [], "method": "needs-manual-anchors", "error": str(error), "trainingReady": False}


def prepare_pair(workspace, pair, *, accept_conventions=False, ffmpeg_dir=None):
    directory = regular_path(workspace / "pairs" / pair["id"])
    directory.mkdir(parents=True, exist_ok=True)
    rules_path = regular_path(directory / "rules.json")
    rules = read_json(rules_path) if rules_path.exists() else rules_template()
    if not isinstance(rules, dict) or set(rules) - set(rules_template()) or not (set(rules_template()) - {"fineTarget", "confirmFullCapoMetadata"}) <= set(rules) or rules["schemaVersion"] != 1 or type(rules["acceptConventions"]) is not bool or type(rules["confirmFixedTuningCapoText"]) is not bool or type(rules.get("confirmFullCapoMetadata", False)) is not bool or not isinstance(rules["rules"], list) or rules.get("fineTarget") not in (None, "last-musical-measure"):
        raise ValueError("rules.json requires schemaVersion:1, accepted conventions, boolean reviewed capo confirmations, rules:list and optional fineTarget:null|'last-musical-measure'.")
    state_path = regular_path(directory / "preparation.json")
    prior = read_json(state_path) if state_path.exists() else None
    if prior is not None and not prior.get("invalidated"):
        load_current(workspace, pair)
        return {"id": pair["id"], "status": "reused"}
    if accept_conventions:
        rules["acceptConventions"] = True
    write_json(rules_path, rules)
    if not rules["acceptConventions"]:
        raise ValueError("Review the documented fingerstyle-v1 notation conventions; explicitly pass --accept-conventions only if O=wrist thump, X=thumb slap and ghost X=percussive hit apply. Otherwise this normalizer is not appropriate for the source.")
    binding = input_binding(workspace, pair, rules)
    source_gp, source_audio = source_paths(workspace, pair)
    staging = regular_path(directory / f".preparing-{uuid4().hex}")
    staging.mkdir()
    try:
        shutil.copyfile(source_gp, staging / RAW_GP_NAME)
        audio = convert_audio(source_audio, staging / TRIMMED_AUDIO_NAME, ffmpeg_dir)
        score, pickup = extract_score(staging / RAW_GP_NAME, pair, rules)
        annotations = {"gpSha256": score["sourceGpSha256"], "rules": rules["rules"]}
        labels = canonicalize(score, annotations, CONVENTIONS)
        if not labels["scoreTimingResolved"]:
            raise ValueError("Unresolved score timing; correct/select the GP rather than inventing bar lengths.")
        normalized_gp, labels, normalization = normalize_gp_bytes((staging / RAW_GP_NAME).read_bytes(), score, labels, annotations, CONVENTIONS)
        if "performerId" in pair:
            labels["performerId"] = pair["performerId"]
        write_bytes(staging / "normalized.gp", normalized_gp)
        candidate = automatic_candidate(labels, normalization, staging / TRIMMED_AUDIO_NAME)
        notation = {
            "rawPickup": pickup, "normalizedPickup": read_gp(staging / "normalized.gp").find("./MasterTrack/Anacrusis") is not None,
            "conditioning": labels["conditioning"], "counts": canonical_counts(labels),
            "sourceIssues": score["issues"],
            "beats": [{
                "id": beat["id"], "referenceOnly": beat["referenceOnly"], "text": beat["text"],
                "deadStrings": sorted(note["string"] for note in beat["notes"] if note["techniques"]["dead"]),
                "deadNoteMarks": dead_note_marks(beat), "pitchedNoteMarks": pitched_note_marks(beat),
                "beatTechniques": beat["techniques"],
            } for beat in score["scoreEvents"]],
        }
        for name, value in (("events", score), ("canonical", labels), ("normalization", normalization), ("alignment", candidate), ("notation", notation)):
            write_json(staging / f"{name}.json", value)
        if input_binding(workspace, pair, rules) != binding:
            raise ValueError("Inputs changed during preparation; no outputs were accepted.")
        artifacts = {name: sha256(staging / name) for name in ARTIFACTS}
        for name in ARTIFACTS:
            write_bytes(directory / name, (staging / name).read_bytes())
        regular_path(directory / "review.json").unlink(missing_ok=True)
        regular_path(directory / "review-report.json").unlink(missing_ok=True)
        cues = regular_path(directory / "cues")
        if cues.exists():
            shutil.rmtree(cues)
        write_json(state_path, {"schemaVersion": 1, "inputs": binding, "artifacts": artifacts, "audio": audio, "invalidated": False})
    finally:
        shutil.rmtree(staging)
    return {"id": pair["id"], "status": "prepared", "alignment": candidate["method"], "alignmentError": candidate.get("error"), "counts": notation["counts"]}


def score_positions(directory):
    labels, normalization = read_json(directory / "canonical.json"), read_json(directory / "normalization.json")
    notation = read_json(directory / "notation.json")
    clock = ScoreClock(labels, normalization)
    positions = {}
    for index, quarter in enumerate(clock.measure_starts):
        source_ordinal = normalization["sourceMeasureVisits"][index]["measureIndex"] + 1
        positions[str(index + 1)] = {
            "scoreQuarter": float(quarter), "referenceSeconds": clock.seconds(quarter),
            "ordinal": index + 1, "displayedBar": index + 1 - int(notation["normalizedPickup"]),
            "sourceOrdinal": source_ordinal, "sourceDisplayedBar": source_ordinal - int(notation["rawPickup"]),
        }
    attacks = [fraction(note["onsetQuarter"], note["id"]) for note in labels["targets"]["notes"] if note["isAttack"] is True and note["sourceSegments"][0]["graceMode"] is None]
    attacks.extend(fraction(gesture["onsetQuarter"], gesture["id"]) for gesture in labels["targets"]["gestures"] if gesture.get("scoreOnsetKnown") and gesture.get("graceMode") is None)
    if attacks:
        first = min(attacks)
        positions["first-attack"] = {"scoreQuarter": float(first), "referenceSeconds": clock.seconds(first)}
    positions["end"] = {"scoreQuarter": float(clock.total_quarter), "referenceSeconds": clock.duration_seconds}
    return labels, normalization, clock, positions


def finite_seconds(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Audio seconds must be finite and nonnegative.")
    return value


def anchored_candidate(anchors, positions, duration, clock):
    points = []
    for key, seconds in anchors.items():
        if key not in positions or type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 <= seconds <= duration:
            raise ValueError("An anchor must name a performance bar ordinal, first-attack or end, and lie within the audio.")
        reference = positions[key]["referenceSeconds"]
        points.append({"scoreQuarter": clock.quarter_at(reference), "referenceSeconds": reference, "clipSeconds": seconds})
    points.sort(key=lambda point: point["referenceSeconds"])
    compact = []
    for point in points:
        if compact and point["referenceSeconds"] == compact[-1]["referenceSeconds"]:
            if point != compact[-1]:
                raise ValueError("Different timestamps name the same score position.")
            continue
        if compact and point["clipSeconds"] <= compact[-1]["clipSeconds"]:
            raise ValueError("Manual anchors must advance strictly in both score and audio time.")
        compact.append(point)
    return {"denseMapping": compact if len(compact) >= 2 else [], "method": "manual-anchors-linear-nominal-time", "anchors": anchors, "trainingReady": False}


def parse_ranges(values, duration):
    ranges = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("Ranges use START:END audio seconds, for example 1.25:19.5.")
        left, right = map(finite_seconds, parts)
        if not left < right <= duration:
            raise ValueError("Ranges must have positive length and lie within the unchanged audio.")
        ranges.append([left, right])
    ranges.sort()
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Ranges must be nonoverlapping.")
    return ranges


def first_full_bar_downbeat(labels, candidate, pickup):
    visits = labels["measureVisits"]
    if type(pickup) is not bool or pickup and len(visits) < 2:
        raise ValueError("Pickup conditioning requires a following full measure.")
    quarter = fraction(
        visits[1 if pickup else 0]["onsetQuarter"],
        "first full-bar downbeat",
    )
    mapping = candidate["denseMapping"]
    score_quarters = np.array([point["scoreQuarter"] for point in mapping], dtype=float)
    clip_seconds = np.array([point["clipSeconds"] for point in mapping], dtype=float)
    if not len(mapping) or not score_quarters[0] <= float(quarter) <= score_quarters[-1]:
        raise ValueError("First full-bar downbeat lies outside the reviewed alignment.")
    return {
        "pickupPresent": pickup,
        "scoreQuarter": [quarter.numerator, quarter.denominator],
        "clipSeconds": float(np.interp(float(quarter), score_quarters, clip_seconds)),
        "source": "normalized-gp-pickup-and-reviewed-score-audio-alignment",
    }


def effective_ranges(approved, excluded, mapping):
    if not mapping and approved:
        raise ValueError("Provide at least two explicit manual anchors before approving ranges without a usable automatic candidate.")
    result = []
    for left, right in approved:
        if not mapping[0]["clipSeconds"] <= left < right <= mapping[-1]["clipSeconds"]:
            raise ValueError("Approval cannot extend outside the mapped timeline; add explicit anchors instead of extrapolating.")
        pieces = [[left, right]]
        for a, b in excluded:
            # An excluded instant must not supervise its neighbors.
            a, b = a - .5, b + .5
            pieces = [part for x, y in pieces for part in ([[x, y]] if b <= x or a >= y else [[x, min(a, y)], [max(b, x), y]]) if part[0] < part[1]]
        result.extend(pieces)
    return result


def load_review(directory, state):
    path = regular_path(directory / "review.json")
    if not path.exists():
        return {"schemaVersion": 1, "anchors": {}, "requestedClipRanges": [], "excludedClipRanges": [], "approval": {}, "candidate": read_json(directory / "alignment.json")}
    review = read_json(path)
    if review.get("preparationSha256") != candidate_digest(state):
        raise ValueError("Review is bound to another preparation; invalidate and review the new inputs.")
    if review.get("candidateSha256") != candidate_digest(review["candidate"]):
        raise ValueError("Reviewed candidate changed outside the review CLI.")
    return review


def render_cue(directory, state, key, position):
    rate, total = state["audio"]["sampleRate"], state["audio"]["sampleCount"]
    marker = min(total - 1, math.floor(position * rate))
    start, stop = max(0, marker - 3 * rate), min(total, marker + 5 * rate)
    with sf.SoundFile(directory / TRIMMED_AUDIO_NAME) as stream:
        stream.seek(start)
        samples = stream.read(stop - start, dtype="int32", always_2d=True)
        subtype = stream.subtype
    output = regular_path(directory / "cues")
    output.mkdir(exist_ok=True)
    for name, values in (("plain", samples), ("cue", samples.astype(np.float64) / 2 ** 32)):
        if name == "cue":
            count = min(round(rate * .045), stop - marker)
            beep = .2 * np.sin(2 * np.pi * 1400 * np.arange(count) / rate) * np.hanning(count)
            values[marker - start:marker - start + count] += beep[:, None]
        content = BytesIO()
        sf.write(content, values, rate, format="WAV", subtype=subtype)
        write_bytes(output / f"{key}.{name}.wav", content.getvalue())
    return {"cuePath": str(output / f"{key}.cue.wav"), "plainPath": str(output / f"{key}.plain.wav"), "excerptStartSeconds": start / rate, "beepOffsetSeconds": (marker - start) / rate, "clipSeconds": position}


def review_pair(workspace, pair, args):
    directory, state = load_current(workspace, pair)
    labels, _, clock, positions = score_positions(directory)
    review = load_review(directory, state)
    before = deepcopy(review)
    duration = state["audio"]["sampleCount"] / state["audio"]["sampleRate"]
    anchors = {} if args.clear_anchors else dict(review["anchors"])
    for value in args.anchor or []:
        key, separator, text = value.partition("=")
        if not separator:
            raise ValueError("Anchors use ORDINAL=SECONDS, first-attack=SECONDS or end=SECONDS.")
        anchors[key] = finite_seconds(text)
    candidate = anchored_candidate(anchors, positions, duration, clock) if anchors else read_json(directory / "alignment.json")
    if candidate["denseMapping"]:
        validate_mapping(candidate["denseMapping"], clock, duration)
    approved = parse_ranges(args.approve_range, duration) if args.approve_range is not None else review["requestedClipRanges"]
    if args.clear_ranges:
        approved = []
    excluded = parse_ranges(args.exclude_range, duration) if args.exclude_range is not None else review["excludedClipRanges"]
    if args.clear_exclusions:
        excluded = []
    ranges = effective_ranges(approved, excluded, candidate["denseMapping"])
    changed = any((
        anchors != review["anchors"], candidate != review["candidate"],
        approved != review["requestedClipRanges"], excluded != review["excludedClipRanges"],
        args.split is not None and args.split != review["approval"].get("split"),
        getattr(args, "voice_supervision_policy", None) is not None
        and args.voice_supervision_policy != review["approval"].get("voiceSupervisionPolicy"),
    ))
    approval = {} if changed else dict(review["approval"])
    if args.split is not None:
        approval["split"] = args.split
    for option, field in CONFIRMATIONS.items():
        if getattr(args, option):
            approval[field] = True
    if args.confirm_percussion_completeness:
        approval["percussionAnnotationsComplete"] = True
    if getattr(args, "voice_supervision_policy", None) is not None:
        approval["voiceSupervisionPolicy"] = args.voice_supervision_policy
    if args.acknowledge_uncertainty:
        approval["uncertaintyAcknowledged"] = True
    if approval.get("approveExperimentalRangesAndSplit") and not ranges:
        raise ValueError("Experimental approval requires explicit nonempty --approve-range bounds.")
    approval.update(
        sourceGpSha256=state["inputs"]["sourceGpSha256"], audioSha256=state["audio"]["sha256"],
        candidateSha256=candidate_digest(candidate), approvedClipRanges=ranges, groupId=pair["groupId"],
    )
    review.update(
        preparationSha256=candidate_digest(state), candidateSha256=candidate_digest(candidate),
        anchors=anchors, candidate=candidate, requestedClipRanges=approved, excludedClipRanges=excluded, approval=approval,
    )
    editing = bool(args.anchor or args.clear_anchors or args.approve_range is not None or args.clear_ranges or args.exclude_range is not None or args.clear_exclusions or args.split or args.acknowledge_uncertainty or args.confirm_percussion_completeness or getattr(args, "voice_supervision_policy", None) is not None or any(getattr(args, option) for option in CONFIRMATIONS))
    if editing:
        if not args.reviewer or not args.reviewer.strip():
            raise ValueError("Recording a decision requires --reviewer with your reviewer identity.")
        review["reviewer"] = args.reviewer
        approval["reviewer"] = args.reviewer
    mapping = candidate["denseMapping"]
    rows = []
    for key, position in positions.items():
        nominal = position["referenceSeconds"]
        seconds = anchors.get(key)
        if seconds is None and mapping and mapping[0]["referenceSeconds"] <= nominal <= mapping[-1]["referenceSeconds"]:
            seconds = float(np.interp(nominal, [point["referenceSeconds"] for point in mapping], [point["clipSeconds"] for point in mapping]))
        rows.append({"anchor": key, **position, "clipSeconds": seconds})
    cues = []
    for key in args.cue or []:
        row = next((row for row in rows if row["anchor"] == key), None)
        if row is None or row["clipSeconds"] is None:
            raise ValueError("Cue is outside the candidate mapping; first provide an explicit anchor for that score position.")
        cue = render_cue(directory, state, key, row["clipSeconds"])
        cues.append(cue)
        if args.listen:
            if not hasattr(os, "startfile"):
                raise ValueError("--listen uses the Windows default player; open the printed WAV paths on other platforms.")
            os.startfile(cue["cuePath"])
    if args.listen and not args.cue:
        raise ValueError("--listen requires at least one --cue ordinal/first-attack/end.")
    load_current(workspace, pair)
    if editing and before != review:
        write_json(directory / "review.json", review)
    report = {
        "id": pair["id"], "groupId": pair["groupId"], "candidateMethod": candidate["method"],
        **({"title": pair["title"]} if "title" in pair else {}),
        "candidateDiagnostics": candidate.get("diagnostics", candidate.get("error")),
        "timingRisks": mapping_risks(mapping),
        "conditioning": labels["conditioning"], "counts": canonical_counts(labels), "barsAndFirstAttack": rows,
        "approvedClipRanges": ranges, "excludedClipRanges": excluded, "approval": approval, "cues": cues,
        "notationEvidence": [beat for beat in read_json(directory / "notation.json")["beats"] if beat["referenceOnly"] or beat["text"] or beat["deadStrings"]],
        "uncertainty": {key: labels["review"][key] for key in ("unresolvedGestures", "issues", "sourceIssues")},
        "notice": "Cues and anchors approve instants only. Explicit experimental ranges approve the selected interpolation; all other timing remains unapproved. Unknown labels stay masked.",
    }
    write_json(directory / "review-report.json", report)
    return report


def release_dataset(workspace, version, validation_groups, identifiers=None, *, reviewer=None, authorize_release=False):
    if authorize_release is not True or not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Release requires --reviewer and --authorize-release: explicitly authorize the entire selected scope and chosen validation groups.")
    safe_id(version)
    if isinstance(validation_groups, str) or not isinstance(validation_groups, Sequence):
        raise ValueError("Validation groups must be an explicit sequence of group IDs.")
    validation_groups = list(read_validation_groups({"validationGroups": list(validation_groups)}))
    entries, payloads, sources, review_hashes = [], {}, {}, {}
    counts = {"train": 0, "validation": 0}
    selected = selection(workspace, identifiers)
    if set(validation_groups) - {pair["groupId"] for pair in selected}:
        raise ValueError("Every validation group must be present in the explicitly selected pairs.")
    for pair in selected:
        directory, state = load_current(workspace, pair)
        review = load_review(directory, state)
        approval = deepcopy(review["approval"])
        split = "validation" if pair["groupId"] in validation_groups else "train"
        if any(approval.get(field) is not True for field in CONFIRMATIONS.values()) or approval.get("split") not in (None, split) or approval.get("groupId") != pair["groupId"]:
            raise ValueError(f"{pair['id']}: missing source/notation/use/group/range approval, or reviewed split differs from --validation-group.")
        labels, normalization, clock, _ = score_positions(directory)
        candidate = review["candidate"]
        validate_mapping(candidate["denseMapping"], clock, state["audio"]["sampleCount"] / state["audio"]["sampleRate"])
        uncertain = mapping_risks(candidate["denseMapping"]) or labels["review"]["unresolvedGestures"] or labels["review"]["issues"] or labels["review"]["sourceIssues"] or any(not all(note["labelMask"].values()) for note in labels["targets"]["notes"])
        if uncertain and approval.get("uncertaintyAcknowledged") is not True:
            raise ValueError(f"{pair['id']}: explicitly --acknowledge-uncertainty after reviewing the retained masks/issues, or quarantine the pair.")
        ranges = effective_ranges(review["requestedClipRanges"], review["excludedClipRanges"], candidate["denseMapping"])
        if ranges != approval.get("approvedClipRanges") or approval.get("candidateSha256") != candidate_digest(candidate) or approval.get("sourceGpSha256") != state["inputs"]["sourceGpSha256"] or approval.get("audioSha256") != state["audio"]["sha256"]:
            raise ValueError("Approval no longer matches this source, candidate or approved ranges.")
        downbeat = first_full_bar_downbeat(
            labels, candidate, read_json(directory / "notation.json")["normalizedPickup"],
        )
        rate = state["audio"]["sampleRate"]
        spans = range_sample_bounds(ranges, rate)
        notes, gestures = projected_targets(labels, candidate, clock)
        percussion_coverage = percussion_annotation_coverage(labels, candidate, normalization) if approval.get("percussionAnnotationsComplete") is True else None
        windows = [{
            "windowId": f"{pair['id']}:{start}-{stop}", "startSample": start, "stopSampleExclusive": stop,
            "targets": targets_in_window(notes, gestures, start, stop, rate, percussion_coverage=percussion_coverage),
        } for start, stop in sample_windows(spans, rate)]
        if not windows:
            raise ValueError(f"{pair['id']}: no approved windows of at least two seconds remain.")
        approval.update(groupId=pair["groupId"], split=split, releaseReviewer=reviewer)
        payload = {
            "schemaVersion": 1, "kind": "local-training-targets", "id": pair["id"],
            "canonical": labels, "normalization": normalization, "candidate": candidate, "windows": windows,
            "downbeatConditioning": downbeat,
            "approval": approval, "sourceAudioSha256": state["inputs"]["sourceAudioSha256"],
            "preparationSha256": candidate_digest(state),
            "preparationRuntime": state["inputs"]["runtime"], "preparationImplementation": state["inputs"]["implementation"],
        }
        entry = {
            "id": pair["id"], "groupId": pair["groupId"], "split": split,
            "audioPath": f"audio\\{pair['id']}.flac", "audioSha256": state["audio"]["sha256"],
            **{key: state["audio"][key] for key in ("sampleRate", "channels", "sampleCount")},
            "targetsPath": f"targets\\{pair['id']}.json",
        }
        entries.append(entry)
        payloads[pair["id"]], sources[pair["id"]] = payload, directory / TRIMMED_AUDIO_NAME
        review_hashes[directory / "review.json"] = sha256(directory / "review.json")
        counts[split] += len(windows)
    if not all(counts.values()):
        raise ValueError("A release requires nonempty train AND validation windows from different relationship groups.")
    authorization = {
        "schemaVersion": 1, "kind": "local-release-authorization", "reviewer": reviewer,
        "version": version, "validationGroups": validation_groups, "authorizedUse": True,
        "approveExperimentalRangesAndSplit": True, "groupingConfirmed": True,
        "distributionAuthorized": False,
        "selectedScope": release_scope([(entry, payloads[entry["id"]]) for entry in entries]),
    }
    authorization["sha256"] = candidate_digest(authorization)
    for payload in payloads.values():
        payload["approval"]["releaseAuthorizationSha256"] = authorization["sha256"]
    manifest = {
        "schemaVersion": 1, "kind": "local-training-dataset", "trainingReady": True,
        "visibility": "private", "distributionAuthorized": False,
        "entries": entries, "counts": {"windowsBySplit": counts}, "version": version,
        "validationGroups": validation_groups,
        "releaseAuthorization": authorization,
    }
    releases = regular_path(workspace / "releases")
    releases.mkdir(exist_ok=True)
    destination = regular_path(releases / version)
    staging = regular_path(releases / f".{version}-{uuid4().hex}.building")
    staging.mkdir()
    try:
        (staging / "audio").mkdir()
        (staging / "targets").mkdir()
        for entry in entries:
            identifier = entry["id"]
            shutil.copyfile(sources[identifier], staging / "audio" / f"{identifier}.flac")
            target = staging / "targets" / f"{identifier}.json"
            write_json(target, payloads[identifier])
            entry["targetsSha256"] = sha256(target)
        write_json(staging / "manifest.json", manifest)
        validate_release(staging / "manifest.json")
        if selection(workspace, identifiers) != selected:
            raise ValueError("Selected pairs changed during release.")
        for pair in selected:
            load_current(workspace, pair)
        if any(sha256(regular_path(path)) != digest for path, digest in review_hashes.items()):
            raise ValueError("Review changed during release.")
        if destination.exists():
            previous, _, _ = validate_release(destination / "manifest.json")
            if previous != manifest:
                raise ValueError("Release version already exists with different contents; choose a NEW version. Frozen releases are never overwritten.")
        else:
            staging.rename(destination)
        validate_release(destination / "manifest.json")
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"manifestPath": str(destination / "manifest.json"), "trainingReady": True, "counts": manifest["counts"], "releaseAuthorization": authorization}


def status(workspace):
    rows = []
    for pair in load_pairs(workspace)["pairs"]:
        identity = {"id": pair["id"], "groupId": pair["groupId"], **({"title": pair["title"]} if "title" in pair else {})}
        try:
            directory, state = load_current(workspace, pair)
            review = load_review(directory, state)
            approved = all(review["approval"].get(field) is True for field in CONFIRMATIONS.values())
            rows.append({**identity, "status": "reviewed" if approved else "needs-review"})
        except (OSError, ValueError) as error:
            rows.append({**identity, "status": "blocked", "reason": str(error)})
    return {"workspace": str(workspace), "pairs": rows}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workspace", type=Path, default=ROOT / "data", help="Private workspace root (default: data).")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create an empty, gitignored local workspace.")
    add = commands.add_parser("add", help="Import two supplied local files into an owned pair; no acquisition or trimming.")
    add.add_argument("--id", required=True)
    add.add_argument("--group", required=True, help="Stable related-song/arrangement/shared-recording group.")
    add.add_argument("--performer", help="Optional explicitly supplied stable performer ID.")
    add.add_argument("--title", help="Optional readable name; stable IDs remain unchanged.")
    add.add_argument("--gp", required=True, type=Path)
    add.add_argument("--audio", required=True, type=Path)
    prepare = commands.add_parser("prepare", help="Inspect, normalize, convert once and propose deterministic DSP timing.")
    prepare.add_argument("--ids", nargs="+")
    prepare.add_argument("--accept-conventions", action="store_true", help="Explicitly accept O=wrist thump, X=thumb slap, ghost X=percussive hit and documented repeat/text conventions for these sources.")
    prepare.add_argument("--ffmpeg-dir")
    invalidation = commands.add_parser("invalidate", help="Deliberately retire mutable approval; frozen releases stay unchanged.")
    invalidation.add_argument("--id", required=True)
    invalidation.add_argument("--reason", required=True)
    review = commands.add_parser("review", help="Show bars/first attack, audition WAV cues, anchor and approve bounded ranges.")
    review.add_argument("--id", required=True)
    review.add_argument("--reviewer", help="reviewer identity, required when recording decisions.")
    review.add_argument("--anchor", action="append", help="Performance ordinal=audio seconds; first-attack/end are also accepted.")
    review.add_argument("--clear-anchors", action="store_true")
    ranges = review.add_mutually_exclusive_group()
    ranges.add_argument("--approve-range", action="append", help="Explicit experimental audio START:END; replaces prior requested ranges.")
    ranges.add_argument("--clear-ranges", action="store_true")
    exclusions = review.add_mutually_exclusive_group()
    exclusions.add_argument("--exclude-range", action="append", help="Uncertain audio START:END, excluded with a 0.5-second guard; replaces prior exclusions.")
    exclusions.add_argument("--clear-exclusions", action="store_true")
    review.add_argument("--split", choices=("train", "validation"))
    for option in CONFIRMATIONS:
        review.add_argument("--" + option.replace("_", "-"), action="store_true")
    review.add_argument("--confirm-percussion-completeness", action="store_true", help="Confirm exhaustive intended O/X/ghost(X) percussion annotations for this source; negatives remain censored where notation/timing is unresolved. Does not grant other approvals.")
    review.add_argument("--voice-supervision-policy", choices=("native-multivoice", "intentional-single-voice", "flattened-or-unknown"),
                        help="Whether source GP voice indices are trusted. Unknown/flattened masks only voice loss.")
    review.add_argument("--acknowledge-uncertainty", action="store_true")
    review.add_argument("--cue", action="append", help="Generate plain/cued WAV for an ordinal, first-attack or end.")
    review.add_argument("--listen", action="store_true", help="Open requested cue WAVs in the Windows default player.")
    proposal = commands.add_parser("propose", help="Write a private dataset proposal without approving sources, releasing data or running training.")
    proposal.add_argument("--name", required=True)
    proposal.add_argument("--validation-ids", nargs="+", required=True, help="Validation recording IDs; their complete relationship groups stay together.")
    release = commands.add_parser("release", help="Freeze a private self-contained release without training.")
    release.add_argument("--version", required=True)
    release.add_argument("--validation-group", dest="validation_groups", action="append", required=True, help="Existing relationship group to hold out; repeat for multiple independent groups.")
    release.add_argument("--ids", nargs="+", help="Explicit subset; omit uncertain pairs rather than fabricating approval.")
    release.add_argument("--reviewer", help="reviewer authorizing this complete release scope and split.")
    release.add_argument("--authorize-release", action="store_true", help="Explicitly authorize the chosen validation groups and ALL selected pairs/ranges (all pairs unless --ids is supplied).")
    commands.add_parser("status", help="Report stale, blocked and reviewed pairs without running training.")
    for name in ("batch", "batch-status", "batch-review", "batch-release", "batch-train"):
        batch = commands.add_parser(name, help="Explicit preparation and joint training pipeline." if name == "batch-train" else "Repeatable GP/trimmed-video preparation; never starts training.")
        batch.add_argument("--manifest", required=True, type=Path)
        batch.add_argument("--output-directory", required=True, type=Path)
        if name == "batch-release":
            batch.add_argument("--reviewer", required=True)
        if name == "batch-train":
            batch.add_argument("--epochs", type=int, default=3, help="Joint audio/video training epochs; both branches learn together.")
            batch.add_argument("--max-steps", type=int)
            batch.add_argument("--max-hours", type=float, help="Optional training-only wall-clock limit, including setup/evaluation and checkpointing. Omit for no time limit. Preparation is separate.")
            batch.add_argument("--cpu-threads", type=int, help="PyTorch compute threads; defaults to 75 percent of logical CPUs, capped at32.")
            batch.add_argument("--video-dropout", type=float, default=.2, help="Probability of masking video on a paired training example for audio-only inference.")
            batch.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if name == "batch-review":
            batch.add_argument("--id", required=True)
            batch.add_argument("--reviewer")
            action = batch.add_mutually_exclusive_group(required=True)
            action.add_argument("--accept-score", action="store_true", help="Explicitly accept reviewed score/audio pairing, notation, grouping and supplied ranges.")
            action.add_argument("--accept-shots", action="store_true", help="Accept the inspected video shot boundaries.")
            action.add_argument("--alignment-offset", type=float, help="Explicit reviewed video start time for trimmed audio zero.")
            batch.add_argument("--range", dest="ranges", action="append", default=[], metavar="START:END")
            batch.add_argument("--anchor", action="append", default=[])
            batch.add_argument("--exclude-range", action="append", default=[])
            batch.add_argument("--acknowledge-uncertainty", action="store_true")
            batch.add_argument("--confirm-percussion-completeness", action="store_true")
            batch.add_argument("--add-cut", type=float, action="append", default=[])
    return result


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        if args.command.startswith("batch"):
            from .paired_preparation import batch_status, finalize_batch, review_batch, run_batch, train_batch

            if args.command == "batch":
                output = run_batch(args.manifest, args.output_directory)
            elif args.command == "batch-status":
                output = batch_status(args.manifest, args.output_directory)
            elif args.command == "batch-release":
                output = finalize_batch(args.manifest, args.output_directory, args.reviewer)
            elif args.command == "batch-train":
                output = train_batch(args.manifest, args.output_directory, epochs=args.epochs, max_steps=args.max_steps, max_hours=args.max_hours,
                                     cpu_threads=args.cpu_threads, video_dropout=args.video_dropout, device=args.device)
            else:
                flags = None
                if args.accept_shots:
                    flags = ["--accept-shots"]
                    for value in args.add_cut:
                        flags.extend(["--add-cut", str(value)])
                elif args.alignment_offset is not None:
                    flags = ["--alignment-offset", str(args.alignment_offset)]
                if args.add_cut and not args.accept_shots:
                    raise ValueError("--add-cut requires --accept-shots.")
                output = review_batch(
                    args.manifest, args.output_directory, args.id, reviewer=args.reviewer,
                    accept_score=args.accept_score, ranges=args.ranges, anchors=args.anchor,
                    exclude_ranges=args.exclude_range, acknowledge_uncertainty=args.acknowledge_uncertainty,
                    percussion_complete=args.confirm_percussion_completeness,
                    video_flags=flags,
                )
            print(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2))
            return 0
        workspace = workspace_path(args.workspace)
        if args.command == "init":
            output = {"pairsPath": str(initialize(workspace))}
        elif args.command == "add":
            output = add_pair(workspace, args.id, args.group, args.gp, args.audio, args.performer, title=args.title)
        elif args.command == "prepare":
            output = [prepare_pair(workspace, pair, accept_conventions=args.accept_conventions, ffmpeg_dir=args.ffmpeg_dir) for pair in selection(workspace, args.ids)]
        elif args.command == "invalidate":
            invalidate(workspace, selection(workspace, [args.id])[0], args.reason)
            output = {"id": args.id, "status": "invalidated", "frozenReleases": "unchanged"}
        elif args.command == "review":
            output = review_pair(workspace, selection(workspace, [args.id])[0], args)
        elif args.command == "propose":
            from .dataset_proposal import propose_dataset

            output = propose_dataset(
                workspace, args.name, validation_ids=args.validation_ids, progress=lambda message: print(message, flush=True),
            )
        elif args.command == "release":
            output = release_dataset(workspace, args.version, args.validation_groups, args.ids, reviewer=args.reviewer, authorize_release=args.authorize_release)
        else:
            output = status(workspace)
        print(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (OSError, ValueError, GpInspectionError, AcquisitionError, sf.LibsndfileError) as error:
        print(f"Local preparation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
