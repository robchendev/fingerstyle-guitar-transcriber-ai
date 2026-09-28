"""Load approved private windows and preserve canonical supervision."""

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import scipy
import soundfile
import torch
from torch.utils.data import Dataset, Sampler

from .canonical_events import fraction
from .dataset_io import ROOT, sha256
from .dataset_release import release_path, validate_release
from .transcriber_audio import (
    BASE_CONDITIONING_DIM, DOWNBEAT_CONDITIONING_DIM, HarnessError,
    audio_features, conditioning_features, read_audio_window,
)


PAIRED_TECHNIQUE_TARGET_POLICY = "native-positive-only-rasgueado-v1"
VOICE_SUPERVISION_POLICIES = (
    "native-multivoice", "intentional-single-voice", "flattened-or-unknown",
)


def _window_frame_times(window, sample_rate, feature_config):
    samples = ((window["stopSampleExclusive"] - window["startSample"]) * feature_config.sample_rate + sample_rate - 1) // sample_rate
    frames = (samples + feature_config.hop_length - 1) // feature_config.hop_length
    return np.arange(frames) * feature_config.hop_seconds


def _inside_intervals(times, intervals):
    inside = np.zeros(len(times), dtype=bool)
    for start, stop in intervals:
        inside |= (times >= start) & (times < stop)
    return inside


class EpochShuffleSampler(Sampler):
    def __init__(self, data_source, seed=17):
        self.data_source, self.seed, self.epoch = data_source, seed, 0
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise HarnessError("Sampler seed must be a nonnegative 63-bit integer.")

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0 or epoch + self.seed >= 2 ** 63:
            raise HarnessError("Invalid deterministic sampler epoch.")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self):
        return len(self.data_source)


def collate_windows(items):
    if not items:
        raise HarnessError("Cannot collate an empty batch.")
    lengths = torch.tensor([len(item["features"]) for item in items], dtype=torch.long)
    longest = int(lengths.max())

    def padded(values):
        output = values[0].new_zeros((len(values), longest, *values[0].shape[1:]))
        for index, value in enumerate(values):
            output[index, :len(value)] = value
        return output

    batch = {
        "features": padded([item["features"] for item in items]),
        "conditioning": padded([item["conditioning"] for item in items]),
        "lengths": lengths,
        "valid_frames": torch.arange(longest)[None, :] < lengths[:, None],
        "targets": {key: padded([item["targets"][key] for item in items]) for key in items[0]["targets"]},
        "masks": {key: padded([item["masks"][key] for item in items]) for key in items[0]["masks"]},
        "metadata": [item["metadata"] for item in items],
    }
    if any("video" in item for item in items):
        if not all("video" in item for item in items):
            raise HarnessError("Cannot mix indexed-video and audio-only dataset items in a batch.")
        from .paired_video import validate_video_tensors

        videos = [item["video"] for item in items]
        for item in items:
            validate_video_tensors(item["video"], len(item["features"]))
        dimensions = {video["structured"].shape[-1] for video in videos}
        if len(dimensions) != 1:
            raise HarnessError("Cannot mix paired-video schemas or feature dimensions in a batch.")
        longest_video = max(len(video["structured"]) for video in videos)
        batch["video"] = {}
        for key in videos[0]:
            extent = longest if key == "frame_indices" else longest_video
            values = [video[key] for video in videos]
            output = values[0].new_full((len(values), extent, *values[0].shape[1:]), -1 if key in ("frame_indices", "segment_id") else 0)
            for index, value in enumerate(values):
                output[index, :len(value)] = value
            batch["video"][key] = output
    return batch


def training_conditioning(record, clip_times, conditioning_dim=BASE_CONDITIONING_DIM):
    data, candidate = record["data"], record["candidate"]
    mapping = candidate["denseMapping"]
    audio = np.array([point["clipSeconds"] for point in mapping])
    quarters = np.array([point["scoreQuarter"] for point in mapping])
    lower = np.nextafter(np.nextafter(audio[0], -np.inf), -np.inf)
    upper = np.nextafter(np.nextafter(audio[-1], np.inf), np.inf)
    if np.any(clip_times < lower) or np.any(clip_times > upper) or np.any(np.diff(audio) < 0) or not np.isfinite(audio).all():
        raise HarnessError("Training frames extend outside the approved candidate mapping.")
    positions = np.interp(clip_times, audio, quarters)
    visits = data.labels["measureVisits"]
    starts = [float(fraction(visit["onsetQuarter"], "measure start")) for visit in visits]
    tempos = [{
        "position": starts[event["measureIndex"]] + float(fraction(event["offsetQuarter"], "tempo offset")),
        "bpm": event["bpm"], "beatUnit": event["beatUnit"], "linear": event["linear"],
    } for event in data.normalization["normalizedTempoEvents"]]
    timing = data.labels["conditioning"]["providedTiming"]
    meters = [{"position": 0., "timeSignature": timing["timeSignature"]}]
    meters.extend({"position": starts[event["measureIndex"]], "timeSignature": event["timeSignature"]} for event in timing["sourceTimeSignatureChanges"])
    instrument = data.labels["conditioning"]["instrument"]
    kwargs = {}
    if conditioning_dim == DOWNBEAT_CONDITIONING_DIM:
        downbeat = record.get("downbeatConditioning")
        if not isinstance(downbeat, dict):
            raise HarnessError("New training requires reviewed first-full-bar downbeat conditioning.")
        quarter = downbeat["scoreQuarter"]
        kwargs = {
            "first_full_bar_downbeat": downbeat["clipSeconds"],
            "score_quarters": positions,
            "first_full_bar_quarter": quarter[0] / quarter[1],
        }
    elif conditioning_dim != BASE_CONDITIONING_DIM:
        raise HarnessError("Unsupported conditioning dimension.")
    return conditioning_features(
        instrument["openStringMidi"], instrument["capoFret"], tempos, meters,
        positions, **kwargs,
    )


def encode_targets(window, canonical, frame_times, model_config, *, negative_onsets_allowed,
                   voice_supervision_policy="native-multivoice"):
    from .transcriber_model import HARMONIC_FRETS, HARMONIC_TYPES, PERCUSSION_TYPES
    from .technique_supervision import TECHNIQUE_DIRECTIONS, TECHNIQUE_TYPES
    from .connection_supervision import BEND_FIELDS, GRACE_MODES, RELATION_TYPES, vocabularies

    if voice_supervision_policy not in VOICE_SUPERVISION_POLICIES:
        raise HarnessError("Unsupported voice supervision policy.")
    count = len(frame_times)
    targets = {name: torch.zeros((count, 6), dtype=torch.long if name in {"fret", "pitch", "voice", "harmonic_kind", "harmonic_node"} else torch.float32) for name in ("note_onset", "fret", "pitch", "voice", "duration_log", "harmonic", "harmonic_kind", "harmonic_node")}
    targets["percussion"] = torch.zeros((count, len(PERCUSSION_TYPES)))
    architecture_version = getattr(model_config, "architecture_version", 1)
    connection_types, note_technique_types = vocabularies(architecture_version)
    if architecture_version >= 2:
        targets["technique"] = torch.zeros((count, len(TECHNIQUE_TYPES)))
        targets["technique_direction"] = torch.zeros((count, len(TECHNIQUE_TYPES)), dtype=torch.long)
        targets["technique_strings"] = torch.zeros((count, len(TECHNIQUE_TYPES), 6))
    if architecture_version >= 3:
        targets["connection"] = torch.zeros((count, 6), dtype=torch.long)
        targets["note_technique"] = torch.zeros((count, 6, len(note_technique_types)))
        targets["bend_curve"] = torch.zeros((count, 6, len(BEND_FIELDS)))
    if architecture_version == 4:
        targets["grace"] = torch.zeros((count, 6))
        for name in ("grace_fret", "grace_mode", "grace_transition"):
            targets[name] = torch.zeros((count, 6), dtype=torch.long)
    masks = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in targets.items()}
    hop = frame_times[1] - frame_times[0] if count > 1 else .02
    valid_interior = (frame_times >= .5) & (frame_times < frame_times[-1] - .5)
    masks["note_onset"][:] = torch.from_numpy(valid_interior[:, None] & np.array(negative_onsets_allowed)[None, :])
    negative_percussion = window["targets"].get("negativePercussionSupervision", False)
    coverage = window["targets"].get("percussionAnnotationCoverage", [])
    if type(negative_percussion) is not bool or not isinstance(coverage, list) or bool(coverage) != negative_percussion:
        raise HarnessError("Percussion negative labels require explicit nonempty annotation coverage.")
    covered = np.zeros(count, dtype=bool)
    previous_end = -1.
    for left, right in coverage:
        if not np.isfinite([left, right]).all() or not 0 <= left < right <= frame_times[-1] + hop + 1e-9 or left < previous_end:
            raise HarnessError("Invalid percussion annotation coverage.")
        covered |= (frame_times >= left) & (frame_times < right)
        previous_end = right
    masks["percussion"][:] = torch.from_numpy((valid_interior & covered)[:, None])
    source_notes = {note["id"]: note for note in canonical["targets"]["notes"]}
    source_gestures = {gesture["id"]: gesture for gesture in canonical["targets"]["gestures"]}
    events = {}
    radius = max(1, int(np.ceil(.05 / hop)))
    for note in window["targets"]["notes"]:
        if not note["supervisionMask"]["onset"]:
            continue
        onset = note["onsetWindowSeconds"]
        index = int(np.argmin(np.abs(frame_times - onset)))
        string = 6 - note["string"]
        if not 0 <= string < 6 or not 0 <= onset <= frame_times[-1] + hop:
            raise HarnessError("A supervised note lies outside its window or string range.")
        masks["note_onset"][max(0, index - radius):min(count, index + radius + 1), string] = False
        events.setdefault((index, string), []).append(note)
    collisions = 0
    for (index, string), notes in events.items():
        targets["note_onset"][index, string] = 1
        masks["note_onset"][index, string] = True
        if len(notes) != 1:
            collisions += 1
            continue
        note = notes[0]
        source = source_notes[note["sourceNoteId"]]
        if note["sourceLabelMask"] != source["labelMask"]:
            raise HarnessError("Window note masks differ from canonical supervision.")
        for field, source_field, limit in (("fret", "fret", model_config.max_fret + 1), ("pitch", "soundingPitchMidi", 128), ("voice", "voiceIndex", model_config.max_voices)):
            permitted = note["supervisionMask"]["fingering" if field == "fret" else "pitch" if field == "pitch" else "onset"]
            if field == "voice":
                permitted &= voice_supervision_policy != "flattened-or-unknown"
            value = note[source_field]
            if permitted:
                if type(value) is not int or not 0 <= value < limit:
                    raise HarnessError(f"Supervised {field} is outside model vocabulary.")
                targets[field][index, string] = value
                masks[field][index, string] = True
        if note["supervisionMask"]["notatedDuration"]:
            duration = float(fraction(note["notatedDurationQuarter"], "note duration"))
            if duration <= 0:
                raise HarnessError("A supervised duration must be positive.")
            targets["duration_log"][index, string] = np.log1p(duration)
            masks["duration_log"][index, string] = True
        harmonic = source["sourceSegments"][0].get("harmonic")
        if harmonic is not None:
            targets["harmonic"][index, string] = 1
            masks["harmonic"][index, string] = True
            if harmonic["type"] in HARMONIC_TYPES:
                targets["harmonic_kind"][index, string] = HARMONIC_TYPES.index(harmonic["type"])
                masks["harmonic_kind"][index, string] = True
            node = float(fraction(harmonic["fret"], "harmonic node"))
            if node in HARMONIC_FRETS:
                targets["harmonic_node"][index, string] = HARMONIC_FRETS.index(node)
                masks["harmonic_node"][index, string] = True
    if architecture_version >= 2:
        attack_frames = {index for index, _ in events}
        for index in attack_frames:
                masks["technique"][index] = True
        technique_events = window["targets"].get("techniques", [])
        if not isinstance(technique_events, list):
                raise HarnessError("Technique targets must be a list.")
        positives = []
        for event in technique_events:
                if not event["supervisionMask"]["onset"] or not event["supervisionMask"]["technique"]:
                    continue
                index = int(np.argmin(np.abs(frame_times - event["onsetWindowSeconds"])))
                if not 0 <= event["onsetWindowSeconds"] <= frame_times[-1] + hop:
                    raise HarnessError("A supervised technique lies outside its window.")
                for technique in event["techniques"]:
                    if technique not in TECHNIQUE_TYPES:
                        raise HarnessError(f"Unsupported supervised technique: {technique}")
                    axis = TECHNIQUE_TYPES.index(technique)
                    masks["technique"][max(0, index - radius):min(count, index + radius + 1), axis] = False
                    positives.append((index, axis, technique, event))
        for index, axis, technique, event in positives:
                targets["technique"][index, axis] = 1
                masks["technique"][index, axis] = True
                if event["supervisionMask"]["direction"].get(technique):
                    direction = event["directions"][technique]
                    if direction not in TECHNIQUE_DIRECTIONS:
                        raise HarnessError(f"Unsupported technique direction: {direction}")
                    targets["technique_direction"][index, axis] = TECHNIQUE_DIRECTIONS.index(direction)
                    masks["technique_direction"][index, axis] = True
                if event["supervisionMask"]["strings"]:
                    masks["technique_strings"][index, axis] = True
                    for string_number in event["stringsByTechnique"][technique]:
                        if type(string_number) is not int or not 1 <= string_number <= 6:
                            raise HarnessError("Technique membership requires physical strings one through six.")
                        targets["technique_strings"][index, axis, 6 - string_number] = 1
    if architecture_version >= 3:
        connection_events = window["targets"].get("connections", [])
        if not isinstance(connection_events, list):
            raise HarnessError("Connection targets must be a list.")
        event_slots = {}
        for event in connection_events:
            onset = event["onsetWindowSeconds"]
            if not np.isfinite(onset) or not 0 <= onset <= frame_times[-1] + hop or type(event["string"]) is not int or not 1 <= event["string"] <= 6:
                raise HarnessError("A connection target lies outside its window or string range.")
            index = int(np.argmin(np.abs(frame_times - event["onsetWindowSeconds"])))
            axis = 6 - event["string"]
            event_slots.setdefault((index, axis), []).append(event)
        for (index, axis), connection_events_at_slot in event_slots.items():
            event = connection_events_at_slot[0]
            if architecture_version == 4:
                if any(item.get("techniqueSchemaVersion") != 4 for item in connection_events_at_slot):
                    raise HarnessError("Architecture v4 requires separately versioned connection supervision.")
                attacks = events.get((index, axis), [])
                if len(connection_events_at_slot) != 1 or len(attacks) != 1 or attacks[0]["sourceNoteId"] != event["sourceNoteId"]:
                    continue
            connection_known = event["supervisionMask"]["connection"]
            if architecture_version == 4 and connection_known and event["connection"] != "none":
                origin = event["origin"]
                origin_onset = origin["proposedOnsetClipSeconds"] - event["proposedOnsetClipSeconds"] + event["onsetWindowSeconds"]
                origin_index = int(np.argmin(np.abs(frame_times - origin_onset)))
                origin_attacks = events.get((origin_index, axis), [])
                connection_known = len(origin_attacks) == 1 and origin_attacks[0]["sourceNoteId"] == origin["sourceNoteId"]
            if connection_known:
                targets["connection"][index, axis] = connection_types.index(event["connection"])
                masks["connection"][index, axis] = True
            if event["supervisionMask"]["techniques"]:
                masks["note_technique"][index, axis] = torch.tensor([
                    event.get("techniqueMasks", {}).get(name, architecture_version < 4)
                    for name in note_technique_types
                ])
                for name, present in event["techniques"].items():
                    targets["note_technique"][index, axis, note_technique_types.index(name)] = int(present)
            if event["supervisionMask"]["bendCurve"]:
                targets["bend_curve"][index, axis] = torch.tensor(event["bendCurve"], dtype=torch.float32) / 100
                masks["bend_curve"][index, axis] = True
            if architecture_version == 4:
                grace = event["grace"]
                targets["grace"][index, axis] = int(grace is not None)
                masks["grace"][index, axis] = event["supervisionMask"]["grace"]
                if grace is not None:
                    for name, value in (
                        ("fret", grace["sourceFret"]), ("mode", grace["mode"]),
                        ("transition", grace["transition"]),
                    ):
                        if not event["supervisionMask"]["graceAttributes"][name]:
                            continue
                        if name == "fret":
                            if type(value) is not int or not 0 <= value <= model_config.max_fret:
                                raise HarnessError("Grace source fret is outside the model vocabulary.")
                        else:
                            value = (GRACE_MODES if name == "mode" else RELATION_TYPES).index(value)
                        targets[f"grace_{name}"][index, axis] = value
                        masks[f"grace_{name}"][index, axis] = True
    percussion_events = []
    for gesture in window["targets"]["gestures"]:
        source = source_gestures[gesture["sourceGestureId"]]
        if gesture["technique"] != source["technique"]:
            raise HarnessError("Window gesture differs from its canonical source.")
        if gesture["technique"] not in PERCUSSION_TYPES or not gesture["supervisionMask"]["gesture"] or not gesture["supervisionMask"]["onset"]:
            continue
        index = int(np.argmin(np.abs(frame_times - gesture["onsetWindowSeconds"])))
        category = PERCUSSION_TYPES.index(gesture["technique"])
        masks["percussion"][max(0, index - radius):min(count, index + radius + 1), category] = False
        percussion_events.append((index, category))
    for index, category in percussion_events:
        targets["percussion"][index, category] = 1
        masks["percussion"][index, category] = True
    return targets, masks, collisions


def negative_onset_coverage(labels):
    allowed = [True] * 6
    for note in labels["targets"]["notes"]:
        if note["isAttack"] is not True or not note["labelMask"]["attack"] or note["sourceSegments"][0]["graceMode"] is not None:
            allowed[6 - note["string"]] = False
    return allowed


class TrainingDataset(Dataset):
    @staticmethod
    def _stat(path):
        value = path.stat()
        return value.st_size, value.st_mtime_ns

    def _check_release_unchanged(self):
        for path, original in self._guards.items():
            if self._stat(path) != original:
                raise HarnessError(f"A bound dataset file changed during use; stop and revalidate the release: {path}")

    def check_unchanged(self):
        self._check_release_unchanged()
        if self.video_index is not None:
            self.video_index.check_unchanged()

    def video_window(self, record, absolute_audio_times):
        if self.video_index is None:
            return None
        return self.video_index.window(record["row"]["id"], absolute_audio_times)

    def video_training_intervals(self, record):
        if not getattr(self, "paired_only", False):
            return None
        bundle = self.video_index.bundles.get(record["row"]["id"])
        return bundle.clip_intervals if bundle is not None else []

    def __len__(self):
        return len(self.windows)

    def _features(self, record, window):
        row, data = record["row"], record["data"]
        key = hashlib.sha256(json.dumps({"audio": row["audioSha256"], "start": window["startSample"], "stop": window["stopSampleExclusive"], "config": asdict(self.feature_config), "featureImplementation": sha256(Path(__file__).with_name("transcriber_audio.py")), "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": soundfile.__version__}, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.npz" if self.cache_dir is not None else None
        if path is not None and (path.resolve() != path.absolute() or path.is_symlink() or path.exists() and path.stat().st_nlink > 1):
            raise HarnessError("Feature cache files must not alias other assets.")
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as cached:
                features = np.array(cached["features"], copy=True)
                times = np.array(cached["times"], copy=True)
            expected_times = _window_frame_times(window, row["sampleRate"], self.feature_config)
            expected_frames = len(expected_times)
            if features.dtype != np.float32 or features.ndim != 2 or features.shape != (expected_frames, self.feature_config.n_mels) or times.shape != (expected_frames,) or not np.isfinite(features).all() or not np.allclose(times, expected_times):
                raise HarnessError("Corrupt feature cache; remove it explicitly before retrying.")
            return torch.from_numpy(features), times
        samples, rate = read_audio_window(data.audio_path, window["startSample"], window["stopSampleExclusive"], sample_rate=row["sampleRate"], channels=row["channels"], sample_count=data.entry["audioAsset"]["sampleCount"])
        features, times = audio_features(samples, rate, self.feature_config)
        if path is not None:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, prefix=".features-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(stream, features=features.numpy(), times=times)
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return features, times

    def __getitem__(self, index):
        self._check_release_unchanged()
        record, window = self.windows[index]
        features, times = self._features(record, window)
        clip_times = times + window["startSample"] / record["row"]["sampleRate"]
        conditioning = training_conditioning(
            record, clip_times, self.model_config.conditioning_dim,
        )
        local_window = window
        if self.model_config.architecture_version >= 2:
            from .technique_supervision import techniques_in_window

            local_window = deepcopy(window)
            local_window["targets"]["techniques"] = techniques_in_window(
                record["techniques"], window["startSample"], window["stopSampleExclusive"], record["row"]["sampleRate"],
            )
        if self.model_config.architecture_version >= 3:
            from .connection_supervision import connections_in_window

            local_window = deepcopy(local_window)
            local_window["targets"]["connections"] = connections_in_window(
                record["connections"], window["startSample"], window["stopSampleExclusive"], record["row"]["sampleRate"],
            )
        targets, masks, collisions = encode_targets(
            local_window, record["data"].labels, times, self.model_config,
            negative_onsets_allowed=record["negativeAllowed"],
            voice_supervision_policy=record["voiceSupervisionPolicy"],
        )
        intervals = self.video_training_intervals(record)
        if intervals is not None:
            prepared = torch.from_numpy(_inside_intervals(clip_times, intervals))
            for mask in masks.values():
                mask &= prepared.reshape(-1, *([1] * (mask.ndim - 1)))
        item = {
            "features": features, "conditioning": conditioning, "targets": targets, "masks": masks,
            "metadata": {
                "windowId": window["windowId"], "stringFrameCollisionsMasked": collisions,
                "voiceSupervisionPolicy": record["voiceSupervisionPolicy"],
            },
        }
        if self.video_index is not None:
            item["video"] = self.video_window(record, clip_times)
        if self.paired_technique_target_policy == PAIRED_TECHNIQUE_TARGET_POLICY:
            item["metadata"]["pairedTechniqueTargetPolicy"] = PAIRED_TECHNIQUE_TARGET_POLICY
            if "technique" in masks:
                from .technique_supervision import TECHNIQUE_TYPES

                # Missing native GP flags do not exclude physical compound-brush a-m-i.
                axis = TECHNIQUE_TYPES.index("rasgueado")
                masks["technique"][:, axis] &= targets["technique"][:, axis] > 0
        return item

    def __init__(self, manifest_path, split, feature_config, model_config, *, root=ROOT, cache_dir=None, video_index_path=None):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path).absolute()
        if not self.manifest_path.is_relative_to(self.root):
            raise HarnessError("The release manifest must be inside its declared private data root.")
        if split not in ("train", "validation"):
            raise HarnessError("Training releases contain train and validation splits only.")
        if (
            feature_config.n_mels != model_config.n_mels
            or model_config.conditioning_dim not in (BASE_CONDITIONING_DIM, DOWNBEAT_CONDITIONING_DIM)
        ):
            raise HarnessError("Feature/model dimensions disagree.")
        self.feature_config, self.model_config = feature_config, model_config
        manifest, records, bindings = validate_release(self.manifest_path)
        self.manifest_sha256 = bindings[self.manifest_path]
        self.video_index = None
        self.video_identity = None
        self.paired_technique_target_policy = PAIRED_TECHNIQUE_TARGET_POLICY if video_index_path is not None else None
        if video_index_path is not None:
            from .paired_video import PairedVideoIndex

            self.video_index = PairedVideoIndex(video_index_path, self.manifest_sha256, records, root=self.root)
            self.video_identity = {**self.video_index.identity, "targetPolicy": PAIRED_TECHNIQUE_TARGET_POLICY}
        self._guards = {path: self._stat(path) for path in bindings}
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.records, self.windows = [], []
        for row, payload in records:
            if row["split"] != split:
                continue
            labels = payload["canonical"]
            data = SimpleNamespace(
                labels=labels, normalization=payload["normalization"],
                audio_path=release_path(self.manifest_path.parent, row["audioPath"], "audio"),
                entry={"audioAsset": {"sampleCount": row["sampleCount"]}},
            )
            record = {
                "data": data, "candidate": payload["candidate"], "row": row, "negativeAllowed": negative_onset_coverage(labels),
                "percussionAnnotationsComplete": payload["approval"].get("percussionAnnotationsComplete") is True,
                "voiceSupervisionPolicy": payload["approval"].get("voiceSupervisionPolicy", "native-multivoice"),
            }
            if "downbeatConditioning" in payload:
                record["downbeatConditioning"] = payload["downbeatConditioning"]
            elif model_config.conditioning_dim == DOWNBEAT_CONDITIONING_DIM:
                raise HarnessError("The release predates required first-full-bar downbeat conditioning; rebuild it.")
            if model_config.architecture_version >= 2:
                from .score_alignment import ScoreClock
                from .technique_supervision import projected_techniques

                record["techniques"] = projected_techniques(labels, payload["candidate"], ScoreClock(labels, payload["normalization"]))
                record["techniqueAnnotationsComplete"] = True
            if model_config.architecture_version >= 3:
                from .connection_supervision import projected_connections

                record["connections"] = projected_connections(
                    labels, payload["candidate"], ScoreClock(labels, payload["normalization"]),
                    architecture_version=model_config.architecture_version,
                )
            self.records.append(record)
            self.windows.extend((record, window) for window in payload["windows"])
        if len(self.windows) != manifest["counts"]["windowsBySplit"][split]:
            raise HarnessError("Training release window count changed during loading.")
        # Prepared footage remains paired through tracking failures. Availability
        # masks govern evidence, not whether genuine audio/GP supervision exists.
        self.video_paired_window_indices = [
            index for index, (record, window) in enumerate(self.windows)
            if self.video_index is not None and record["row"]["id"] in self.video_index.bundles
            and any(
                max(start, window["startSample"] / record["row"]["sampleRate"])
                < min(stop, window["stopSampleExclusive"] / record["row"]["sampleRate"])
                for start, stop in self.video_index.bundles[record["row"]["id"]].clip_intervals
            )
        ]
        self.video_paired_coverage = {
            "preparedAudioFrames": 0, "audioFramesWithObservedVideo": 0,
            "audioFramesWithUsableVideo": 0, "audioFramesWithUsablePluckingVideo": 0,
            "audioFramesWithGeometry": 0, "audioFramesWithIndependentHand": 0,
            "audioFramesWithCoarseContext": 0,
            "audioFramesWithUsableUnassignedVideo": 0,
            "windowsWithUsableVideo": 0, "trackingGapOnlyWindows": 0,
        }
        for index in self.video_paired_window_indices:
            record, window = self.windows[index]
            rate = record["row"]["sampleRate"]
            times = _window_frame_times(window, rate, feature_config) + window["startSample"] / rate
            bundle = self.video_index.bundles[record["row"]["id"]]
            prepared = _inside_intervals(times, bundle.clip_intervals)
            available, techniques = bundle.availability_at(times)
            observed = available.any(-1) & prepared
            plucking = available[:, 1] & techniques & prepared
            unassigned = available[:, 2:].any(-1) & techniques & prepared
            usable = (available[:, 0] | plucking | unassigned) & prepared
            geometry, independent, coarse = bundle.feature_availability_at(times)
            self.video_paired_coverage["preparedAudioFrames"] += int(prepared.sum())
            self.video_paired_coverage["audioFramesWithObservedVideo"] += int(observed.sum())
            self.video_paired_coverage["audioFramesWithUsableVideo"] += int(usable.sum())
            self.video_paired_coverage["audioFramesWithUsablePluckingVideo"] += int(plucking.sum())
            self.video_paired_coverage["audioFramesWithUsableUnassignedVideo"] += int(unassigned.sum())
            self.video_paired_coverage["audioFramesWithGeometry"] += int((geometry.any(-1) & prepared).sum())
            self.video_paired_coverage["audioFramesWithIndependentHand"] += int((independent.any(-1) & prepared).sum())
            self.video_paired_coverage["audioFramesWithCoarseContext"] += int((coarse.any(-1) & prepared).sum())
            self.video_paired_coverage["windowsWithUsableVideo"] += int(usable.any())
            self.video_paired_coverage["trackingGapOnlyWindows"] += int(not observed.any())
