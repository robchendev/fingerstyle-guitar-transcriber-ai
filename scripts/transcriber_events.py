"""Stitch window predictions and score decoded events only in approved coverage."""

from collections import defaultdict
import math

import numpy as np
import torch

from .score_alignment import ScoreClock
from .training_windows import projected_targets
from .transcriber_audio import BASE_CONDITIONING_DIM, HarnessError
from .transcriber_data import PAIRED_TECHNIQUE_TARGET_POLICY, training_conditioning
from .connection_supervision import RELATION_TYPES, SUPERVISION_VERSION, V4_NOTE_TECHNIQUE_TYPES


class OutputTimeline:
    def __init__(self, times):
        self.times = np.asarray(times, dtype=np.float64)
        if self.times.ndim != 1 or not len(self.times) or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise HarnessError("Prediction timeline requires finite, increasing frame times.")
        self.values = {}
        self.weights = torch.zeros(len(self.times))

    def add(self, outputs, local_times, stop_seconds):
        local_times = np.asarray(local_times, dtype=np.float64)
        if local_times.ndim != 1 or not len(local_times) or not np.isfinite(local_times).all() or np.any(np.diff(local_times) <= 0) or not math.isfinite(stop_seconds) or stop_seconds <= local_times[-1]:
            raise HarnessError("Invalid prediction window timeline.")
        if self.values and set(outputs) != set(self.values):
            raise HarnessError("Prediction heads changed between windows.")
        indices = np.flatnonzero((self.times >= local_times[0] - 1e-9) & (self.times < stop_seconds - 1e-9))
        if not len(indices):
            raise HarnessError("Prediction window does not cover a timeline frame.")
        positions = self.times[indices]
        right = np.searchsorted(local_times, positions, side="right").clip(0, len(local_times) - 1)
        left = (right - 1).clip(0)
        widths = local_times[right] - local_times[left]
        blend = np.divide(positions - local_times[left], widths, out=np.zeros_like(positions), where=widths > 0).clip(0, 1)
        blend = torch.tensor(blend, dtype=torch.float32)
        window_weight = torch.hann_window(max(len(local_times), 2), periodic=False)[:len(local_times)].clamp_min(.05)
        weights = window_weight[left] * (1 - blend) + window_weight[right] * blend
        for name, output in outputs.items():
            value = output.detach().cpu()
            if value.shape[0] != len(local_times) or not value.is_floating_point() or not torch.isfinite(value).all():
                raise HarnessError("Nonfinite or inconsistent window predictions.")
            if name not in self.values:
                self.values[name] = torch.zeros((len(self.times), *value.shape[1:]), dtype=value.dtype)
            if self.values[name].shape[1:] != value.shape[1:] or self.values[name].dtype != value.dtype:
                raise HarnessError("Prediction head dimensions changed between windows.")
            shape = (-1, *([1] * (value.ndim - 1)))
            mixed = value[left] * (1 - blend.view(shape)) + value[right] * blend.view(shape)
            self.values[name][indices] += mixed * weights.view(shape)
        self.weights[indices] += weights

    def finish(self):
        if not self.values or torch.any(self.weights <= 0):
            raise HarnessError("Predicted timeline has uncovered frames.")
        return {name: values / self.weights.view(-1, *([1] * (values.ndim - 1))) for name, values in self.values.items()}


def merge_intervals(intervals):
    result = []
    for left, right in sorted(intervals):
        if not math.isfinite(left) or not math.isfinite(right) or left >= right:
            raise HarnessError("Scoring intervals must be finite and nonempty.")
        if result and left <= result[-1][1] + 1e-9:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


def inside(time, intervals):
    return any(left <= time < right for left, right in intervals)


def match_events(truth, predictions, tolerance, keys):
    """Maximum-cardinality ordered matching within each pitch/string/class key."""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise HarnessError("Event tolerance must be finite and positive.")
    expected, actual = defaultdict(list), defaultdict(list)
    for index, event in enumerate(truth):
        if not math.isfinite(event["onsetSeconds"]) or event["onsetSeconds"] < 0:
            raise HarnessError("Reference event times must be finite and nonnegative.")
        expected[tuple(event[key] for key in keys)].append((event["onsetSeconds"], index))
    for index, event in enumerate(predictions):
        if not math.isfinite(event["onsetSeconds"]) or event["onsetSeconds"] < 0:
            raise HarnessError("Predicted event times must be finite and nonnegative.")
        actual[tuple(event[key] for key in keys)].append((event["onsetSeconds"], index))
    matches = []
    for key, targets in expected.items():
        targets, outputs = sorted(targets), sorted(actual[key])
        a = b = 0
        while a < len(targets) and b < len(outputs):
            delta = outputs[b][0] - targets[a][0]
            if delta < -tolerance - 1e-9:
                b += 1
            elif delta > tolerance + 1e-9:
                a += 1
            else:
                matches.append((targets[a][1], outputs[b][1]))
                a += 1
                b += 1
    return matches


def event_counts(truth, predictions, tolerance, keys, negative_known, *, has_negative_coverage, matches=None):
    if matches is None:
        matches = match_events(truth, predictions, tolerance, keys)
    matched_predictions = {prediction for _, prediction in matches}
    false_positive = sum(index not in matched_predictions and negative_known(event) for index, event in enumerate(predictions))
    return {
        "true_positive": len(matches), "false_positive": false_positive, "false_negative": len(truth) - len(matches),
        "reference_events": len(truth), "predicted_events": len(predictions),
        "unscorable_predictions": len(predictions) - len(matches) - false_positive,
        "absolute_onset_error_sum": sum(abs(truth[a]["onsetSeconds"] - predictions[b]["onsetSeconds"]) for a, b in matches),
        "has_negative_coverage": has_negative_coverage,
    }


def summarize_counts(counts):
    tp, fp, fn = counts["true_positive"], counts["false_positive"], counts["false_negative"]
    complete = counts["has_negative_coverage"]
    return {
        **counts,
        "precision": tp / (tp + fp) if complete and tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if complete and 2 * tp + fp + fn else None,
        "mean_absolute_onset_error_seconds": counts["absolute_onset_error_sum"] / tp if tp else None,
    }


def checkpoint_event_score(report):
    metrics = report["metricsByToleranceSeconds"]["0.1"]
    selected = [metrics[name] for name in ("string_fret_pitch_onset", "wrist_thump", "thumb_slap", "percussive_hit") if metrics[name]["has_negative_coverage"]]
    technique_metric = None
    if "technique_string_set" in metrics and metrics["technique_string_set"]["has_negative_coverage"]:
        technique_metric = metrics["technique_string_set"]
        selected.append(technique_metric)
    tp = sum(value["true_positive"] for value in selected)
    fp = sum(value["false_positive"] for value in selected)
    fn = sum(value["false_negative"] for value in selected)
    if not selected or not sum(value["reference_events"] for value in selected):
        raise HarnessError("Decoded checkpoint selection requires scorable validation events, not only unknown or empty labels.")
    score = 2 * tp / (2 * tp + fp + fn)
    if report.get("techniqueSupervisionVersion") == SUPERVISION_VERSION:
        names = [
            *(f"relation_{name}" for name in RELATION_TYPES[1:]),
            *(f"note_technique_{name}" for name in V4_NOTE_TECHNIQUE_TYPES),
            "grace_attributes",
        ]
        observed = [metrics[name] for name in names if name in metrics and metrics[name]["reference_events"] and metrics[name]["f1"] is not None]
        if not observed:
            raise HarnessError("Architecture v4 checkpoint selection requires covered non-none technique/anchored-grace validation events.")
        technique_score = sum(value["f1"] for value in observed) / len(observed)
        return {
            "score": 2 * score * technique_score / (score + technique_score) if score + technique_score else 0.,
            "metric": "harmonic-mean-base-event-micro-f1-and-non-none-technique-macro-f1@100ms-v4",
            "baseEventScore": score, "nonNoneTechniqueScore": technique_score,
            "techniqueClasses": len(observed), "windows": report["windowVisits"],
        }
    return {
        "score": score,
        "metric": (
            "joint-note-covered-percussion-and-technique-string-set-micro-f1@100ms"
            if technique_metric else "joint-note-and-covered-percussion-micro-f1@100ms"
        ),
        "windows": report["windowVisits"],
    }


def _relation_matches(truth, predictions, tolerance):
    keys = ("string", "voiceIndex", "soundingPitchMidi", "connection")
    grouped = defaultdict(list)
    for index, event in enumerate(predictions):
        grouped[tuple(event[key] for key in keys)].append(index)
    candidates = {}
    for index, event in enumerate(truth):
        origin = event["connectionOrigin"]
        candidates[index] = [
            other for other in grouped[tuple(event[key] for key in keys)]
            if abs(event["onsetSeconds"] - predictions[other]["onsetSeconds"]) <= tolerance + 1e-9
            and predictions[other].get("connectionOrigin") is not None
            and all(origin[key] == predictions[other]["connectionOrigin"].get(key) for key in ("string", "voiceIndex", "soundingPitchMidi"))
            and abs(origin["onsetSeconds"] - predictions[other]["connectionOrigin"]["onsetSeconds"]) <= tolerance + 1e-9
        ]
    assigned = {}

    def assign(index, seen):
        for other in candidates[index]:
            if other in seen:
                continue
            seen.add(other)
            if other not in assigned or assign(assigned[other], seen):
                assigned[other] = index
                return True
        return False

    for index in candidates:
        assign(index, set())
    return [(index, other) for other, index in assigned.items()]


def score_note_attributes(connections, predictions, scoring, tolerance, *, note_technique_threshold=.5, unknown_notes=(), negative_onsets_allowed=None):
    """Pitch-correct anchor metrics; relation matches require both endpoints."""
    if not math.isfinite(note_technique_threshold) or not 0 <= note_technique_threshold <= 1:
        raise HarnessError("Note-technique threshold must be between zero and one.")
    negative_onsets_allowed = [False] * 6 if negative_onsets_allowed is None else negative_onsets_allowed
    if len(negative_onsets_allowed) != 6 or any(type(value) is not bool for value in negative_onsets_allowed):
        raise HarnessError("Technique scoring requires six explicit onset-coverage masks.")
    anchors = [
        {**event, "onsetSeconds": event["proposedOnsetClipSeconds"]}
        for event in connections if event["proposedOnsetClipSeconds"] is not None
    ]
    pitch_keys = ("string", "voiceIndex", "soundingPitchMidi")
    result = {}

    def coverage(known):
        unknown = [event for event in anchors if not known(event)] + list(unknown_notes)
        resolved = [event for event in anchors if known(event) and inside(event["onsetSeconds"], scoring)]

        def negative(event):
            return inside(event["onsetSeconds"], scoring) and (negative_onsets_allowed[6 - event["string"]] or any(
                other["string"] == event["string"] and abs(other["onsetSeconds"] - event["onsetSeconds"]) <= tolerance
                for other in resolved
            )) and not any(
                other["string"] == event["string"] and abs(other["onsetSeconds"] - event["onsetSeconds"]) <= tolerance
                for other in unknown
            )

        return resolved, negative

    known, negative = coverage(lambda event: event["connectionMask"] and (
        event["connection"] == "none" or event.get("origin") is not None
        and event["origin"]["proposedOnsetClipSeconds"] is not None
        and inside(event["origin"]["proposedOnsetClipSeconds"], scoring)
    ))
    truth = [{
        **event, "connectionOrigin": {
            **event["origin"], "onsetSeconds": event["origin"]["proposedOnsetClipSeconds"],
        },
    } for event in known if event["connection"] != "none"]
    guessed = [event for event in predictions if event.get("connection", "none") != "none"]
    for name in (None, *RELATION_TYPES[1:]):
        expected = [event for event in truth if name is None or event["connection"] == name]
        actual = [event for event in guessed if name is None or event["connection"] == name]
        result["relation_joint" if name is None else f"relation_{name}"] = event_counts(
            expected, actual, tolerance, (*pitch_keys, "connection"), negative,
            has_negative_coverage=bool(known) or any(negative_onsets_allowed), matches=_relation_matches(expected, actual, tolerance),
        )
    pooled = []
    for name in V4_NOTE_TECHNIQUE_TYPES:
        known, negative = coverage(lambda event: event["techniqueMasks"][name])
        expected = [event for event in known if event["techniques"][name]]
        actual = [event for event in predictions if event.get("noteTechniques", {}).get(name, 0) >= note_technique_threshold]
        counts = event_counts(expected, actual, tolerance, pitch_keys, negative, has_negative_coverage=bool(known) or any(negative_onsets_allowed))
        result[f"note_technique_{name}"] = counts
        pooled.append(counts)
    result["note_technique_joint"] = {
        key: any(value[key] for value in pooled) if key == "has_negative_coverage" else sum(value[key] for value in pooled)
        for key in pooled[0]
    }
    for name, attributes, masks in (
        ("grace_presence", (), ()),
        ("grace_fret", ("sourceFret", "sourcePitchMidi", "intervalSemitones"), ("fret",)),
        ("grace_mode", ("mode",), ("mode",)),
        ("grace_transition", ("transition",), ("transition",)),
        ("grace_attributes", ("sourceFret", "sourcePitchMidi", "intervalSemitones", "mode", "transition"), ("fret", "mode", "transition")),
    ):
        known, negative = coverage(lambda event: event["graceMask"] and (
            event["grace"] is None or all(event["graceAttributeMasks"][key] for key in masks)
        ))
        expected = [{**event, **{key: event["grace"][key] for key in attributes}} for event in known if event["grace"] is not None]
        actual = [{**event, **{key: event["grace"].get(key) for key in attributes}} for event in predictions if event.get("grace") is not None]
        result[name] = event_counts(expected, actual, tolerance, (*pitch_keys, *attributes), negative, has_negative_coverage=bool(known) or any(negative_onsets_allowed))
    return result


def _spans(windows):
    groups = []
    for window in sorted(windows, key=lambda item: (item["startSample"], item["stopSampleExclusive"])):
        if groups and window["startSample"] <= groups[-1]["stop"]:
            groups[-1]["windows"].append(window)
            groups[-1]["stop"] = max(groups[-1]["stop"], window["stopSampleExclusive"])
        else:
            groups.append({"start": window["startSample"], "stop": window["stopSampleExclusive"], "windows": [window]})
    return groups


@torch.no_grad()
def evaluate_events(model, dataset, device, *, tolerances=(.05, .1, .2), onset_threshold=.5, percussion_threshold=.5, technique_threshold=.5,
                    connection_threshold=.5, note_technique_threshold=.5, grace_threshold=.5, progress=None):
    from .transcriber_model import PERCUSSION_TYPES, decode_events
    from .technique_supervision import TECHNIQUE_TYPES

    tolerances = tuple(tolerances)
    if not tolerances or len(set(tolerances)) != len(tolerances) or any(not math.isfinite(value) or value <= 0 for value in tolerances):
        raise HarnessError("Supply unique positive event tolerances in seconds.")
    config = dataset.feature_config
    conditioning_dim = getattr(
        getattr(dataset, "model_config", None), "conditioning_dim", BASE_CONDITIONING_DIM,
    )
    records = []
    totals = {}
    video_identity = getattr(dataset, "video_identity", None)
    positive_only_rasgueado = (
        getattr(dataset, "paired_technique_target_policy", None) == PAIRED_TECHNIQUE_TARGET_POLICY
        or isinstance(video_identity, dict) and video_identity.get("targetPolicy") == PAIRED_TECHNIQUE_TARGET_POLICY
    )
    version4 = getattr(getattr(model, "config", None), "architecture_version", None) == 4
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for record in dataset.records:
            row, data = record["row"], record["data"]
            windows = [window for candidate_record, window in dataset.windows if candidate_record is record]
            rate = row["sampleRate"]
            scoring = merge_intervals([
                [window["startSample"] / rate + .5, window["stopSampleExclusive"] / rate - .5 - config.hop_seconds]
                for window in windows if (window["stopSampleExclusive"] - window["startSample"]) / rate > 1 + config.hop_seconds
            ])
            intervals = dataset.video_training_intervals(record) if callable(getattr(dataset, "video_training_intervals", None)) else None
            if intervals is not None:
                scoring = merge_intervals([
                    [max(left, start), min(right, stop)]
                    for left, right in scoring for start, stop in intervals
                    if max(left, start) < min(right, stop)
                ])
            if not scoring:
                raise HarnessError("No interior event-scoring coverage remains.")
            percussion_coverage = []
            for window in windows:
                if window["targets"].get("negativePercussionSupervision") is True:
                    origin = window["startSample"] / rate
                    percussion_coverage.extend([[origin + left, origin + right] for left, right in window["targets"]["percussionAnnotationCoverage"]])
            percussion_coverage = merge_intervals([
                [max(left, a), min(right, b)] for left, right in percussion_coverage for a, b in scoring
                if max(left, a) < min(right, b)
            ])
            predictions = {"notes": [], "percussion": [], "techniques": []}
            spans = _spans(windows)
            for span_index, span in enumerate(spans):
                begin, end = span["start"] / rate, span["stop"] / rate
                frames = math.ceil((span["stop"] - span["start"]) * config.sample_rate / rate / config.hop_length)
                times = begin + np.arange(frames) * config.hop_seconds
                times = times[times < end]
                timeline = OutputTimeline(times)
                for window in span["windows"]:
                    dataset.check_unchanged()
                    features, local_times = dataset._features(record, window)
                    local_times = local_times + window["startSample"] / rate
                    conditioning = training_conditioning(
                        record, local_times, conditioning_dim,
                    )
                    optional = {}
                    if callable(getattr(dataset, "video_window", None)):
                        video = dataset.video_window(record, local_times)
                        if video is not None:
                            optional["video"] = {name: value.unsqueeze(0).to(device) for name, value in video.items()}
                    outputs = model(
                        features[None].to(device), conditioning[None].to(device),
                        torch.tensor([len(features)], dtype=torch.long), **optional,
                    )
                    timeline.add({name: value[0] for name, value in outputs.items()}, local_times, window["stopSampleExclusive"] / rate)
                instrument = data.labels["conditioning"]["instrument"]
                decoded = decode_events(
                    timeline.finish(), torch.tensor(times), tuning=instrument["openStringMidi"], capo=instrument["capoFret"],
                    onset_threshold=onset_threshold, percussion_threshold=percussion_threshold, technique_threshold=technique_threshold,
                    connection_threshold=connection_threshold, grace_threshold=grace_threshold,
                )
                for kind in predictions:
                    predictions[kind].extend(decoded.get(kind, []))
                version4 |= decoded["policy"].get("techniqueSupervision") == SUPERVISION_VERSION
                if progress is not None:
                    progress(f"Event evaluation: {row['id']} span {span_index + 1}/{len(spans)}")
            clock = ScoreClock(data.labels, data.normalization)
            projected_notes, projected_gestures = projected_targets(data.labels, record["candidate"], clock)
            notes = [{**note, "onsetSeconds": note["proposedOnsetClipSeconds"]} for note in projected_notes if note["proposedOnsetClipSeconds"] is not None]
            gestures = [{**gesture, "onsetSeconds": gesture["proposedOnsetClipSeconds"]} for gesture in projected_gestures if gesture["proposedOnsetClipSeconds"] is not None]
            technique_truth = []
            for event in record.get("techniques", []):
                onset = event["proposedOnsetClipSeconds"]
                if onset is None:
                    continue
                for technique in event["techniques"]:
                    technique_truth.append({
                        "technique": technique,
                        "onsetSeconds": onset,
                        "direction": event["directions"].get(technique),
                        "directionKnown": event["directionMasks"].get(technique, False),
                        "stringSet": tuple(sorted(event["stringsByTechnique"][technique])),
                    })
            resolved_attack_times = [
                note["onsetSeconds"] for note in notes
                if note["onsetTimingKnownInScore"] and note["sourceLabelMask"].get("attack")
            ]
            scored = {}
            for tolerance in tolerances:
                comparisons = {}
                for name, keys, masks in (
                    ("string_onset", ("string",), ("attack",)),
                    ("string_pitch_onset", ("string", "soundingPitchMidi"), ("attack", "pitch")),
                    ("string_fret_pitch_onset", ("string", "fret", "soundingPitchMidi"), ("attack", "pitch", "fingering")),
                ):
                    truth = [note for note in notes if inside(note["onsetSeconds"], scoring) and note["onsetTimingKnownInScore"] and all(note["sourceLabelMask"][mask] for mask in masks)]
                    unknown = [note for note in notes if not inside(note["onsetSeconds"], scoring) or not note["onsetTimingKnownInScore"] or not all(note["sourceLabelMask"][mask] for mask in masks)]

                    def known_negative(event):
                        return inside(event["onsetSeconds"], scoring) and record["negativeAllowed"][6 - event["string"]] and not any(note["string"] == event["string"] and abs(note["onsetSeconds"] - event["onsetSeconds"]) <= tolerance for note in unknown)

                    comparisons[name] = event_counts(truth, predictions["notes"], tolerance, keys, known_negative, has_negative_coverage=any(record["negativeAllowed"]))
                for technique in PERCUSSION_TYPES:
                    truth = [gesture for gesture in gestures if inside(gesture["onsetSeconds"], scoring) and gesture["technique"] == technique and gesture["onsetTimingKnownInScore"] and gesture["sourceLabelMask"].get("gesture", True)]
                    guessed = [gesture for gesture in predictions["percussion"] if gesture["technique"] == technique]
                    outside = [gesture for gesture in gestures if gesture["technique"] == technique and not inside(gesture["onsetSeconds"], scoring)]

                    def known_percussion_negative(event):
                        return inside(event["onsetSeconds"], percussion_coverage) and not any(abs(gesture["onsetSeconds"] - event["onsetSeconds"]) <= tolerance for gesture in outside)

                    comparisons[technique] = event_counts(truth, guessed, tolerance, ("technique",), known_percussion_negative, has_negative_coverage=bool(percussion_coverage))
                if predictions["techniques"] or technique_truth:
                    technique_complete = record.get("techniqueAnnotationsComplete") is True

                    def known_technique_negative(event):
                        if positive_only_rasgueado and event["technique"] == "rasgueado":
                            # Native positive attributes remain scorable; absence
                            # is not evidence against a physical a-m-i gesture.
                            return inside(event["onsetSeconds"], scoring) and any(
                                known["technique"] == "rasgueado"
                                and abs(known["onsetSeconds"] - event["onsetSeconds"]) <= tolerance
                                for known in technique_truth
                            )
                        return inside(event["onsetSeconds"], scoring) and any(
                            abs(onset - event["onsetSeconds"]) <= tolerance
                            for onset in resolved_attack_times
                        )

                    truth = [event for event in technique_truth if inside(event["onsetSeconds"], scoring)]
                    guessed = [event for event in predictions["techniques"] if inside(event["onsetSeconds"], scoring)]
                    comparisons["technique_onset"] = event_counts(
                        truth, guessed, tolerance, ("technique",), known_technique_negative,
                        has_negative_coverage=technique_complete,
                    )
                    direction_truth = [event for event in truth if event["directionKnown"]]
                    direction_guessed = [event for event in guessed if event.get("direction") in ("Down", "Up")]
                    unknown_direction = [event for event in truth if not event["directionKnown"]]

                    def known_direction_negative(event):
                        return known_technique_negative(event) and not any(
                            unknown["technique"] == event["technique"]
                            and abs(unknown["onsetSeconds"] - event["onsetSeconds"]) <= tolerance
                            for unknown in unknown_direction
                        )

                    comparisons["technique_direction"] = event_counts(
                        direction_truth, direction_guessed, tolerance, ("technique", "direction"),
                        known_direction_negative, has_negative_coverage=technique_complete,
                    )
                    string_truth = [{**event, "strings": event["stringSet"]} for event in truth]
                    string_guessed = [{**event, "strings": tuple(sorted(set(event.get("strings", []))))} for event in guessed]
                    comparisons["technique_string_set"] = event_counts(
                        string_truth, string_guessed, tolerance, ("technique", "strings"),
                        known_technique_negative, has_negative_coverage=technique_complete,
                    )
                    for technique in TECHNIQUE_TYPES:
                        expected = [event for event in truth if event["technique"] == technique]
                        actual = [event for event in guessed if event["technique"] == technique]
                        comparisons[f"technique_{technique}"] = event_counts(
                            expected, actual, tolerance, ("technique",), known_technique_negative,
                            has_negative_coverage=technique_complete and not (
                                positive_only_rasgueado and technique == "rasgueado"
                            ),
                        )
                if version4:
                    if "connections" not in record:
                        raise HarnessError("Architecture v4 event evaluation requires v4 canonical connections.")
                    if any(event.get("techniqueSchemaVersion") != 4 for event in record["connections"]):
                        raise HarnessError("Do not score v4 predictions against legacy slide supervision.")
                    comparisons.update(score_note_attributes(
                        record["connections"], predictions["notes"], scoring, tolerance,
                        note_technique_threshold=note_technique_threshold,
                        unknown_notes=[note for note in notes if not note["onsetTimingKnownInScore"] or not note["sourceLabelMask"]["pitch"]],
                        negative_onsets_allowed=record["negativeAllowed"],
                    ))
                tolerance_key = f"{tolerance:g}"
                scored[tolerance_key] = {key: summarize_counts(value) for key, value in comparisons.items()}
                for name, counts in comparisons.items():
                    aggregate = totals.setdefault(tolerance_key, {}).setdefault(name, {key: False if key == "has_negative_coverage" else 0 for key in counts})
                    for key, value in counts.items():
                        aggregate[key] = aggregate[key] or value if key == "has_negative_coverage" else aggregate[key] + value
            seconds = sum(right - left for left, right in scoring)
            records.append({
                "id": row["id"], "scoredClipRanges": scoring, "scoredAudioSeconds": seconds,
                "percussionAnnotationsComplete": record["percussionAnnotationsComplete"],
                "predictedPercussionEventsPerMinute": sum(inside(event["onsetSeconds"], scoring) for event in predictions["percussion"]) * 60 / seconds,
                "metricsByToleranceSeconds": scored, "predictions": predictions,
            })
        dataset.check_unchanged()
    finally:
        for module, mode in modes:
            module.training = mode
    return {
        "schemaVersion": 1, "kind": "decoded-event-evaluation", "visibility": "private", "trainingPerformed": False,
        "recordings": records,
        "windowVisits": len(dataset.windows),
        **({"pairedTechniqueTargetPolicy": PAIRED_TECHNIQUE_TARGET_POLICY} if positive_only_rasgueado else {}),
        **({"techniqueSupervisionVersion": SUPERVISION_VERSION} if version4 else {}),
        "metricsByToleranceSeconds": {key: {name: summarize_counts(value) for name, value in groups.items()} for key, groups in totals.items()},
        "settings": {"onsetThreshold": onset_threshold, "percussionThreshold": percussion_threshold, "techniqueThreshold": technique_threshold,
                     "connectionThreshold": connection_threshold, "noteTechniqueThreshold": note_technique_threshold,
                     "graceThreshold": grace_threshold, "tolerancesSeconds": tolerances, "windowBoundaryGuardSeconds": .5},
        "policy": "Window logits are stitched before decoding; gaps remain separate. Canonical event IDs are scored once, not once per overlapping window. Metrics are masked string-specific event metrics, not complete-score quality. Unmatched predictions in unresolved annotation coverage are censored. Percussion precision/F1 require explicit negative annotation coverage; otherwise only known-positive recall and emission counts are available. No acoustic-release correctness is inferred from notated durations.",
    }
