"""Local PyTorch commands. Training is an explicitly requested operation."""

import argparse
from collections import Counter
from dataclasses import asdict, fields, replace
from datetime import datetime
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import scipy
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from .dataset_io import ROOT, publish_json, read_json, sha256
from .transcriber_audio import (
    DOWNBEAT_CONDITIONING_DIM, FeatureConfig, HarnessError, audio_features,
    conditioning_features, read_audio_window,
)
from .transcriber_data import PAIRED_TECHNIQUE_TARGET_POLICY, EpochShuffleSampler, TrainingDataset, collate_windows


def private_output(path, root=ROOT):
    root = Path(root).resolve()
    path = Path(path)
    path = path if path.is_absolute() else root / path
    absolute = path.absolute()
    if path.resolve() != absolute or not any(absolute.is_relative_to(root / name) for name in ("runs", "cache", "checkpoints")):
        raise HarnessError("Output must be an unaliased path under the private runs, cache or checkpoints directory.")
    return absolute


def default_config():
    from .transcriber_model import ModelConfig
    from .transcriber_runtime import TrainingConfig
    return {
        "schemaVersion": 1, "features": asdict(FeatureConfig()), "model": asdict(ModelConfig(architecture_version=4, conditioning_dim=DOWNBEAT_CONDITIONING_DIM)),
        "training": asdict(TrainingConfig()),
        "data": {"manifest": "data\\releases\\dataset-v1\\manifest.json", "batch_size": 4, "num_workers": 0, "num_threads": max(1, min(32, (os.cpu_count() or 1) * 3 // 4)), "cache": "cache\\transcriber"},
    }


def load_config(path=None):
    from .transcriber_model import ModelConfig
    from .transcriber_runtime import TrainingConfig
    config = read_json(Path(path)) if path else default_config()
    required = {"schemaVersion", "features", "model", "training", "data"}
    if not isinstance(config, dict) or type(config.get("schemaVersion")) is not int or config["schemaVersion"] != 1 or not required <= config.keys() or config.keys() - required - {"video"}:
        raise HarnessError("Expected version-1 harness configuration.")
    instances = []
    for key, cls in (("features", FeatureConfig), ("model", ModelConfig), ("training", TrainingConfig)):
        values = config[key]
        if not isinstance(values, dict) or set(values) - {field.name for field in fields(cls)}:
            raise HarnessError(f"Unknown {key} configuration fields.")
        instances.append(cls(**values))
    data = config["data"]
    if not isinstance(data, dict) or set(data) != {"manifest", "batch_size", "num_workers", "num_threads", "cache"}:
        raise HarnessError("Unknown or missing data-loader configuration.")
    for key, lower, upper in (("batch_size", 1, 64), ("num_workers", 0, 0), ("num_threads", 1, 32)):
        if type(data[key]) is not int or not lower <= data[key] <= upper:
            raise HarnessError(f"Invalid data-loader {key}.")
    if not isinstance(data["manifest"], str) or not isinstance(data["cache"], str):
        raise HarnessError("Manifest and cache paths must be explicit strings.")
    if instances[0].n_mels != instances[1].n_mels or instances[1].conditioning_dim != DOWNBEAT_CONDITIONING_DIM:
        raise HarnessError("New training requires 16-dimensional downbeat-conditioned inputs.")
    if "video" in config:
        video = config["video"]
        if not isinstance(video, dict) or set(video) != {"index", "model"} or not isinstance(video["index"], str) or not video["index"].strip() or not isinstance(video["model"], dict):
            raise HarnessError("Video configuration requires an explicit paired index and model configuration.")
        video_model_config(video["model"])
    return config, *instances


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(config, feature_config, model_config, split, root, manifest_override=None):
    manifest = Path(manifest_override or config["data"]["manifest"])
    if not manifest.is_absolute():
        manifest = Path(root) / manifest
    kwargs = {}
    if "video" in config:
        index = Path(config["video"]["index"])
        kwargs["video_index_path"] = index if index.is_absolute() else Path(root) / index
    dataset = TrainingDataset(manifest, split, feature_config, model_config, root=root, cache_dir=private_output(config["data"]["cache"], root), **kwargs)
    if "video" in config:
        indices = dataset.video_paired_window_indices
        if not indices:
            raise HarnessError(f"No source-bound prepared paired windows are available in the {split} split.")
        dataset.video_coverage = {
            "releaseWindows": len(dataset.windows), "pairedWindows": len(indices), "releaseRecordings": len(dataset.records),
            "selectionPolicy": "Indexed recordings and windows overlapping prepared clips only; supervision outside prepared intervals masked; local tracking gaps retain audio/GP supervision.",
            **dataset.video_paired_coverage,
        }
        dataset.windows = [dataset.windows[index] for index in indices]
        paired_records = {id(record) for record, _ in dataset.windows}
        dataset.records = [record for record in dataset.records if id(record) in paired_records]
        dataset.paired_only = True
        dataset.video_coverage["pairedRecordings"] = len(dataset.records)
        dataset.video_coverage["excludedWindows"] = dataset.video_coverage["releaseWindows"] - len(indices)
        dataset.video_coverage["excludedRecordings"] = dataset.video_coverage["releaseRecordings"] - len(dataset.records)
    return dataset


def make_loader(dataset, config, training, *, shuffle):
    sampler = EpochShuffleSampler(dataset, training.seed) if shuffle else None
    return DataLoader(
        dataset, batch_size=config["data"]["batch_size"], sampler=sampler, shuffle=False,
        num_workers=config["data"]["num_workers"], collate_fn=collate_windows,
        generator=torch.Generator().manual_seed(training.seed),
    )


def run_identity(dataset, config, features, model, training, device):
    from .connection_supervision import SUPERVISION_VERSION

    settings = asdict(training)
    settings.pop("epochs", None)
    settings.pop("max_steps", None)
    settings.pop("max_seconds", None)
    modules = ("transcriber.py", "transcriber_audio.py", "transcriber_data.py", "transcriber_model.py", "transcriber_runtime.py", "transcriber_events.py", "technique_supervision.py", "connection_supervision.py", "dataset_release.py", "training_windows.py", "score_alignment.py", "dataset_io.py", "canonical_events.py", "gp_events.py", "inspect_gp_files.py", "settings.py")
    result = {
        "schemaVersion": 1, "manifest_sha256": dataset.manifest_sha256,
        "features": asdict(features), "model": asdict(model), "training": settings,
        "techniqueSupervisionVersion": SUPERVISION_VERSION if model.architecture_version >= 4 else f"historical-architecture-v{model.architecture_version}",
        "batch_size": config["data"]["batch_size"], "num_workers": config["data"]["num_workers"],
        "device": str(device), "implementationSha256": {name: sha256(Path(__file__).with_name(name)) for name in modules},
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__},
        "dataPolicy": "private-approved-training-release; no source legends or presentation settings as model inputs",
    }
    if "video" in config:
        from .transcriber_video import VideoConfig

        result["video"] = {"config": asdict(VideoConfig(**config["video"]["model"])), "data": dataset.video_identity}
        result["video"]["windowPolicy"] = "indexed-prepared-intervals-retain-tracking-gaps-v1"
        result["initialization"] = {
            "kind": "joint-audio-numeric-video-from-scratch",
            "audioParametersFrozen": False, "optimizerStateImported": False,
        }
        for name in ("transcriber_video.py", "paired_video.py", "video_features.py"):
            result["implementationSha256"][name] = sha256(Path(__file__).with_name(name))
    return result


def video_model_config(values, *, allow_legacy=False):
    from .transcriber_video import VideoConfig
    from .fretboard_features import SCHEMA_VERSION, STRUCTURED_DIM

    if not isinstance(values, dict):
        raise HarnessError("Video model configuration must be an object.")
    values = dict(values)
    if values.keys() - {field.name for field in fields(VideoConfig)}:
        raise HarnessError("Unknown video model configuration fields. Only numeric landmark/motion/geometry inputs are supported, not RGB checkpoints.")
    if allow_legacy and values.get("architecture_version") in (4, 5):
        values.setdefault(
            "feature_group_version",
            None if values["architecture_version"] == 4 else "anatomy-representation-groups-v1",
        )
        values.setdefault("experiment_mode", "geometry")
        try:
            return VideoConfig(**values)
        except (TypeError, ValueError) as error:
            raise HarnessError(f"Invalid legacy inference video configuration: {error}") from error
    if values.get("architecture_version") != 6 or values.get("input_schema_version") != 5:
        raise HarnessError("Paired models require explicit joint video architecture_version 6 and input_schema_version 5; historical, frozen or unversioned paired configurations cannot be resumed or reinterpreted.")
    if values.get("structured_dim") != STRUCTURED_DIM:
        raise HarnessError(f"Paired models require structured_dim {STRUCTURED_DIM} with four fretboard-aware views; historical configurations cannot be reinterpreted.")
    return VideoConfig(**values)


def _wrap_video_model(model, config):
    if "video" not in config:
        return model
    from .transcriber_video import AudioVideoTranscriber

    return AudioVideoTranscriber(model, video_model_config(config["video"]["model"]))


def preflight(args):
    from .transcriber_model import FingerstyleTranscriber
    from .transcriber_runtime import _progress_due, format_duration

    log_progress("Preflight: loading configuration...")
    config, features, model_config, training = load_config(args.config)
    torch.set_num_threads(config["data"]["num_threads"])
    seed_everything(training.seed)
    log_progress(f"Preflight: configuration loaded | CPU threads {config['data']['num_threads']} | no training.")
    if args.forward:
        log_progress("Preflight: initializing untrained acoustic model...")
    model = FingerstyleTranscriber(model_config).eval() if args.forward else None
    report = {"schemaVersion": 1, "kind": "transcriber-preflight", "trainingRun": False, "weights": "untrained-in-memory-only" if model else "not-created", "splits": {}}
    initialized = False
    for split in ("train", "validation"):
        log_progress(f"Preflight {split}: loading data and verifying input identities...")
        dataset = make_dataset(config, features, model_config, split, args.data_root, args.manifest)
        log_progress(f"Preflight {split}: loaded {len(dataset)} windows.")
        if model is not None and not initialized:
            log_progress("Preflight: preparing the forward-pass model...")
            model = _wrap_video_model(model, config).eval()
            initialized = True
            log_progress("Preflight: model ready.")
        counts = Counter()
        target_counts = Counter()
        paired_target_counts = Counter()
        first = None
        started = last_log = time.perf_counter()
        log_progress(f"Preflight {split}: scanning features, targets and masks...")
        for item in dataset:
            if first is None:
                first = item
            counts["windows"] += 1
            voice_policy = item["metadata"].get("voiceSupervisionPolicy", "native-multivoice")
            counts[f"voice_policy_{voice_policy}"] += 1
            counts["frames"] += len(item["features"])
            counts["collisions_masked"] += item["metadata"]["stringFrameCollisionsMasked"]
            if "video" in item:
                counts["video_frames"] += len(item["video"]["structured"])
                counts["video_available_view_frames"] += int(item["video"]["structured_available"].any(-1).sum())
                counts["audio_frames_with_video"] += int((item["video"]["frame_indices"] >= 0).sum())
                observed = bool((item["video"]["frame_indices"] >= 0).any())
                counts["windows_with_observed_video"] += int(observed)
                counts["tracking_gap_only_windows"] += int(not observed)
                counts["structured_available_values"] += int(item["video"]["structured_available"].sum())
                counts["structured_available_view_frames"] += int(item["video"]["structured_available"].any(-1).sum())
                from .technique_supervision import TECHNIQUE_TYPES
                from .transcriber_model import PERCUSSION_TYPES

                video = item["video"]
                availability = video["structured_available"]
                guitar_observed = availability[:, :, :42].any(-1) | availability[:, :, 84:98].any(-1)
                hand_observed = availability[:, :, 98:140].any(-1) | availability[:, :, 184:186].any(-1)
                counts["guitar_relative_observation_view_frames"] += int(guitar_observed.sum())
                counts["independent_hand_observation_view_frames"] += int(hand_observed.sum())
                counts["unassigned_hand_observation_view_frames"] += int(hand_observed[:, 2:].sum())
                indices = video["frame_indices"]
                safe_indices = indices.clamp_min(0)
                view_available = video["structured_available"].any(-1)
                plucking = (indices >= 0) & view_available[safe_indices, 1] & video["technique_available"][safe_indices]
                for task, names in (("technique", TECHNIQUE_TYPES), ("percussion", PERCUSSION_TYPES)):
                    if task not in item["targets"]:
                        continue
                    for axis, name in enumerate(names):
                        positive = (item["targets"][task][:, axis] > .5) & item["masks"][task][:, axis]
                        target_counts[name] += int(positive.sum())
                        paired_target_counts[name] += int((positive & plucking).sum())
            for name, mask in item["masks"].items():
                counts[f"supervised_{name}"] += int(mask.sum())
            now = time.perf_counter()
            if _progress_due(counts["windows"], len(dataset), now, last_log):
                log_progress(f"Preflight {split}: windows {counts['windows']}/{len(dataset)} | frames {counts['frames']} | elapsed {format_duration(now - started)}")
                last_log = now
        log_progress(f"Preflight {split}: feature/target scan completed.")
        row = dict(counts)
        row["voiceSupervisionPolicy"] = {
            policy: counts[f"voice_policy_{policy}"]
            for policy in ("native-multivoice", "intentional-single-voice", "flattened-or-unknown")
        }
        if "video" in config:
            row["videoCoverage"] = dataset.video_coverage
            if dataset.video_coverage["audioFramesWithUsableVideo"] == 0:
                report.setdefault("warnings", []).append(f"{split}: prepared paired coverage contains no usable numeric video on acoustic frames; local gaps remain supervised, but an entirely missing-visual training split cannot start joint training.")
                log_progress(f"Preflight warning: {report['warnings'][-1]}")
            row["positiveTargetFrames"] = dict(target_counts)
            row["positiveTargetFramesWithUsablePluckingVideo"] = dict(paired_target_counts)
            if "videoIdentity" in report and report["videoIdentity"] != dataset.video_identity:
                raise HarnessError("Video inputs changed between preflight splits.")
            report["videoIdentity"] = dataset.video_identity
            report["targetPolicy"] = "Existing frozen canonical GP supervision only. Frame counts include repeated overlapping windows. Arpeggio is finger-roll notation; brush strokes are not silently relabeled as compound rasgueados. No heuristic detector labels enter targets."
        if model is not None:
            log_progress(f"Preflight {split}: running no-gradient forward pass...")
            batch = collate_windows([first])
            with torch.no_grad():
                if "video" in batch:
                    outputs = model(batch["features"], batch["conditioning"], batch["lengths"], video=batch["video"])
                else:
                    outputs = model(batch["features"], batch["conditioning"], batch["lengths"])
            if not all(torch.isfinite(value).all() for value in outputs.values()):
                raise HarnessError("Model forward produced nonfinite output.")
            row["forwardShapes"] = {name: list(value.shape) for name, value in outputs.items()}
            log_progress(f"Preflight {split}: forward pass completed; outputs are finite.")
        report["splits"][split] = row
        report["manifestSha256"] = dataset.manifest_sha256
        log_progress(f"Preflight {split}: completed.")
    if "video" in config:
        report["implementationSha256"] = {
            name: sha256(Path(__file__).with_name(name))
            for name in ("transcriber.py", "transcriber_data.py", "paired_video.py", "video_features.py", "transcriber_video.py", "transcriber_model.py", "transcriber_runtime.py", "transcriber_events.py")
        }
        report["videoInputs"] = "Numeric guitar-relative observations, independently available local hand positions/motion/orientation and coarse instrument-context evidence with masks; no RGB neural-network inputs or heuristic technique labels. Coarse coverage is reported by the dataset in videoCoverage, not absolute fret/contact accuracy. Unassigned hands are not known plucking-technique evidence."
        report["initialization"] = "joint-audio-numeric-video-from-scratch"
        report["optimization"] = "One optimizer trains acoustic, numeric-video and fusion parameters together; preflight performs no optimizer updates."
    output_path = private_output(args.output, args.data_root)
    log_progress(f"Preflight: saving report to {output_path}...")
    publish_json(output_path, report)
    log_progress("Preflight: report saved.")
    print({"preflight": "completed", "windows": {key: value["windows"] for key, value in report["splits"].items()}, "trainingRun": False}, flush=True)
    return report


def log_progress(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def print_training_summary(result, summary_path):
    from .transcriber_runtime import format_duration

    records, windows = result["dataset_recordings"], result["dataset_windows"]
    step_summary = f"  Optimizer steps: {result['training_steps_processed']} this invocation; {result['global_step']} total"
    if "optimizer_updates" in result:
        step_summary = (
            f"  Consumed training batches: {result['training_steps_processed']} this invocation; {result['global_step']} total"
            f" | optimizer updates this invocation: {result['optimizer_updates']}"
            f" | skipped updates: {result['optimizer_skipped_batches']}"
        )
    validation, best = result.get("validation"), result.get("best_score")
    validation_text = f"{validation['loss']:.6f}" if validation is not None else "not completed"
    best_text = f"{best:.6f}" if best is not None else "not selected"
    lines = [
        "", "Training summary",
        f"  Total elapsed: {format_duration(result['elapsed_seconds'])} (setup {format_duration(result['setup_seconds'])}; training/validation {format_duration(result['training_elapsed_seconds'])})",
        f"  Dataset: {records['train']} training recordings / {windows['train']} windows; {records['validation']} validation recordings / {windows['validation']} windows",
        f"  Window visits this invocation: {result['training_windows_processed']} training; {result['validation_windows_processed']} validation (includes repeated passes)",
        step_summary,
        f"  Completed epochs: {result['epoch']}/{result['epochs_requested']} ({result['epochs_completed_this_run']} this invocation)",
        f"  Stopped by: {result['stopped_by']}",
        f"  Last completed validation loss: {validation_text}; best {best_text}; validation pending: {result.get('validation_pending', False)}",
        f"  Latest checkpoint: {result['latest_checkpoint']}",
        f"  Best checkpoint: {result['best_checkpoint']}",
        f"  Saved summary: {summary_path}",
    ]
    if result.get("best_event_checkpoint"):
        lines.insert(-1, f"  Best decoded-event checkpoint: {result['best_event_checkpoint']} | score {result['best_event_score']:.6f}")
        lines.insert(4, f"  Additional decoded-event validation window visits: {result['event_validation_windows_processed']}")
    if result.get("stopped_by") == "max_seconds":
        lines.insert(-1, f"  Paused at wall-clock budget; native operations can overrun. Reported overrun: {result.get('budget_overrun_seconds', 0.):.3f}s")
        if result.get("resume_action"):
            lines.insert(-1, f"  {result['resume_action']}")
    print("\n".join(lines), flush=True)


def train(args):
    started = time.perf_counter()
    from .transcriber_runtime import TrainingBudget, TrainingBudgetExpired, load_checkpoint

    config, features, model_config, training = load_config(args.config)
    max_hours = getattr(args, "max_hours", None)
    if max_hours is not None:
        training = replace(training, max_seconds=max_hours * 3600)
        config = {**config, "training": asdict(training)}
    budget = TrainingBudget(training.max_seconds, started_at=started)
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    run_dir = private_output(args.run_dir, args.data_root)
    if args.resume:
        if Path(args.resume).absolute() != run_dir / "latest.pt":
            raise HarnessError("Resume must use latest.pt from this exact run directory.")
        if (run_dir / ".training.lock").exists():
            raise HarnessError("This training run is locked; do not replace its summary while another trainer may be running.")
    elif run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        raise HarnessError("A fresh run directory must be absent or empty. A setup-only budget pause has no checkpoint: use a new empty run directory.")
    try:
        budget.check()
        result = _train_with_budget(args, config, features, model_config, training, run_dir, budget)
    except TrainingBudgetExpired:
        cursor = checkpoint["cursor"] if checkpoint else {"epoch": 0, "next_batch_index": 0, "global_step": 0}
        history = checkpoint["history"] if checkpoint else []
        counts = checkpoint.get("resume_state", history[-1] if history else {}) if checkpoint else {}
        result = {
            "run_dir": str(run_dir), **cursor, **budget.report(),
            "status": "paused", "stopped_by": "max_seconds", "training_started": False,
            "training_elapsed_seconds": 0., "setup_seconds": time.perf_counter() - started,
            "validation_pending": counts.get("validation_pending", not bool(history)),
            "resume_phase": counts.get("phase", "setup"),
            "best_score": checkpoint["best_score"] if checkpoint else None,
            "validation": history[-1]["validation"] if history else None,
            "latest_checkpoint": str(run_dir / "latest.pt") if checkpoint else None,
            "best_checkpoint": str(run_dir / "best.pt") if history else None,
            "best_event_checkpoint": None, "best_event_score": None,
            "epochs_requested": training.epochs, "epochs_remaining": max(0, training.epochs - cursor["epoch"]),
            "epochs_completed_this_run": 0, "training_steps_processed": 0,
            "optimizer_updates": 0, "optimizer_skipped_batches": 0,
            "total_optimizer_updates": counts.get("optimizer_updates", cursor["global_step"]),
            "total_optimizer_skipped_batches": counts.get("optimizer_skipped_batches", 0),
            "training_windows_processed": 0, "validation_windows_processed": 0, "event_validation_windows_processed": 0,
            "dataset_recordings": {"train": None, "validation": None}, "dataset_windows": {"train": None, "validation": None},
            "resume_action": "Resume the unchanged latest.pt with a fresh per-invocation budget." if checkpoint else "No model or optimizer was created. Restart with a larger budget in a new empty run directory; this setup-only summary is not a checkpoint.",
        }
        if checkpoint is None:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise HarnessError("Fresh run directory became nonempty before budget-pause publication.")
            run_dir.mkdir(parents=True, exist_ok=True)
    result.update(budget.report())
    summary_path = run_dir / "summary.json"
    publish_json(summary_path, result)
    print_training_summary(result, summary_path)
    return result


def _train_with_budget(args, config, features, model_config, training, run_dir, budget):
    from .transcriber_model import FingerstyleTranscriber
    from .transcriber_runtime import resolve_device, run_training
    from .transcriber_events import checkpoint_event_score, evaluate_events
    torch.set_num_threads(config["data"]["num_threads"])
    device = resolve_device(training.device)
    seed_everything(training.seed)
    step_cap = training.max_steps if training.max_steps is not None else "none"
    log_progress(f"Setup: device {device} | CPU threads {config['data']['num_threads']} | batch size {config['data']['batch_size']} | epochs {training.epochs} | step cap {step_cap}")
    log_progress("Loading training data...")
    budget.check()
    train_data = make_dataset(config, features, model_config, "train", args.data_root, args.manifest)
    budget.check()
    log_progress(f"Loaded {len(train_data)} training windows from {len(train_data.records)} recordings.")
    if "video" in config and train_data.video_coverage["audioFramesWithUsableVideo"] == 0:
        raise HarnessError("Joint training has no usable numeric video on any selected prepared training frame. Local tracking-gap windows are retained, but prepare or correct the missing visual inputs before starting a joint run.")
    log_progress("Loading validation data...")
    validation_data = make_dataset(config, features, model_config, "validation", args.data_root, args.manifest)
    budget.check()
    log_progress(f"Loaded {len(validation_data)} validation windows from {len(validation_data.records)} recordings.")
    if "video" in config and validation_data.video_coverage["audioFramesWithUsableVideo"] == 0:
        log_progress("Warning: validation contains prepared paired footage but no usable numeric video; validation will exercise the learned audio-only path.")
    if train_data.manifest_sha256 != validation_data.manifest_sha256:
        raise HarnessError("Training and validation reference different releases.")
    if "video" in config and train_data.video_identity != validation_data.video_identity:
        raise HarnessError("Training and validation reference different paired-video inputs.")
    identity = run_identity(train_data, config, features, model_config, training, device)
    budget.check()
    train_loader = make_loader(train_data, config, training, shuffle=True)
    validation_loader = make_loader(validation_data, config, training, shuffle=False)
    budget.check()
    model = FingerstyleTranscriber(model_config)
    if "video" in config:
        model = _wrap_video_model(model, config)
        log_progress("Joint acoustic/numeric-video/fusion optimization; all parameters trainable. " + ("Restoring exact joint run state." if args.resume else "All weights randomly initialized; no checkpoint imported."))
        for split, dataset in (("train", train_data), ("validation", validation_data)):
            coverage = dataset.video_coverage
            log_progress(f"Paired {split}: {len(dataset)} windows; excluded {coverage['excludedWindows']} release windows and {coverage['excludedRecordings']} recordings outside the paired selection.")
            log_progress(f"Visual {split}: {coverage['audioFramesWithUsableVideo']} usable-video acoustic frames / {coverage['preparedAudioFrames']} prepared acoustic frames; {coverage['trackingGapOnlyWindows']} tracking-gap-only windows retained.")
    setup_seconds = time.perf_counter() - budget.started_at

    def event_progress(message):
        budget.check()
        log_progress(message)

    def event_evaluator(current_model):
        report = evaluate_events(current_model, validation_data, device, tolerances=(.1,), progress=event_progress)
        return checkpoint_event_score(report)

    result = run_training(
        model, train_loader, validation_loader,
        training, run_dir, identity, resume=args.resume, progress=log_progress, event_evaluator=event_evaluator,
        started_at=budget.started_at, deadline=budget.deadline,
    )
    result.update(
        training_elapsed_seconds=result.get("training_elapsed_seconds", result["elapsed_seconds"]),
        setup_seconds=setup_seconds,
        dataset_recordings={"train": len(train_data.records), "validation": len(validation_data.records)},
    )
    if "video" in config:
        result["videoCoverage"] = {"train": train_data.video_coverage, "validation": validation_data.video_coverage}
        result["trainingMode"] = "joint-audio-numeric-video-from-scratch"
    else:
        result["trainingMode"] = "audio-from-scratch"
    return result




def checkpoint_model(path, device_name, *, allow_inference=True):
    from .transcriber_model import FingerstyleTranscriber, ModelConfig
    from .transcriber_runtime import INFERENCE_FORMAT, checkpoint_identity, load_checkpoint, resolve_device
    checkpoint = load_checkpoint(path, allow_inference=allow_inference)
    identity = checkpoint_identity(checkpoint)
    if checkpoint.get("format") != INFERENCE_FORMAT:
        if checkpoint["global_step"] <= 0:
            raise HarnessError("Inference/evaluation requires a checkpoint with actual training steps, not random initialization.")
        counts = checkpoint.get("resume_state", checkpoint["history"][-1] if checkpoint["history"] else {})
        if "video" in identity and counts.get("optimizer_updates") == 0:
            raise HarnessError("Joint inference/evaluation requires actual optimizer updates, not only skipped batches.")
    model = FingerstyleTranscriber(ModelConfig(**identity["model"]))
    if "video" in identity:
        from .transcriber_video import AudioVideoTranscriber, VideoConfig

        video_values = dict(identity["video"]["config"])
        if video_values.get("architecture_version") == 4:
            video_values.setdefault("feature_group_version", None)
        video_values.setdefault("experiment_mode", "geometry")
        model = AudioVideoTranscriber(model, VideoConfig(**video_values))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = resolve_device(device_name)
    return model.to(device).eval(), checkpoint, device


def _evaluation_video_config(args, identity, config):
    index = getattr(args, "video_index", None)
    audio_only = getattr(args, "audio_only", False)
    if index and audio_only:
        raise HarnessError("Choose paired evaluation or --audio-only, not both.")
    if index and "video" not in identity:
        raise HarnessError("An audio-only checkpoint cannot consume video inputs.")
    if "video" in identity and not index and not audio_only:
        raise HarnessError("Paired checkpoint evaluation needs --video-index or explicit --audio-only.")
    if index:
        config["video"] = {"index": index, "model": identity["video"]["config"]}


def evaluate(args):
    from .transcriber_model import ModelConfig
    from .transcriber_runtime import evaluate_model, TrainingConfig
    model, checkpoint, device = checkpoint_model(args.checkpoint, args.device, allow_inference=False)
    identity = checkpoint["identity"]
    config = default_config()
    _evaluation_video_config(args, identity, config)
    config["data"].update(batch_size=identity["batch_size"], num_workers=0)
    torch.set_num_threads(config["data"]["num_threads"])
    feature_config, model_config = FeatureConfig(**identity["features"]), ModelConfig(**identity["model"])
    training = TrainingConfig(**checkpoint["training_config"])
    dataset = make_dataset(config, feature_config, model_config, args.split, args.data_root, args.manifest)
    if "video" in identity:
        dataset.paired_technique_target_policy = PAIRED_TECHNIQUE_TARGET_POLICY
    if dataset.manifest_sha256 != identity["manifest_sha256"]:
        raise HarnessError("Evaluation release differs from this run; do not silently substitute a different split.")
    metrics = evaluate_model(model, make_loader(dataset, config, training, shuffle=False), device, sparsity_weight=training.sparsity_weight)
    report = {"schemaVersion": 1, "kind": "transcriber-evaluation", "split": args.split, "checkpointSha256": sha256(Path(args.checkpoint)), "manifestSha256": dataset.manifest_sha256, "metrics": metrics, "developmentOnly": True, "visibility": "private"}
    if "video" in identity:
        report["pairedVideo"] = {
            "inputIdentity": dataset.video_identity if "video" in config else None,
            "audioOnlyAblation": "video" not in config,
            "targetPolicy": PAIRED_TECHNIQUE_TARGET_POLICY,
            "evaluationCoverage": dataset.video_coverage if "video" in config else "full-requested-release-split",
        }
    publish_json(private_output(args.output, args.data_root), report)
    print(metrics)
    return report


def evaluate_decoded_events(args):
    from .transcriber_model import ModelConfig, PERCUSSION_TYPES
    from .transcriber_events import evaluate_events

    checkpoint_hash = sha256(Path(args.checkpoint))
    implementation = {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")}
    model, checkpoint, device = checkpoint_model(args.checkpoint, args.device, allow_inference=False)
    identity = checkpoint["identity"]
    config = default_config()
    _evaluation_video_config(args, identity, config)
    torch.set_num_threads(config["data"]["num_threads"])
    dataset = make_dataset(config, FeatureConfig(**identity["features"]), ModelConfig(**identity["model"]), args.split, args.data_root, args.manifest)
    if "video" in identity:
        dataset.paired_technique_target_policy = PAIRED_TECHNIQUE_TARGET_POLICY
    report = evaluate_events(
        model, dataset, device, tolerances=args.tolerances, onset_threshold=args.onset_threshold,
        percussion_threshold=args.percussion_threshold, technique_threshold=args.technique_threshold,
        connection_threshold=args.connection_threshold, note_technique_threshold=args.note_technique_threshold,
        grace_threshold=args.grace_threshold, progress=log_progress,
    )
    if sha256(Path(args.checkpoint)) != checkpoint_hash:
        raise HarnessError("Checkpoint changed during event evaluation.")
    if implementation != {path.name: sha256(path) for path in Path(__file__).parent.glob("*.py")}:
        raise HarnessError("Implementation changed during event evaluation.")
    report.update(
        checkpointSha256=checkpoint_hash, manifestSha256=dataset.manifest_sha256, split=args.split,
        trainingManifestSha256=identity["manifest_sha256"], sameReleaseAsTraining=dataset.manifest_sha256 == identity["manifest_sha256"],
        developmentOnly=True,
        implementationSha256=implementation,
        runtime={"torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__, "soundfile": sf.__version__},
    )
    if "video" in identity:
        report["pairedVideo"] = {
            "inputIdentity": dataset.video_identity if "video" in config else None,
            "audioOnlyAblation": "video" not in config,
            "targetPolicy": PAIRED_TECHNIQUE_TARGET_POLICY,
            "evaluationCoverage": dataset.video_coverage if "video" in config else "full-requested-release-split",
        }
    output_path = private_output(args.output, args.data_root)
    publish_json(output_path, report)
    print("Decoded-event evaluation (masked coverage)")
    for tolerance, metrics in report["metricsByToleranceSeconds"].items():
        joint = metrics["string_fret_pitch_onset"]
        pitch = metrics["string_pitch_onset"]
        joint_text = f"{joint['f1']:.3f}" if joint["f1"] is not None else "unavailable"
        pitch_text = f"{pitch['f1']:.3f}" if pitch["f1"] is not None else "unavailable"
        print(f"  {float(tolerance) * 1000:g} ms: string/pitch F1 {pitch_text}; string/fret/pitch F1 {joint_text}")
        percussion = [metrics[name] for name in PERCUSSION_TYPES if metrics[name]["has_negative_coverage"]]
        if percussion:
            tp, fp, fn = (sum(item[key] for item in percussion) for key in ("true_positive", "false_positive", "false_negative"))
            f1 = f"{2 * tp / (2 * tp + fp + fn):.3f}" if 2 * tp + fp + fn else "unavailable"
            print(f"    Percussion F1 {f1} | matched {tp}, extra {fp}, missed {fn}")
        else:
            print("    Percussion precision/F1 unavailable: no confirmed negative coverage.")
    print(f"Saved event report: {output_path}")
    return report


def inference_metadata(value, conditioning_dim=DOWNBEAT_CONDITIONING_DIM):
    required = {"openStringMidi", "capoFret", "tempo", "timeSignature"}
    if conditioning_dim == DOWNBEAT_CONDITIONING_DIM:
        required.add("firstFullBarDownbeatSeconds")
    optional = {"tempoChanges", "timeSignatureChanges", "firstFullBarDownbeatSeconds"}
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - optional:
        downbeat_requirement = " and firstFullBarDownbeatSeconds" if conditioning_dim == DOWNBEAT_CONDITIONING_DIM else ""
        raise HarnessError(
            "Inference metadata requires tuning, full capo, BPM/beat unit, meter"
            f"{downbeat_requirement}; only explicit time-based schedules are optional."
        )
    if not isinstance(value["tempo"], dict) or not {"bpm", "beatUnit"} <= set(value["tempo"]) or set(value["tempo"]) - {"bpm", "beatUnit", "linear"}:
        raise HarnessError("Initial tempo requires explicit BPM and beat unit.")
    tempo = [{"position": 0., **value["tempo"]}]
    changes = value.get("tempoChanges", [])
    meter_changes = value.get("timeSignatureChanges", [])
    if not isinstance(changes, list) or not isinstance(meter_changes, list):
        raise HarnessError("Timing changes must be explicit event lists.")
    for event in changes:
        if not isinstance(event, dict) or not {"timeSeconds", "bpm", "beatUnit"} <= set(event) or set(event) - {"timeSeconds", "bpm", "beatUnit", "linear"}:
            raise HarnessError("A tempo change requires timeSeconds, BPM and beat unit.")
        tempo.append({"position": event["timeSeconds"], "bpm": event["bpm"], "beatUnit": event["beatUnit"], "linear": event.get("linear", False)})
    meters = [{"position": 0., "timeSignature": value["timeSignature"]}]
    for event in meter_changes:
        if not isinstance(event, dict) or set(event) != {"timeSeconds", "timeSignature"}:
            raise HarnessError("A meter change requires timeSeconds and timeSignature.")
        meters.append({"position": event["timeSeconds"], "timeSignature": event["timeSignature"]})
    downbeat = value.get("firstFullBarDownbeatSeconds")
    conditioning = conditioning_features(
        value["openStringMidi"], value["capoFret"], tempo, meters, [0.],
        first_full_bar_downbeat=downbeat if conditioning_dim == DOWNBEAT_CONDITIONING_DIM else None,
    )
    if conditioning.shape[1] != conditioning_dim:
        raise HarnessError("Inference metadata conditioning differs from the checkpoint contract.")
    return tempo, meters


def infer(args):
    from .transcriber_model import decode_events
    from .transcriber_events import OutputTimeline
    from .transcriber_runtime import checkpoint_identity
    model, checkpoint, device = checkpoint_model(args.checkpoint, args.device)
    identity = checkpoint_identity(checkpoint)
    torch.set_num_threads(4)
    metadata = read_json(Path(args.metadata))
    conditioning_dim = identity["model"]["conditioning_dim"]
    tempos, meters = inference_metadata(metadata, conditioning_dim)
    config = FeatureConfig(**identity["features"])
    audio_path = Path(args.audio).resolve()
    if audio_path.suffix.lower() not in (".mp3", ".flac", ".wav"):
        raise HarnessError("Inference accepts local MP3, FLAC or WAV audio.")
    audio_hash = sha256(audio_path)
    info = sf.info(audio_path)
    duration = info.frames / info.samplerate
    if not 0 < duration <= 900 or not 1 <= info.channels <= 8:
        raise HarnessError("Inference audio must be nonempty, at most 15 minutes, and have one to eight channels.")
    video_bundle = None
    if getattr(args, "video_bundle", None):
        if "video" not in identity:
            raise HarnessError("--video-bundle requires a paired-video checkpoint.")
        from .paired_video import load_inference_video

        video_bundle = load_inference_video(args.video_bundle, audio_hash)
    total_frames = math.ceil(duration / config.hop_seconds)
    frame_times = np.arange(total_frames) * config.hop_seconds
    timeline = OutputTimeline(frame_times)
    stride_frames = max(1, round(6 / config.hop_seconds))
    window_frames = max(stride_frames, round(8 / config.hop_seconds))
    peak = 0.
    with torch.no_grad():
        for start_frame in range(0, total_frames, stride_frames):
            start_sample = round(start_frame * config.hop_seconds * info.samplerate)
            stop_sample = min(info.frames, round((start_frame + window_frames) * config.hop_seconds * info.samplerate))
            samples, rate = read_audio_window(audio_path, start_sample, stop_sample, sample_rate=info.samplerate, channels=info.channels, sample_count=info.frames)
            peak = max(peak, float(np.max(np.abs(samples))))
            features, local_times = audio_features(samples, rate, config)
            times = local_times + start_sample / rate
            conditioning = conditioning_features(
                metadata["openStringMidi"], metadata["capoFret"], tempos, meters, times,
                first_full_bar_downbeat=metadata.get("firstFullBarDownbeatSeconds") if conditioning_dim == DOWNBEAT_CONDITIONING_DIM else None,
            )
            if video_bundle is not None:
                video = {name: value.unsqueeze(0).to(device) for name, value in video_bundle.window(times).items()}
                outputs = model(features[None].to(device), conditioning[None].to(device), torch.tensor([len(features)], dtype=torch.long), video=video)
            else:
                outputs = model(features[None].to(device), conditioning[None].to(device), torch.tensor([len(features)], dtype=torch.long))
            count = min(len(features), total_frames - start_frame)
            timeline.add({name: value[0, :count] for name, value in outputs.items()}, times[:count], stop_sample / rate)
            if start_frame + count == total_frames:
                break
    if peak <= 1e-8:
        raise HarnessError("Input audio is silent; no transcription hypotheses were published.")
    if sha256(audio_path) != audio_hash:
        raise HarnessError("Audio changed during inference.")
    if video_bundle is not None:
        video_bundle.check_unchanged()
    events = decode_events(
        timeline.finish(), torch.tensor(frame_times), tuning=metadata["openStringMidi"], capo=metadata["capoFret"],
        onset_threshold=args.onset_threshold, percussion_threshold=args.percussion_threshold,
        technique_threshold=getattr(args, "technique_threshold", .5),
        connection_threshold=getattr(args, "connection_threshold", .5),
        grace_threshold=getattr(args, "grace_threshold", .5),
    )
    report = {
        "schemaVersion": 1, "kind": "fingerstyle-transcription-hypotheses", "visibility": "private", "distributionAuthorized": False,
        "audioSha256": audio_hash, "checkpointSha256": sha256(Path(args.checkpoint)),
        "audioDurationSeconds": duration,
        "modelArchitectureVersion": identity.get("model", {}).get("architecture_version", 1),
        "metadata": metadata, "timeUnit": "input-audio-seconds", "notatedDurationUnit": "quarter-note",
        "gpWriterImplemented": True, "gpWrittenByThisCommand": False, "modelTrainingPerformedByThisCommand": False,
        **events,
    }
    if "video" in identity:
        report["pairedVideo"] = {"provided": video_bundle is not None, "inputIdentity": video_bundle.identity if video_bundle is not None else None, "audioOnlyFallback": video_bundle is None, "architectureVersion": model.video_config.architecture_version, "inputSchemaVersion": model.video_config.input_schema_version, "featureDimension": model.video_config.structured_dim, "audioOnlyPolicy": "jointly-learned-audio-path"}
    publish_json(private_output(args.output, args.data_root), report)
    print({"notes": len(report["notes"]), "percussionHypotheses": len(report["percussion"]), "gpWritten": False})
    return report


def export_gp(args):
    from .draft_cleanup import DraftProfile
    from .gp_output import write_gp_outputs

    predictions_path = Path(args.predictions).resolve()
    predictions = read_json(predictions_path)
    beat_evidence = read_json(Path(args.beat_evidence).resolve()) if args.beat_evidence else None
    full_path = private_output(args.full_output, args.data_root)
    single_path = private_output(args.single_output, args.data_root)
    if full_path.suffix.lower() != ".gp" or single_path.suffix.lower() != ".gp" or full_path == single_path:
        raise HarnessError("GP outputs require two different .gp paths under the private runs directory.")
    before = sha256(predictions_path)
    profile = DraftProfile(
        note_threshold=args.draft_note_threshold,
        percussion_threshold=args.draft_percussion_threshold,
        thumb_slap_threshold=args.thumb_slap_threshold,
        harmonic_threshold=args.draft_harmonic_threshold,
        include_harmonics=args.include_harmonics,
        brush_threshold=args.brush_threshold,
        arpeggio_threshold=args.arpeggio_threshold,
        pick_stroke_threshold=args.pick_stroke_threshold,
        rasgueado_threshold=args.rasgueado_threshold,
        brush_membership_threshold=args.brush_membership_threshold,
        arpeggio_membership_threshold=args.arpeggio_membership_threshold,
        pick_stroke_membership_threshold=args.pick_stroke_membership_threshold,
        rasgueado_membership_threshold=args.rasgueado_membership_threshold,
        connection_threshold=args.connection_threshold,
        note_technique_threshold=args.note_technique_threshold,
        grace_threshold=args.grace_threshold,
        chord_tolerance_seconds=args.chord_tolerance,
        same_string_gap_seconds=args.same_string_gap,
        strict_note_confidence=args.strict_note_confidence,
        rhythm_policy=args.rhythm_policy,
    )
    report = write_gp_outputs(
        args.template, predictions, full_path, single_path, profile=profile,
        beat_evidence=beat_evidence, include_unsupported_tail=args.include_beat_unsupported_tail,
        progress=log_progress,
    )
    if sha256(predictions_path) != before:
        raise HarnessError("Prediction hypotheses changed during GP export.")
    report.update(
        visibility="private",
        predictionsSha256=before,
        fullOutputSha256=sha256(full_path),
        singleOutputSha256=sha256(single_path),
    )
    report_path = private_output(args.report, args.data_root)
    publish_json(report_path, report)
    log_progress(f"GP export: report saved: {report_path}")
    print({
        "fullVoices": str(full_path),
        "singleVoice": str(single_path),
        "measures": report["fullVoices"]["measureCount"],
        "report": str(report_path),
    })
    return report


def analyze_beats(args):
    from .beat_tracking import track_beats

    audio = Path(args.audio).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    audio_hash = sha256(audio)
    checkpoint_hash = sha256(checkpoint)
    report = track_beats(audio, checkpoint, device=args.device)
    if sha256(audio) != audio_hash or sha256(checkpoint) != checkpoint_hash:
        raise HarnessError("Beat-analysis inputs changed before publication.")
    output = private_output(args.output, args.data_root)
    publish_json(output, report)
    print({"beats": report["beatCount"], "downbeats": report["downbeatCount"], "output": str(output)})
    return report


DRAFT_CLI_NAMES = {"note_threshold": "draft-note-threshold", "percussion_threshold": "draft-percussion-threshold",
                   "harmonic_threshold": "draft-harmonic-threshold", "chord_tolerance_seconds": "chord-tolerance",
                   "same_string_gap_seconds": "same-string-gap"}


def draft_cli_values(args):
    from .draft_cleanup import DraftProfile

    values = {field.name: getattr(args, DRAFT_CLI_NAMES.get(field.name, field.name.replace("_", "-")).replace("-", "_"))
              for field in fields(DraftProfile)}
    DraftProfile(**values)
    return {DRAFT_CLI_NAMES.get(name, name.replace("_", "-")): value for name, value in values.items()}


def add_draft_arguments(parser):
    from .draft_cleanup import DraftProfile

    for name, value in asdict(DraftProfile()).items():
        flag = "--" + DRAFT_CLI_NAMES.get(name, name.replace("_", "-"))
        if isinstance(value, bool):
            parser.add_argument(flag, action="store_true", help={
                "strict_note_confidence": "Apply the note threshold to every note, including technique-completed chord members.",
                "include_harmonics": "Include consistency-gated harmonic guesses; disabled by default.",
            }[name])
        elif name == "rhythm_policy":
            parser.add_argument(flag, choices=("adaptive", "fingerstyle"), default=value,
                                help="fingerstyle uses ordinary binary16ths or coarser and reserves32nds for supported downstroke figures; adaptive also allows tuplets.")
        else:
            parser.add_argument(flag, type=float, default=value)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    configure = commands.add_parser("config", help="Write a local audio configuration; add --video-index for joint audio/video training.")
    configure.add_argument("--output", default="runs/config.json")
    configure.add_argument("--video-index", help="Source-bound numeric paired index for fresh joint acoustic/video/fusion training (default acoustic architecture v4); no audio checkpoint.")
    configure.add_argument("--manifest", help="Training release manifest; when --video-index is supplied, its release identity must match.")
    for name in ("preflight", "train", "evaluate", "evaluate-events"):
        command = commands.add_parser(name)
        command.add_argument("--data-root", default=str(ROOT))
        command.add_argument("--manifest")
        if name in ("preflight", "train"):
            command.add_argument("--config")
        if name == "preflight":
            command.add_argument("--output", default="runs/preflight.json")
            command.add_argument("--forward", action="store_true", help="One untrained eval-mode forward per split; no optimizer or checkpoint.")
        elif name == "train":
            command.add_argument("--run-dir", required=True)
            command.add_argument("--resume", help="Exact resume from this run's latest.pt; never converts a historical acoustic run into a joint run.")
            command.add_argument("--max-hours", type=float, help="Optional explicit override of training.max_seconds. Default configs have no time limit. Includes setup, validation and checkpoint I/O; cooperatively pauses at safe boundaries. Native operations can overrun.")
        else:
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--split", choices=("train", "validation"), default="validation")
            command.add_argument("--device", default="auto")
            command.add_argument("--output", default="runs/event-evaluation.json" if name == "evaluate-events" else "runs/evaluation.json")
            video_evaluation = command.add_mutually_exclusive_group()
            video_evaluation.add_argument("--video-index", help="Paired evaluation using the existing release splits.")
            video_evaluation.add_argument("--audio-only", action="store_true", help="Evaluate the joint checkpoint's learned audio-only path without video.")
            if name == "evaluate-events":
                command.add_argument("--tolerances", nargs="+", type=float, default=[.05, .1, .2])
                command.add_argument("--onset-threshold", type=float, default=.5)
                command.add_argument("--percussion-threshold", type=float, default=.5)
                command.add_argument("--technique-threshold", type=float, default=.5)
                command.add_argument("--connection-threshold", type=float, default=.5)
                command.add_argument("--note-technique-threshold", type=float, default=.5)
                command.add_argument("--grace-threshold", type=float, default=.5)
    inference = commands.add_parser("infer")
    inference.add_argument("--data-root", default=str(ROOT))
    inference.add_argument("--checkpoint", default=str(ROOT / "models" / "transcriber.pt"))
    inference.add_argument("--audio", required=True)
    inference.add_argument("--metadata", required=True)
    inference.add_argument("--output", default="runs/predictions.json")
    inference.add_argument("--device", default="auto")
    inference.add_argument("--video-bundle", help="Optional source/audio-bound numeric landmarks, motion and guitar-geometry bundle for a joint checkpoint; omitted uses its jointly learned audio-only path.")
    inference.add_argument("--onset-threshold", type=float, default=.5)
    inference.add_argument("--percussion-threshold", type=float, default=.5)
    inference.add_argument("--technique-threshold", type=float, default=.5)
    inference.add_argument("--connection-threshold", type=float, default=.5)
    inference.add_argument("--grace-threshold", type=float, default=.5)
    beats = commands.add_parser("analyze-beats")
    beats.add_argument("--data-root", default=str(ROOT))
    beats.add_argument("--audio", required=True)
    beats.add_argument("--checkpoint", required=True, help="Local Beat This! checkpoint; remote downloads are never automatic.")
    beats.add_argument("--output", default="runs/beat-evidence.json")
    beats.add_argument("--device", default="cpu")
    export = commands.add_parser("export-gp")
    export.add_argument("--data-root", default=str(ROOT))
    export.add_argument("--predictions", required=True)
    export.add_argument("--template", required=True)
    export.add_argument("--beat-evidence", help="Private analyze-beats output for beat-anchored constrained rhythm inference.")
    export.add_argument("--full-output", default="runs/transcription.full-voices.gp")
    export.add_argument("--single-output", default="runs/transcription.single-voice.gp")
    export.add_argument("--report", default="runs/gp-output.json")
    add_draft_arguments(export)
    export.add_argument("--include-beat-unsupported-tail", action="store_true", help="Keep attacks after the final detected beat using extrapolated timing; raw JSON always retains them.")
    from .transcription_pipeline import add_commands

    add_commands(commands)
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv)
    try:
        if args.command == "config":
            log_progress("Configuration: creating defaults...")
            config = default_config()
            if args.video_index:
                from .paired_video import index_manifest
                from .transcriber_video import VideoConfig

                log_progress(f"Configuration: validating paired index {args.video_index}...")
                manifest = index_manifest(args.video_index, manifest_path=args.manifest, default_manifest=config["data"]["manifest"])
                config["data"]["manifest"] = str(manifest.relative_to(ROOT))
                config["video"] = {"index": args.video_index, "model": asdict(VideoConfig())}
                log_progress("Configuration: paired index validated.")
            elif args.manifest:
                config["data"]["manifest"] = args.manifest
            output_path = private_output(args.output)
            log_progress(f"Configuration: writing {output_path}...")
            publish_json(output_path, config)
            log_progress("Configuration: completed.")
        elif args.command in ("transcribe", "transcribe-review", "transcribe-status"):
            import json
            from .transcription_pipeline import review_transcription, run_transcription, transcription_status

            result = {"transcribe": run_transcription, "transcribe-review": review_transcription, "transcribe-status": transcription_status}[args.command](args)
            print(json.dumps(result, indent=2))
            if result["status"] == "blocked":
                return 1
        else:
            {"preflight": preflight, "train": train, "evaluate": evaluate, "evaluate-events": evaluate_decoded_events, "infer": infer, "analyze-beats": analyze_beats, "export-gp": export_gp}[args.command](args)
    except (HarnessError, OSError, ValueError, RuntimeError) as error:
        print(f"Transcriber error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
