"""Resumable batch orchestration of existing canonical and paired-video preparation."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import partial
import os
import math
from pathlib import Path
import subprocess
import sys

from .dataset_io import ROOT, publish_json, read_json, sha256
from .dataset_release import candidate_digest, validate_release
from .paired_video import PairedVideoIndex, build_index
from .prepare_training_data import regular_path, safe_id, workspace_path
from .score_alignment import ScoreClock
from .technique_supervision import TECHNIQUE_TYPES, projected_techniques
from .training_windows import projected_targets


MANIFEST_KIND = "paired-preparation-batch"
REUSE_STAGES = {"alignment", "shots", "hands", "roles", "fretboard", "bundle"}


def _path(value, root, label, *, existing=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires a path.")
    path = Path(value)
    path = regular_path(path if path.is_absolute() else Path(root) / path)
    if existing and not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def load_batch(path, *, root=ROOT):
    path = regular_path(path)
    raw = read_json(path)
    required = {"schemaVersion", "kind", "records"}
    optional = {"workspace", "releaseManifest", "releaseVersion", "acceptConventions", "ffmpegDirectory", "videoPython", "handModel", "poseModel", "fretboardModel", "fretboardDevice", "reviewMode", "reviewBudget", "workers"}
    if isinstance(raw, dict) and "imageSize" in raw:
        raise ValueError("RGB crop inputs are no longer supported. Remove imageSize and prepare new structure-only bundles.")
    if not isinstance(raw, dict) or not required <= raw.keys() or raw.keys() - required - optional or type(raw["schemaVersion"]) is not int or raw["schemaVersion"] != 1 or raw["kind"] != MANIFEST_KIND:
        raise ValueError("Expected version1 paired-preparation-batch with explicit records.")
    if ("releaseManifest" in raw) == ("releaseVersion" in raw):
        raise ValueError("Supply one existing releaseManifest or one new releaseVersion.")
    result = dict(raw)
    result["reviewMode"] = raw.get("reviewMode", "automatic")
    if result["reviewMode"] not in ("automatic", "manual"):
        raise ValueError("reviewMode must be automatic or manual.")
    result["reviewBudget"] = raw.get("reviewBudget", 12)
    if type(result["reviewBudget"]) is not int or result["reviewBudget"] < 0:
        raise ValueError("reviewBudget must be a nonnegative integer number of optional shot reviews.")
    result["workers"] = raw.get("workers", 16)
    if type(result["workers"]) is not int or not 1 <= result["workers"] <= 16:
        raise ValueError("workers must be an integer from 1 to 16.")
    result["workspace"] = str(workspace_path(_path(raw.get("workspace", "data"), root, "workspace")))
    if not Path(result["workspace"]).is_relative_to(Path(root).resolve()):
        raise ValueError("Paired batch workspace must be inside the project private root so the existing paired loader can bind its release.")
    result["acceptConventions"] = raw.get("acceptConventions", False)
    if type(result["acceptConventions"]) is not bool:
        raise ValueError("acceptConventions must be an explicit boolean.")
    if "releaseManifest" in raw:
        result["releaseManifest"] = str(_path(raw["releaseManifest"], root, "release manifest", existing=True))
    else:
        safe_id(raw["releaseVersion"])
    result["videoPython"] = str(_path(raw.get("videoPython", os.environ.get("VIDEO_PYTHON", "scripts\\video-evidence\\.venv\\Scripts\\python.exe")), root, "video Python", existing=True))
    result["handModel"] = str(_path(raw.get("handModel"), root, "hand model"))
    pose = raw.get("poseModel")
    result["poseModel"] = str(_path(pose, root, "pose model")) if pose is not None else None
    fretboard = raw.get("fretboardModel")
    result["fretboardModel"] = str(_path(fretboard, root, "fretboard model")) if fretboard is not None else None
    result["fretboardDevice"] = raw.get("fretboardDevice", "cpu")
    if not isinstance(result["fretboardDevice"], str) or not result["fretboardDevice"].strip():
        raise ValueError("fretboardDevice must be a nonempty Ultralytics device string.")
    result["ffmpegDirectory"] = str(_path(raw["ffmpegDirectory"], root, "FFmpeg directory")) if raw.get("ffmpegDirectory") else None
    if not isinstance(raw["records"], list) or not raw["records"]:
        raise ValueError("Batch records must be a nonempty list.")
    records, identifiers, groups = [], set(), {}
    for record in raw["records"]:
        fields = {"id", "groupId", "split", "gp", "video", "pluckingScreenSide"}
        extras = {"audio", "title", "clips", "reuse", "voiceSupervisionPolicy"}
        if not isinstance(record, dict) or not fields <= record.keys() or record.keys() - fields - extras:
            raise ValueError("Each batch record requires id, groupId, split, gp, trimmed local video and pluckingScreenSide; audio is optional.")
        identifier = safe_id(record["id"])
        if identifier.casefold() in identifiers:
            raise ValueError("Batch IDs must be unique, including case.")
        identifiers.add(identifier.casefold())
        group, split = record["groupId"], record["split"]
        if not isinstance(group, str) or not group.strip() or split not in ("train", "validation"):
            raise ValueError("Each record needs an explicit nonempty group and train/validation split.")
        if group in groups and groups[group] != split:
            raise ValueError("A related recording group cannot cross training and validation.")
        groups[group] = split
        row = dict(record)
        for name in ("gp", "video", *(("audio",) if "audio" in record else ())):
            row[name] = str(_path(record[name], root, f"{identifier} {name}", existing=True))
        media = [row[name] for name in ("gp", "audio", "video") if name in row]
        if Path(row["gp"]).suffix.lower() != ".gp" or len(set(media)) != len(media):
            raise ValueError("Supply distinct original GP, trimmed audio and video files.")
        side = row["pluckingScreenSide"]
        if side not in ("left", "right"):
            raise ValueError("pluckingScreenSide must explicitly be left or right.")
        row["pluckingScreenSide"] = side
        if row.get("voiceSupervisionPolicy") not in (
            None, "native-multivoice", "intentional-single-voice", "flattened-or-unknown",
        ):
            raise ValueError("voiceSupervisionPolicy is unsupported.")
        if "clips" in row:
            previous = None
            if not isinstance(row["clips"], list) or not row["clips"]:
                raise ValueError("Optional clips must be nonempty source-PTS pairs.")
            for clip in row["clips"]:
                if not isinstance(clip, list) or len(clip) != 2 or any(type(value) is not int for value in clip) or not 0 <= clip[0] < clip[1] or previous is not None and clip[0] < previous:
                    raise ValueError("Clips require ordered nonoverlapping integer source PTS.")
                previous = clip[1]
        reuse = row.get("reuse", {})
        if not isinstance(reuse, dict) or reuse.keys() - REUSE_STAGES:
            raise ValueError("Unknown reusable video stage.")
        row["reuse"] = {name: str(_path(value, root, f"{identifier} reusable {name}", existing=True)) for name, value in reuse.items()}
        records.append(row)
    result["records"] = records
    return result


def _output_directory(path, root, *, create=True):
    directory = _path(str(path), root, "batch output")
    if not directory.is_relative_to(Path(root).resolve() / "runs" / "video-evidence"):
        raise ValueError("Batch output must stay under private runs\\video-evidence.")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _media_inputs(batch, directory, *, create):
    from .local_media import prepare_soundtrack

    records = []
    for record in batch["records"]:
        row = dict(record)
        if "audio" not in row:
            audio, receipt = prepare_soundtrack(
                row["video"], directory / "source-cache",
                ffmpeg_dir=batch["ffmpegDirectory"], create=create,
            )
            row.update(audio=str(audio), audioOrigin=str(receipt))
        records.append(row)
    return {**batch, "records": records}


@contextmanager
def _lock(directory):
    path = directory / ".batch.lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError(f"Batch is already running or its prior process was interrupted. Inspect {path} and remove only that stale lock after confirming its PID has exited.") from error
    try:
        with stream:
            stream.write(str(os.getpid()))
            stream.flush()
            yield
    finally:
        path.unlink(missing_ok=True)


def _source_hashes(batch):
    hashes = {record["id"]: {name: sha256(record[name]) for name in ("gp", "audio", "video") if name in record} for record in batch["records"]}
    for record in batch["records"]:
        if record.get("audioOrigin"):
            hashes[record["id"]]["audioOrigin"] = sha256(record["audioOrigin"])
    video_groups = {}
    for record in batch["records"]:
        digest = hashes[record["id"]]["video"]
        identity = record["groupId"], record["split"]
        if digest in video_groups and video_groups[digest] != identity:
            raise ValueError("Records sharing identical source video must remain in the same relationship group and split.")
        video_groups[digest] = identity
    return hashes


def _identity(manifest_path, batch):
    identity = {"manifestSha256": sha256(manifest_path), "configurationSha256": candidate_digest(batch), "sources": _source_hashes(batch)}
    if batch.get("releaseManifest"):
        identity["releaseManifestSha256"] = sha256(batch["releaseManifest"])
    return identity


def _bind_state(directory, manifest_path, batch):
    path = directory / "batch-state.json"
    identity = _identity(manifest_path, batch)
    if path.exists():
        state = read_json(regular_path(path))
        if state.get("identity") != identity or state.get("kind") != "paired-preparation-state":
            raise ValueError("Batch configuration or source files changed. Use a new output directory; completed artifacts are not silently replaced.")
    else:
        state = {"schemaVersion": 1, "kind": "paired-preparation-state", "identity": identity, "records": {}, "status": "pending"}
        publish_json(path, state)
    return state


@contextmanager
def _running_state(directory, manifest_path, batch):
    state = _bind_state(directory, manifest_path, batch)
    state["status"] = "running"
    publish_json(directory / "batch-state.json", state)
    try:
        yield state
    except (OSError, ValueError) as error:
        state["status"] = "blocked"
        state["actions"] = [{"stage": "batch-error", "reason": str(error), "resume": "Resolve the reported input/stage problem, then rerun the same batch. Changed source/configuration requires a new output directory."}]
        _summary(directory, state)
        raise


def _worker(request_path, result_path, video_python, *, review_flags=None, runner=subprocess.run):
    command = [video_python, str(ROOT / "scripts" / "video-evidence" / "batch_video.py"),
               "review" if review_flags is not None else "run", "--request", str(request_path), "--output", str(result_path)]
    if review_flags is not None:
        command.extend(review_flags)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    result = runner(command, check=False, env=environment)
    if result.returncode:
        raise ValueError(f"Video preparation failed with exit code {result.returncode}; resolve the reported stage and rerun the same batch.")
    if not result_path.is_file():
        raise ValueError("Video worker did not publish its result.")
    document = read_json(result_path)
    if document.get("kind") != "paired-video-preparation-result" or document.get("status") not in ("ready", "needs-review", "blocked"):
        raise ValueError("Video worker returned an unsupported result.")
    return document


def _request(batch, source, canonical, directory):
    result = {
        "schemaVersion": 1, "kind": "paired-video-preparation-request", "id": source["id"],
        "video": source["video"], "audio": canonical["audio"],
        "outputDirectory": str(directory / "video"),
        "reviewMode": batch["reviewMode"],
        "pluckingScreenSide": source["pluckingScreenSide"], "reuse": source["reuse"],
        "handModel": batch["handModel"], "poseModel": batch["poseModel"],
        "fretboardModel": batch["fretboardModel"], "fretboardDevice": batch["fretboardDevice"],
    }
    if "clips" in source:
        result["clips"] = source["clips"]
    return result


def paired_coverage(manifest_path, index_path, *, root=ROOT):
    manifest, records, bindings = validate_release(manifest_path)
    index = PairedVideoIndex(index_path, bindings[Path(manifest_path)], records, root=root)
    result = {split: {
        "recordings": 0, "pairedWindows": 0, "usableVideoWindows": 0, "gpPositiveOccurrences": {},
        "positiveOccurrencesWithUsableVideo": {}, "positiveOccurrencesWithUnassignedHands": {}, "records": [], "visualCoverage": {
            "nativeFrames": 0, "framesWithGuitarGeometry": 0, "framesWithIndependentHands": 0,
            "framesWithIndependentHandsWithoutGeometry": 0, "framesWithUnassignedHands": 0,
            "recordingsWithGuitarGeometry": 0, "recordingsWithIndependentHands": 0,
            "framesWithCoarseContext": 0, "framesWithBodyRelativeContext": 0, "recordingsWithCoarseContext": 0,
        },
    } for split in ("train", "validation")}
    names = ("wrist_thump", "thumb_slap", "percussive_hit", *TECHNIQUE_TYPES)
    totals = {split: tuple(Counter({name: 0 for name in names}) for _ in range(3)) for split in result}
    for entry, payload in records:
        identifier, split = entry["id"], entry["split"]
        if identifier not in index.bundles:
            continue
        labels = payload["canonical"]
        clock = ScoreClock(labels, payload["normalization"])
        _, gestures = projected_targets(labels, payload["candidate"], clock)
        techniques = projected_techniques(labels, payload["candidate"], clock)
        events = [(event["technique"], event["proposedOnsetClipSeconds"]) for event in gestures if event["technique"] in names and event["onsetTimingKnownInScore"]]
        events.extend((kind, event["proposedOnsetClipSeconds"]) for event in techniques for kind in event["techniques"] if event["scoreOnsetKnown"])
        approved = payload["approval"]["approvedClipRanges"]
        events = [(name, time) for name, time in events if time is not None and any(left <= time < right for left, right in approved)]
        times = sorted({time for _, time in events})
        availability, technique_available = index.availability_at(identifier, times)
        by_time = {time: bool(availability[i, 1] and technique_available[i]) for i, time in enumerate(times)}
        anonymous_by_time = {time: bool(availability[i, 2:].any() and technique_available[i]) for i, time in enumerate(times)}
        counts, usable, anonymous = (Counter({name: 0 for name in names}) for _ in range(3))
        for name, time in events:
            counts[name] += 1
            usable[name] += int(by_time[time])
            anonymous[name] += int(anonymous_by_time[time])
        bundle = index.bundles[identifier]
        intervals = bundle.clip_intervals
        masks = bundle.arrays["structured_available"]
        geometry_frames = masks[..., :98].any(axis=(1, 2))
        hand_frames = masks[..., 98:140].any(axis=(1, 2))
        unassigned_frames = masks[:, 2:, 98:140].any(axis=(1, 2))
        coarse_frames = masks[..., 186:194].any(axis=(1, 2))
        visual = {
            "nativeFrames": len(masks), "framesWithGuitarGeometry": int(geometry_frames.sum()),
            "framesWithIndependentHands": int(hand_frames.sum()),
            "framesWithIndependentHandsWithoutGeometry": int((hand_frames & ~geometry_frames).sum()),
            "framesWithUnassignedHands": int(unassigned_frames.sum()),
            "recordingsWithGuitarGeometry": int(geometry_frames.any()),
            "recordingsWithIndependentHands": int(hand_frames.any()),
            "framesWithCoarseContext": int(coarse_frames.sum()),
            "framesWithBodyRelativeContext": int(masks[..., 192].any(axis=1).sum()),
            "recordingsWithCoarseContext": int(coarse_frames.any()),
        }
        for name, value in visual.items():
            result[split]["visualCoverage"][name] += value
        windows = sum(any(
            left < row["stopSampleExclusive"] / entry["sampleRate"] and right > row["startSample"] / entry["sampleRate"]
            for left, right in intervals
        ) for row in payload["windows"])
        usable_windows = sum(index.has_usable(identifier, row["startSample"] / entry["sampleRate"], row["stopSampleExclusive"] / entry["sampleRate"]) for row in payload["windows"])
        result[split]["recordings"] += 1
        result[split]["pairedWindows"] += windows
        result[split]["usableVideoWindows"] += usable_windows
        result[split]["records"].append({"id": identifier, "pairedWindows": windows, "usableVideoWindows": usable_windows,
                                        "visualCoverage": visual, "gpPositiveOccurrences": dict(counts),
                                        "positiveOccurrencesWithUsableVideo": dict(usable), "positiveOccurrencesWithUnassignedHands": dict(anonymous)})
        totals[split][0].update(counts)
        totals[split][1].update(usable)
        totals[split][2].update(anonymous)
    for split in result:
        result[split]["gpPositiveOccurrences"] = dict(totals[split][0])
        result[split]["positiveOccurrencesWithUsableVideo"] = dict(totals[split][1])
        result[split]["positiveOccurrencesWithUnassignedHands"] = dict(totals[split][2])
    return {
        "splits": result, "missingUsableTrainingPositiveClasses": [name for name in names if totals["train"][1][name] == 0],
        "interpretation": "Unique projected positive GP occurrences inside approved audio ranges; usable means an aligned structured plucking observation outside local mismatch masks. Not an accuracy score or musical acceptance gate; no overlap-window double counting.",
        "nativeRasgueadoPolicy": "Only explicit native markers counted; physical a-m-i patterns are not silently relabeled.",
        "geometryPolicy": "Geometry and coarse-context slots are unavailable. Only independent-hand evidence is prepared; unknown roles stay unassigned.",
    }


def _summary(directory, state):
    report = {
        "schemaVersion": 1, "kind": "paired-preparation-summary", "visibility": "private",
        "status": state["status"], "trainingPerformed": False,
        "manifestPath": state.get("manifestPath"), "indexPath": state.get("indexPath"),
        "records": state.get("records", {}), "actions": state.get("actions", []),
        "coverage": state.get("coverage"), "identity": state["identity"],
        "optionalReviewCount": state.get("optionalReviewCount", 0),
        "readinessPolicy": "Ready means existing-loader compatible inputs, not sufficient class coverage or musical quality. Optional shot review is bounded. No LLM or training is invoked.",
    }
    lines = [f"Batch status: {state['status']}", report["readinessPolicy"], ""]
    for action in report["actions"]:
        label = "OPTIONAL" if action.get("optional") else "REQUIRED"
        lines.append(f"{label} {action.get('id', 'batch')} / {action.get('stage', 'review')}: {action.get('reason', '')}")
        command = action.get("command")
        if isinstance(command, list) and command:
            lines.append("& " + " ".join("'" + str(value).replace("'", "''") + "'" for value in command))
        for key in ("path", "reportPath", "notationPath", "suggestedRanges"):
            if key in action:
                lines.append(f"{key}: {action[key]}")
        lines.append("")
    if report["coverage"]:
        for split, values in report["coverage"]["splits"].items():
            lines.append(f"{split}: {values['recordings']} recordings, {values['pairedWindows']} prepared paired windows, {values['usableVideoWindows']} with usable visual observations")
            lines.append(f"Positive occurrences with known plucking-hand evidence: {values['positiveOccurrencesWithUsableVideo']}")
            lines.append(f"Positive occurrences with unassigned-hand evidence (not proof of plucking): {values['positiveOccurrencesWithUnassignedHands']}")
            lines.append(f"Guitar-relative versus independent-hand evidence: {values['visualCoverage']}")
        lines.append("Training classes without known-plucking positive evidence: " + ", ".join(report["coverage"]["missingUsableTrainingPositiveClasses"]))
    if report["optionalReviewCount"]:
        lines.append(f"Optional shot reviews: {report['optionalReviewCount']}; only the configured bounded shortlist is shown above.")
    if report["indexPath"]:
        lines.append(f"Paired input index: {report['indexPath']}")
    next_actions = directory / "next-actions.txt"
    from .prepare_training_data import write_bytes

    write_bytes(next_actions, ("\n".join(lines) + "\n").encode("utf-8"))
    report["nextActionsPath"] = str(next_actions)
    publish_json(directory / "summary.json", report)
    publish_json(directory / "batch-state.json", state)
    return report


def run_batch(manifest_path, output_directory, *, root=ROOT, runner=subprocess.run, progress=partial(print, flush=True)):
    progress("Loading batch manifest...")
    manifest_path = _path(str(manifest_path), root, "batch manifest", existing=True)
    batch = load_batch(manifest_path, root=root)
    directory = _output_directory(output_directory, root)
    progress(f"Loaded {len(batch['records'])} records | workers {batch['workers']}. Verifying source hashes and saved job identity...")
    with _lock(directory):
        state_path = regular_path(directory / "batch-state.json")
        if state_path.is_file():
            identity = read_json(state_path).get("identity", {})
            sources = _source_hashes(batch)
            saved = identity.get("sources", {})
            if identity.get("manifestSha256") != sha256(manifest_path) or any(
                any(saved.get(identifier, {}).get(key) != digest for key, digest in hashes.items())
                for identifier, hashes in sources.items()
            ):
                raise ValueError("Batch configuration or source files changed. Use a new output directory.")
        batch = _media_inputs(batch, directory, create=True)
        return _run_prepared_batch(manifest_path, directory, batch, root=root, runner=runner, progress=progress)


def _run_prepared_batch(manifest_path, directory, batch, *, root, runner, progress):
    from .batch_canonical import ensure_canonical

    with _running_state(directory, manifest_path, batch) as state:
        progress("Source hashes and job identity verified.")
        progress("Canonical GP/audio preparation or frozen-release reuse...")
        canonical = ensure_canonical(batch, root=root)
        state["actions"] = canonical.get("actions", [])
        if canonical["status"] != "ready":
            state["status"] = canonical["status"]
            progress(f"Canonical preparation: {state['status']}; saving required actions.")
            return _summary(directory, state)
        progress("Canonical GP/audio inputs ready.")
        state["manifestPath"] = canonical["manifestPath"]
        available = {row["id"]: row for row in canonical["records"]}
        bundles, actions = [], []

        def prepare_record(record):
            identifier = record["id"]
            progress(f"{identifier}: preparing or resuming video inputs...")
            working = directory / identifier
            working.mkdir(exist_ok=True)
            request_path, result_path = working / "request.json", working / "result.json"
            request = _request(batch, record, available[identifier], working)
            if request_path.exists() and read_json(request_path) != request:
                raise ValueError(f"{identifier}: stage request changed; use a new batch output.")
            if not request_path.exists():
                publish_json(request_path, request)
            if batch["workers"] == 1:
                result = _worker(request_path, result_path, batch["videoPython"], runner=runner)
            else:
                progress(f"{identifier}: live stage/frame logs: {working / 'worker.log'}")
                with (working / "worker.log").open("a", encoding="utf-8") as log:
                    def logged_runner(command, **kwargs):
                        return runner(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)
                    result = _worker(request_path, result_path, batch["videoPython"], runner=logged_runner)
            if result.get("id") != identifier:
                raise ValueError("Video worker returned a different recording identity.")
            expected = {"video": sha256(record["video"]), "audio": sha256(available[identifier]["audio"])}
            if result.get("inputSha256") != expected:
                raise ValueError("Video worker input identity differs from canonical sources.")
            return identifier, result

        def completed_record(identifier, result):
            state["records"][identifier] = result
            for action in result.get("actions", []):
                action = {**action, "id": identifier}
                actions.append(action)
            if result["status"] == "ready":
                bundle = result.get("artifacts", {}).get("bundle")
                if not isinstance(bundle, str):
                    raise ValueError("Ready video stage did not provide a paired-input bundle.")
                bundles.append(bundle)
            required = [action for action in actions if not action.get("optional")]
            optional = sorted((action for action in actions if action.get("optional")), key=lambda action: action.get("priority", 0), reverse=True)
            state["actions"] = required + optional[:batch["reviewBudget"]]
            state["optionalReviewCount"] = len(optional)
            _summary(directory, state)
            progress(f"{identifier}: {result['status']} ({len(state['records'])}/{len(batch['records'])} records recorded)")

        if batch["workers"] == 1:
            for record in batch["records"]:
                completed_record(*prepare_record(record))
        else:
            groups = {}
            for record in batch["records"]:
                key = state["identity"]["sources"][record["id"]]["video"]
                groups.setdefault(key, []).append(record)
            pending_groups = iter(groups.values())
            with ThreadPoolExecutor(max_workers=batch["workers"]) as executor:
                active = {}
                for _ in range(batch["workers"]):
                    group = next(pending_groups, None)
                    if group:
                        active[executor.submit(prepare_record, group[0])] = group[1:]
                while active:
                    future = next(as_completed(active))
                    remaining = active.pop(future)
                    try:
                        completed_record(*future.result())
                    except (OSError, ValueError):
                        for pending in active:
                            pending.cancel()
                        raise
                    group = remaining or next(pending_groups, None)
                    if group:
                        active[executor.submit(prepare_record, group[0])] = group[1:]
        bundles = [state["records"][record["id"]]["artifacts"]["bundle"] for record in batch["records"]
                   if state["records"].get(record["id"], {}).get("status") == "ready"]
        if len(bundles) != len(batch["records"]):
            state["status"] = "needs-review" if all(row["status"] != "blocked" for row in state["records"].values()) else "blocked"
            progress(f"Video preparation: {state['status']}; saving required actions.")
            return _summary(directory, state)
        index_path = directory / "paired-index.json"
        progress("Verifying existing paired index..." if index_path.exists() else "Building paired index...")
        if index_path.exists():
            _, records, bindings = validate_release(Path(state["manifestPath"]))
            index = PairedVideoIndex(index_path, bindings[Path(state["manifestPath"])], records, root=root)
            if {key: value.identity["bundleSha256"] for key, value in index.bundles.items()} != {read_json(Path(path))["id"]: sha256(path) for path in bundles}:
                raise ValueError("Completed index no longer matches this batch's bundles; use a new output.")
        else:
            build_index(Path(state["manifestPath"]), bundles, index_path, root=root)
        state["indexPath"] = str(index_path)
        progress(f"Paired index ready: {index_path}")
        progress("Counting source targets with usable paired visual coverage...")
        state["coverage"] = paired_coverage(Path(state["manifestPath"]), index_path, root=root)
        coverage = state["coverage"]["splits"]
        progress("Paired visual/target coverage counted.")
        state["status"] = "ready" if all(row["pairedWindows"] > 0 for row in coverage.values()) and coverage["train"]["usableVideoWindows"] > 0 else "needs-coverage"
        if state["status"] == "needs-coverage":
            state["actions"].append({"stage": "coverage", "reason": "Both original splits need prepared paired windows and training needs some usable visual observations. All-masked geometry AND hand evidence is not usable joint-training input. Review a bounded set of useful proposals or improve preparation; do not move related recordings across splits."})
        progress("Verifying final source identities...")
        if state["identity"]["sources"] != _source_hashes(batch) or state["identity"]["manifestSha256"] != sha256(manifest_path):
            raise ValueError("Batch source files changed while preparing inputs.")
        report = _summary(directory, state)
        progress(f"Batch {state['status']}; summary saved: {directory / 'summary.json'}")
        return report


def batch_status(manifest_path, output_directory, *, root=ROOT):
    manifest_path = _path(str(manifest_path), root, "batch manifest", existing=True)
    batch = load_batch(manifest_path, root=root)
    directory = _output_directory(output_directory, root, create=False)
    state_path = regular_path(directory / "batch-state.json")
    if not state_path.is_file():
        return {"status": "pending", "trainingPerformed": False, "actions": [{
            "stage": "start", "reason": "Run this batch to prepare local sources.",
            "command": [sys.executable, "-m", "scripts.prepare_training_data", "batch", "--manifest", str(manifest_path), "--output-directory", str(directory)],
        }]}
    batch = _media_inputs(batch, directory, create=False)
    state = read_json(state_path)
    if state.get("kind") != "paired-preparation-state" or state.get("identity") != _identity(manifest_path, batch):
        raise ValueError("Batch configuration or source files changed. Use a new output directory.")
    if state["status"] == "ready":
        manifest, records, bindings = validate_release(Path(state["manifestPath"]))
        PairedVideoIndex(Path(state["indexPath"]), bindings[Path(state["manifestPath"])], records, root=root)
    return {**state, "trainingPerformed": False, "nextActionsPath": str(directory / "next-actions.txt")}


def review_batch(manifest_path, output_directory, identifier, *, reviewer=None, accept_score=False, ranges=(), anchors=(), exclude_ranges=(), acknowledge_uncertainty=False, percussion_complete=False, video_flags=None, root=ROOT, runner=subprocess.run):
    from .batch_canonical import review_canonical

    manifest_path = _path(str(manifest_path), root, "batch manifest", existing=True)
    batch = load_batch(manifest_path, root=root)
    if identifier not in {row["id"] for row in batch["records"]}:
        raise ValueError("Review ID is not in this batch.")
    if sum((bool(accept_score), bool(video_flags))) != 1:
        raise ValueError("Choose exactly one score or video review action.")
    directory = _output_directory(output_directory, root)
    with _lock(directory):
        batch = _media_inputs(batch, directory, create=False)
        state = _bind_state(directory, manifest_path, batch)
        if accept_score:
            result = review_canonical(batch, identifier, reviewer=reviewer, ranges=list(ranges), anchors=list(anchors), exclude_ranges=list(exclude_ranges), acknowledge_uncertainty=acknowledge_uncertainty, percussion_complete=percussion_complete, accept=True, root=root)
        else:
            request_path, result_path = directory / identifier / "request.json", directory / identifier / "result.json"
            if not request_path.is_file():
                raise ValueError("Run the batch first to prepare source-bound review assets.")
            result = _worker(request_path, result_path, batch["videoPython"], review_flags=video_flags, runner=runner)
        state["status"] = "review-updated-rerun-batch"
        _summary(directory, state)
        return result


def finalize_batch(manifest_path, output_directory, reviewer, *, root=ROOT):
    from .batch_canonical import finalize_canonical

    manifest_path = _path(str(manifest_path), root, "batch manifest", existing=True)
    batch = load_batch(manifest_path, root=root)
    directory = _output_directory(output_directory, root)
    with _lock(directory):
        batch = _media_inputs(batch, directory, create=False)
        state = _bind_state(directory, manifest_path, batch)
        result = finalize_canonical(batch, reviewer=reviewer, root=root)
        state["status"] = "release-published-rerun-batch"
        _summary(directory, state)
        return result


def train_batch(manifest_path, output_directory, *, epochs=3, max_steps=None, max_hours=None, cpu_threads=None, video_dropout=.2, device="auto", root=ROOT, runner=subprocess.run, train_command=None):
    """Prepare paired inputs, then execute one explicitly invoked joint training run."""
    from . import transcriber
    from .transcriber_runtime import load_checkpoint
    from .transcriber_video import VideoConfig
    from dataclasses import asdict

    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer or omitted.")
    if max_hours is not None and (type(max_hours) not in (int, float) or not math.isfinite(max_hours) or max_hours <= 0):
        raise ValueError("max_hours must be positive and finite, or omitted for no time limit.")
    video_config = VideoConfig(modality_dropout=video_dropout)
    if cpu_threads is None:
        cpu_threads = transcriber.default_config()["data"]["num_threads"]
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 32:
        raise ValueError("cpu_threads must be an integer from 1 to 32.")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Training device must be auto, cpu or cuda.")
    if train_command is None:
        train_command = transcriber.main
    prepared = run_batch(manifest_path, output_directory, root=root, runner=runner)
    if prepared["status"] != "ready":
        return {**prepared, "requestedAction": "train", "trainingStarted": False, "reason": "Complete the explicit preparation/review actions, then repeat batch-train."}
    transcriber.log_progress("Batch training: preparation ready; verifying training inputs and settings...")
    directory = _output_directory(output_directory, root)
    manifest_hash, index_hash = sha256(prepared["manifestPath"]), sha256(prepared["indexPath"])
    batch = load_batch(_path(str(manifest_path), root, "batch manifest", existing=True), root=root)
    index = read_json(Path(prepared["indexPath"]))
    if {row["id"] for row in index["records"]} != {row["id"] for row in batch["records"]}:
        raise ValueError("The paired index must contain exactly this batch's selected recordings.")
    settings = {
        "manifestSha256": manifest_hash, "indexSha256": index_hash,
        "epochs": epochs, "maxSteps": max_steps, "videoDropout": video_dropout, "device": device,
        "trainingMode": "joint",
        "cpuThreads": cpu_threads,
    }
    with _lock(directory):
        state_path = directory / "training-state.json"
        if state_path.exists():
            state = read_json(regular_path(state_path))
            if state.get("kind") != "joint-paired-batch-training" or state.get("settings") != settings:
                raise ValueError("This training job's inputs/settings changed. Use a new job directory, or the trainer's explicit resume command for a changed total ceiling.")
        else:
            state = {"schemaVersion": 1, "kind": "joint-paired-batch-training", "settings": settings, "attempt": 0, "status": "pending"}
        training_root = directory / "training"
        training_root.mkdir(exist_ok=True)
        config = transcriber.default_config()
        config["data"]["manifest"] = prepared["manifestPath"]
        config["data"]["num_threads"] = cpu_threads
        config["training"].update(device=device, epochs=epochs, max_steps=max_steps)
        config["video"] = {"index": prepared["indexPath"], "model": asdict(video_config)}
        config_path = training_root / "joint-config.json"
        if config_path.exists() and read_json(config_path) != config:
            raise ValueError("The saved joint configuration changed.")
        if not config_path.exists():
            publish_json(config_path, config)
        transcriber.log_progress(f"Batch training: configuration ready: {config_path}")
        if state["status"] == "complete":
            if sha256(Path(state["checkpoint"])) != state["checkpointSha256"]:
                raise ValueError("The completed joint checkpoint changed.")
            transcriber.log_progress("Batch training: completed checkpoint verified; reusing finished run.")
            return state
        run_path = training_root / f"joint-{state['attempt']:03d}"
        resume = run_path / "latest.pt"
        if run_path.exists() and any(run_path.iterdir()) and not resume.is_file():
            state["attempt"] += 1
            run_path = training_root / f"joint-{state['attempt']:03d}"
            if run_path.exists():
                raise ValueError(f"Refusing to adopt an existing training attempt: {run_path}")
            resume = run_path / "latest.pt"
        command = ["train", "--data-root", str(root), "--config", str(config_path), "--run-dir", str(run_path)]
        if max_hours is not None:
            command.extend(["--max-hours", str(max_hours)])
        if resume.is_file():
            command.extend(["--resume", str(resume)])
        state.update(status="running", runDirectory=str(run_path), configPath=str(config_path),
                     preparationSummary=str(directory / "summary.json"), invocationMaxHours=max_hours)
        publish_json(state_path, state)
        budget = "no wall-clock limit" if max_hours is None else f"budget {max_hours:g} hours"
        transcriber.log_progress(f"Batch training: {'resuming' if resume.is_file() else 'starting'} {run_path} | {budget}.")
        try:
            if train_command(command) != 0:
                raise ValueError("Joint training failed; completed checkpoints are preserved. Repeat the same batch-train command to resume.")
            summary_path = run_path / "summary.json"
            if not summary_path.is_file():
                raise ValueError("Training returned without its summary; no completed result is recorded.")
            summary = read_json(summary_path)
            if summary.get("stopped_by") == "max_seconds":
                latest = run_path / "latest.pt"
                if latest.is_file():
                    checkpoint = load_checkpoint(latest)
                    if (checkpoint["identity"]["manifest_sha256"] != manifest_hash
                            or checkpoint["identity"].get("video", {}).get("config") != asdict(video_config)):
                        raise ValueError("Paused checkpoint does not match this joint model and release.")
                state.update(status="paused", stoppedBy="max_seconds", checkpoint=str(latest) if latest.is_file() else None,
                             checkpointSha256=sha256(latest) if latest.is_file() else None,
                             summaryPath=str(summary_path), validationPending=summary.get("validation_pending", False),
                             optimizerUpdatesThisInvocation=summary.get("optimizer_updates", summary.get("training_steps_processed", 0)))
                return state
            checkpoint_path = run_path / "best-events.pt"
            if not checkpoint_path.is_file():
                raise ValueError("Training returned without its checkpoint and summary; no completed result is recorded.")
            trained = load_checkpoint(checkpoint_path)
            if (trained["global_step"] <= 0 or trained["identity"]["manifest_sha256"] != manifest_hash
                    or trained["identity"].get("video", {}).get("config") != asdict(video_config)):
                raise ValueError("Training checkpoint does not match this job's joint model and release.")
            state.update(status="complete", checkpoint=str(checkpoint_path), checkpointSha256=sha256(checkpoint_path),
                         optimizerUpdatesThisInvocation=summary.get("optimizer_updates", summary.get("training_steps_processed")),
                         summaryPath=str(run_path / "summary.json"))
        finally:
            if state["status"] == "running":
                state["status"] = "training-failed"
            publish_json(state_path, state)
        return state
