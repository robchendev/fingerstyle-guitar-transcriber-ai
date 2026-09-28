"""One resumable local audio/optional-video to GP workflow; never trains models."""

from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from .audio_tools import AcquisitionError, executable, run_media
from .dataset_io import ROOT, publish_json, read_json, sha256
from .prepare_training_data import regular_path
from .transcriber_audio import HarnessError


STATE_NAME = "transcription-state.json"
SUMMARY_NAME = "transcription.json"
SOURCE_OPTIONS = ("audio", "metadata", "checkpoint", "template", "beat_checkpoint",
                  "video", "video_python", "hand_model", "pose_model",
                  "ffmpeg", "ffprobe")


def add_commands(commands):
    from .transcriber import add_draft_arguments

    command = commands.add_parser("transcribe", help="Local audio or video to both GP drafts in one resumable job; no training.",
                                  description="Run the existing infer, analyze-beats and export-gp APIs with their normal defaults. Rerun the identical command to resume. Results and review actions are in transcription.json.")
    command.add_argument("--data-root", default=str(ROOT))
    command.add_argument("--audio", help="Local audio file. Omit when supplying --video to extract its embedded soundtrack automatically.")
    for name in ("metadata", "checkpoint", "template", "beat-checkpoint", "output-directory"):
        command.add_argument("--" + name, required=True, help={
            "metadata": "JSON with exact six pre-capo openStringMidi pitches in string6-to1 order, capoFret, tempo {bpm, beatUnit:[numerator,denominator]}, timeSignature:[numerator,denominator], and firstFullBarDownbeatSeconds. No assumed tuning, tempo, or pickup phase.",
            "beat-checkpoint": "Existing local Beat This! checkpoint; never downloaded automatically.",
            "template": "Existing modern GP template, read only.",
            "output-directory": "New dedicated directory under runs, or the identical existing job to resume.",
        }.get(name))
    command.add_argument("--device", default="auto")
    command.add_argument("--beat-device", default="cpu")
    command.add_argument("--video", help="Original video; its soundtrack is extracted when --audio is omitted. Frames become numeric hand evidence, not RGB model inputs. Requires a self-contained trained schema-4 joint checkpoint.")
    command.add_argument("--ffmpeg-dir", help="Existing FFmpeg/FFprobe directory for video-only soundtrack extraction; otherwise use PATH or the project's installed tools.")
    command.add_argument("--video-python", help="Existing isolated video Python; no environment installation or downloads.")
    command.add_argument("--hand-model", help="Existing cached MediaPipe hand model.")
    command.add_argument("--pose-model", help="Optional existing cached MediaPipe pose model.")
    command.add_argument("--plucking-screen-side", choices=("geometry", "left", "right"), default="geometry")
    add_draft_arguments(command)
    for name in ("transcribe-status", "transcribe-review"):
        command = commands.add_parser(name, help="Inspect a source-bound job." if name.endswith("status") else "Record a video review and resume the same transcription job.")
        command.add_argument("--data-root", default=str(ROOT))
        command.add_argument("--output-directory", required=True)
        if name.endswith("review"):
            command.add_argument("--accept-shots", action="store_true")
            command.add_argument("--add-cut", type=float, action="append", default=[])
            command.add_argument("--alignment-offset", type=float)


def _path(value, root, *, existing=False):
    path = Path(value)
    path = regular_path(path if path.is_absolute() else root / path)
    if existing and not path.is_file():
        raise HarnessError(f"Required local input is missing: {path}")
    return path


def _directory(args):
    from .transcriber import private_output

    root = regular_path(args.data_root)
    directory = regular_path(private_output(args.output_directory, root))
    if not directory.is_relative_to(root / "runs") or directory == root / "runs":
        raise HarnessError("Transcription output must be a dedicated job directory under private runs.")
    return root, directory


def _options(args, root, directory):
    options = {name: getattr(args, name, None) for name in SOURCE_OPTIONS}
    if not options["audio"] and not options["video"]:
        raise HarnessError("Supply --audio, --video, or both.")
    _checkpoint(_path(options["checkpoint"], root, existing=True), require_video=bool(options["video"]))
    if options["video"]:
        options["video_python"] = options["video_python"] or str(ROOT / "scripts" / "video-evidence" / ".venv" / "Scripts" / "python.exe")
        options["hand_model"] = options["hand_model"] or str(ROOT / "runs" / "video-evidence" / "models" / "hand_landmarker.task")
    elif any(options[name] for name in ("video_python", "hand_model", "pose_model")) or args.plucking_screen_side != "geometry":
        raise HarnessError("Video options require --video.")
    if not options["audio"]:
        try:
            for name in ("ffmpeg", "ffprobe"):
                options[name] = executable(name, getattr(args, "ffmpeg_dir", None))
        except AcquisitionError as error:
            raise HarnessError(str(error)) from error
    elif getattr(args, "ffmpeg_dir", None):
        raise HarnessError("--ffmpeg-dir is only used when extracting audio from --video without --audio.")
    for name, value in options.items():
        if value is not None:
            path = _path(value, root, existing=True)
            if path.is_relative_to(directory):
                raise HarnessError("Source files must be outside the transcription output directory.")
            options[name] = str(path)
    options.update(data_root=str(root), output_directory=str(directory), device=args.device,
                   beat_device=args.beat_device, plucking_screen_side=args.plucking_screen_side)
    from .transcriber import draft_cli_values

    options["draft_profile"] = draft_cli_values(args)
    return options


def _implementation(video):
    scripts = Path(__file__).parent
    modules = (
        "transcription_pipeline.py", "audio_tools.py", "local_media.py", "transcriber.py", "dataset_io.py", "prepare_training_data.py",
        "transcriber_audio.py", "transcriber_model.py", "transcriber_video.py", "video_features.py", "transcriber_runtime.py", "transcriber_events.py",
        "draft_cleanup.py", "stroke_normalization.py", "gp_output.py", "gp_stylesheet.py", "gp_events.py", "gp_normalization.py",
        "canonical_events.py", "beat_tracking.py",
        "voice_optimizer.py", "technique_supervision.py", "connection_supervision.py",
        "percussion_supervision.py", "inspect_gp_files.py", "settings.py",
    )
    # Include shared score/rhythm helpers without coupling jobs to unrelated preparation commands.
    paths = {scripts / name for name in modules}
    paths.update(scripts.glob("*rhythm*.py"))
    paths.update(scripts.glob("*fingering*.py"))
    if video:
        paths.add(scripts / "paired_video.py")
        paths.update(path for path in (scripts / "video-evidence").glob("*.py") if not path.name.startswith("test_"))
    return {str(path.relative_to(scripts)): sha256(path) for path in sorted(paths)}


def _identity(options):
    from .transcriber_runtime import resolve_device

    try:
        packages = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "soundfile", "ortools", "beat-this")}
    except importlib.metadata.PackageNotFoundError as error:
        raise HarnessError(f"Required local transcription dependency is missing: {error.name}. Use the project's existing inference environment; nothing was downloaded.") from error
    return {
        "options": options,
        "inputs": {name: {"path": options[name], "sha256": sha256(_path(options[name], Path(options["data_root"]), existing=True))}
                   for name in SOURCE_OPTIONS if options[name] is not None},
        "implementationSha256": _implementation(bool(options["video"])),
        "runtime": {"python": sys.version, "executable": str(regular_path(sys.executable)),
                    "inferenceDevice": str(resolve_device(options["device"])),
                    "beatDevice": str(resolve_device(options["beat_device"])),
                    "packages": packages},
    }


@contextmanager
def _lock(directory):
    path = regular_path(directory / ".transcription.lock")
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise HarnessError(f"Transcription is running or was interrupted. Inspect {path}; remove only this stale lock after confirming its PID has exited.") from error
    try:
        with stream:
            stream.write(str(os.getpid()))
            stream.flush()
            yield
    finally:
        path.unlink()


def _save(directory, state):
    publish_json(directory / STATE_NAME, state)


def _check_identity(state):
    if _identity(state["options"]) != state["identity"]:
        raise HarnessError("Transcription inputs, options, runtime or implementation changed. Use a new output directory; existing artifacts are not replaced.")


def _check_hashes(hashes):
    for value, digest in hashes.items():
        path = regular_path(value)
        if not path.is_file() or sha256(path) != digest:
            raise HarnessError(f"Transcription artifact changed or is missing: {path}. Use a new output directory.")


def _read_state(directory):
    state = read_json(regular_path(directory / STATE_NAME))
    if state.get("schemaVersion") != 1 or state.get("kind") != "transcription-pipeline-state" or state.get("options", {}).get("output_directory") != str(directory):
        raise HarnessError("Not a transcription job for this output directory.")
    _check_identity(state)
    for stage in state["stages"].values():
        if stage["status"] == "complete":
            _check_hashes(stage["hashes"])
    if state.get("summarySha256"):
        _check_hashes({str(directory / SUMMARY_NAME): state["summarySha256"]})
    if state.get("videoRequest"):
        _check_hashes({state["videoRequest"]: state["videoRequestSha256"]})
    if state.get("videoAlignment"):
        _check_hashes({state["videoAlignment"]: state["videoAlignmentSha256"]})
    if state.get("videoResult"):
        _check_hashes({state["videoResult"]: state["videoResultSha256"]})
    return state


def _summary(directory, state):
    outputs = {}
    for stage in state["stages"].values():
        if stage["status"] == "complete":
            outputs.update(stage["outputs"])
    return {
        "schemaVersion": 1, "kind": "transcription-pipeline-result", "status": state["status"],
        "visibility": "private", "trainingPerformed": False,
        "outputDirectory": str(directory), "state": str(directory / STATE_NAME),
        "report": str(directory / SUMMARY_NAME), "outputs": outputs,
        "inputIdentity": state["identity"]["inputs"],
        "stages": {name: value["status"] for name, value in state["stages"].items()},
        "videoPreparation": state.get("videoResult"), "actions": state.get("actions", []),
    }


def _publish(directory, state):
    summary = _summary(directory, state)
    path = directory / SUMMARY_NAME
    if path.exists():
        if not state.get("summarySha256"):
            raise HarnessError(f"Refusing to overwrite an unrecorded summary: {path}")
        _check_hashes({str(path): state["summarySha256"]})
    publish_json(path, summary)
    state["summarySha256"] = sha256(path)
    _save(directory, state)
    return summary


def _attempt(directory, state, name):
    attempt = regular_path(directory / "attempts" / f"{name}-{uuid4().hex}")
    attempt.mkdir(parents=True, exist_ok=False)
    state["attempts"].append({"stage": name, "directory": str(attempt)})
    state["stages"][name] = {"status": "running", "directory": str(attempt)}
    _save(directory, state)
    return attempt


def _complete(directory, state, name, outputs, *, additional_hashes=None):
    hashes = {}
    for value in outputs.values():
        path = regular_path(value)
        if not path.is_file():
            raise HarnessError(f"Stage {name} did not publish required artifact: {path}")
        hashes[str(path)] = sha256(path)
    hashes.update(additional_hashes or {})
    _check_identity(state)
    _check_hashes(hashes)
    state["stages"][name].update(status="complete", outputs={key: str(value) for key, value in outputs.items()}, hashes=hashes)
    _save(directory, state)


def _stage(directory, state, name, operation):
    from .transcriber import log_progress

    if state["stages"].get(name, {}).get("status") == "complete":
        _check_hashes(state["stages"][name]["hashes"])
        log_progress(f"Transcription {name}: reusing completed output.")
        return state["stages"][name]["outputs"]
    log_progress(f"Transcription {name}: starting...")
    attempt = _attempt(directory, state, name)
    outputs = operation(attempt)
    if any(not regular_path(value).is_relative_to(attempt) for value in outputs.values()):
        raise HarnessError("Stage outputs must remain inside their unique attempt directory.")
    _complete(directory, state, name, outputs)
    log_progress(f"Transcription {name}: completed.")
    return state["stages"][name]["outputs"]


def _audio_input(state):
    if state["options"]["audio"]:
        return state["identity"]["inputs"]["audio"]
    stage = state["stages"].get("audio", {})
    if stage.get("status") != "complete":
        raise HarnessError("Video soundtrack extraction must complete before transcription.")
    path = stage["outputs"]["audio"]
    return {"path": path, "sha256": stage["hashes"][path]}


def _soundtrack_timeline(options):
    from .local_media import soundtrack_timeline

    streams = json.loads(run_media([
        options["ffprobe"], "-v", "error", "-show_entries",
        "stream=index,codec_type:stream_disposition=attached_pic", "-of", "json", options["video"],
    ])).get("streams", [])
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not audio:
        raise HarnessError("The video has no audio stream. Supply a separate --audio file.")
    if len(audio) != 1:
        raise HarnessError("The video has multiple audio streams. Supply --audio explicitly to select the intended performance.")
    if not any(stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic", 0) for stream in streams):
        raise HarnessError("--video must contain a video stream; use --audio for audio-only media.")
    try:
        stream = soundtrack_timeline(options["video"], options["ffprobe"])
    except ValueError as error:
        raise HarnessError(f"Cannot establish the embedded soundtrack's sample timeline: {error}") from error
    return {"streamIndex": stream["index"], "sampleRate": stream["sampleRate"], "channels": stream["channels"],
            "sampleCount": stream["sampleCount"], "firstPts": stream["firstDecodedPts"], "timeBase": stream["timeBase"],
            "videoStartSecondsForAudioZero": float(stream["firstDecodedPts"] * Fraction(*stream["timeBase"]))}


def _video_environment(options):
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    directories = list(dict.fromkeys(str(Path(options[name]).parent) for name in ("ffmpeg", "ffprobe") if options.get(name)))
    if directories:
        environment["PATH"] = os.pathsep.join([*directories, environment.get("PATH", "")])
    return environment


def _extract_audio(attempt, state):
    import soundfile as sf
    from .transcriber import log_progress

    options = state["options"]
    audio = attempt / "soundtrack.wav"
    raw_alignment = attempt / "soundtrack-correlation.json"
    alignment = attempt / "video-alignment.json"
    receipt = attempt / "audio-extraction.json"
    try:
        log_progress("Video soundtrack: reading audio streams and source timestamps...")
        timeline = _soundtrack_timeline(options)
        log_progress("Video soundtrack: decoding to a private floating-point WAV; original video is unchanged...")
        run_media([
            options["ffmpeg"], "-nostdin", "-v", "error", "-n", "-threads", "1", "-copyts",
            "-i", options["video"], "-map", f"0:{timeline['streamIndex']}", "-vn", "-sn", "-dn",
            "-c:a", "pcm_f32le", "-threads", "1", str(audio),
        ])
        info = sf.info(str(audio))
        if (info.samplerate, info.channels, info.frames) != (timeline["sampleRate"], timeline["channels"], timeline["sampleCount"]):
            raise HarnessError("Extracted audio does not match the source's decoded sample timeline.")
        log_progress("Video soundtrack: binding extracted audio to the original video clock...")
        environment = _video_environment(options)
        # The isolated producer only publishes under its own private root and must not download tools.
        video_root = ROOT / "runs" / "video-evidence"
        video_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="transcription-alignment-", dir=video_root) as temporary:
            isolated_report = Path(temporary) / "alignment.json"
            completed = subprocess.run([
                options["video_python"], "-c",
                "from batch_video import _installed_tools; _installed_tools(); from cli import main; raise SystemExit(main())",
                "align-audio",
                "--video", options["video"], "--trimmed-audio", str(audio), "--output", str(isolated_report),
            ], cwd=ROOT / "scripts" / "video-evidence", env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600, check=False)
            if completed.returncode:
                raise HarnessError(f"Extracted soundtrack alignment failed: {completed.stderr.strip()[-1200:]}")
            publish_json(raw_alignment, read_json(isolated_report))
    except (AcquisitionError, subprocess.TimeoutExpired) as error:
        raise HarnessError(f"Video soundtrack extraction failed: {error}") from error
    report = read_json(raw_alignment)
    digest = sha256(audio)
    correlation = report.get("correlation")
    if (report.get("status") != "supported" or report.get("rate") != [1, 1]
            or report.get("videoStartSecondsForTrimmedAudioZero") != 0
            or report.get("videoSha256") != state["identity"]["inputs"]["video"]["sha256"]
            or report.get("trimmedAudioSha256") != digest
            or type(correlation) not in (int, float) or not .99 <= correlation <= 1 + 1e-8):
        raise HarnessError("Extracted soundtrack could not be confirmed against the original video; no guessed timing was used.")
    publish_json(receipt, {
        "schemaVersion": 1, "kind": "embedded-video-audio-extraction",
        "videoSha256": state["identity"]["inputs"]["video"]["sha256"], "audioSha256": digest,
        "audioPath": str(audio), **timeline, "trainingPerformed": False,
        "policy": "Decode the sole embedded audio stream without trimming, resampling or channel mixing. Audio zero is the first decoded source audio PTS.",
    })
    publish_json(alignment, {
        **report, "method": "embedded-soundtrack-timestamps",
        "videoStartSecondsForTrimmedAudioZero": timeline["videoStartSecondsForAudioZero"],
        "mapping": "trimmedAudioSeconds = videoSeconds - firstDecodedSourceAudioSeconds",
        "reviewRequired": False, "correlation": min(1., correlation),
        "extractionPath": str(receipt), "extractionSha256": sha256(receipt),
    })
    return {"audio": audio, "audioExtraction": receipt, "alignment": alignment, "soundtrackCorrelation": raw_alignment}


def _video_request(directory, state):
    if state.get("videoRequest"):
        return Path(state["videoRequest"])
    options = state["options"]
    digest = hashlib.sha256(str(directory).encode("utf-8")).hexdigest()
    output = regular_path(ROOT / "runs" / "video-evidence" / "batches" / f"transcription-{digest}" / "input")
    if output.exists():
        raise HarnessError(f"Unrecorded video output already exists: {output}. Use a new job directory.")
    if any(Path(source["path"]).is_relative_to(output) for source in state["identity"]["inputs"].values()):
        raise HarnessError("Video preparation output must not contain any source inputs.")
    request = {
        "schemaVersion": 1, "kind": "paired-video-preparation-request", "id": "input",
        "audio": _audio_input(state)["path"], "video": options["video"],
        "outputDirectory": str(output),
        "pluckingScreenSide": options["plucking_screen_side"], "reuse": {},
        "handModel": options["hand_model"], "poseModel": options["pose_model"],
    }
    if not options["audio"]:
        alignment = regular_path(output.parent / "soundtrack-alignment.json")
        if alignment.exists():
            raise HarnessError(f"Refusing to overwrite an unrecorded source alignment: {alignment}")
        publish_json(alignment, read_json(state["stages"]["audio"]["outputs"]["alignment"]))
        request["reuse"]["alignment"] = str(alignment)
        state.update(videoAlignment=str(alignment), videoAlignmentSha256=sha256(alignment))
    path = regular_path(directory / "video-request.json")
    if path.exists():
        raise HarnessError(f"Refusing to overwrite an unrecorded video request: {path}")
    publish_json(path, request)
    state.update(videoRequest=str(path), videoRequestSha256=sha256(path), videoDirectory=str(output))
    _save(directory, state)
    return path


def _worker(state, output, review_flags=None):
    command = [state["options"]["video_python"], str(ROOT / "scripts" / "video-evidence" / "batch_video.py"),
               "review" if review_flags is not None else "run", "--request", state["videoRequest"], "--output", str(output)]
    if review_flags is not None:
        command.extend(review_flags)
    else:
        command.append("--reset-failed")
    completed = subprocess.run(command, cwd=ROOT, env=_video_environment(state["options"]), check=False)
    if completed.returncode and not output.is_file():
        raise HarnessError(f"Video preparation failed with exit code {completed.returncode}. Inspect the worker error; no audio-only substitution was made.")
    result = read_json(regular_path(output))
    expected = {"audio": _audio_input(state)["sha256"], "video": state["identity"]["inputs"]["video"]["sha256"]}
    if (result.get("schemaVersion") != 1 or result.get("kind") != "paired-video-preparation-result"
            or result.get("id") != "input" or result.get("status") not in ("ready", "needs-review", "blocked")
            or result.get("inputSha256") != expected):
        raise HarnessError("Video worker returned an invalid or differently source-bound result.")
    if completed.returncode and result["status"] != "blocked":
        raise HarnessError(f"Video preparation failed with exit code {completed.returncode}; its result cannot be treated as successful.")
    extracted_alignment = state.get("videoAlignment")
    for name, path in result.get("artifacts", {}).items():
        path = regular_path(path)
        owned = path.is_relative_to(Path(state["videoDirectory"])) or name == "alignment" and str(path) == extracted_alignment
        if not owned or not path.is_file():
            raise HarnessError("Video worker artifacts must be existing files owned by this job.")
    return result


def _prepare_video(directory, state, review_flags=None):
    from .transcriber import log_progress

    if state["stages"].get("video", {}).get("status") == "complete":
        log_progress("Transcription video: reusing completed hand-evidence preparation.")
        return state["stages"]["video"]["outputs"]["videoBundle"]
    log_progress("Transcription video: starting alignment, shot inspection and hand tracking...")
    _video_request(directory, state)
    attempt = _attempt(directory, state, "video")
    result_path = attempt / "video-result.json"
    result = _worker(state, result_path, review_flags)
    state.update(videoResult=str(result_path), videoResultSha256=sha256(result_path))
    if result["status"] != "ready":
        state["status"] = result["status"]
        state["actions"] = []
        for action in result.get("actions", []):
            review = {"shots": ["--accept-shots"],
                      "alignment": ["--alignment-offset", "<SECONDS>"]}.get(action.get("stage"))
            if review:
                action = {**action, "workerCommand": action.get("command"),
                          "command": [sys.executable, "-m", "scripts.transcriber", "transcribe-review",
                                      "--data-root", state["options"]["data_root"], "--output-directory", str(directory), *review]}
            state["actions"].append(action)
        state["stages"]["video"]["status"] = result["status"]
        _save(directory, state)
        log_progress(f"Transcription video: {result['status']}; required actions saved.")
        return None
    from .paired_video import load_inference_video

    bundle_path = result.get("artifacts", {}).get("bundle")
    if not bundle_path:
        raise HarnessError("Ready video preparation did not publish its paired bundle.")
    bundle = load_inference_video(bundle_path, _audio_input(state)["sha256"])
    if bundle.identity["videoSha256"] != state["identity"]["inputs"]["video"]["sha256"]:
        raise HarnessError("Video bundle original video identity differs from this job.")
    report = read_json(bundle_path)
    bound = {str(Path(bundle_path).with_name(report["arraysPath"])): report["arraysSha256"]}
    bound.update({path: report["inputSha256"][name] for name, path in report["inputPaths"].items()})
    _complete(directory, state, "video", {"videoBundle": bundle_path, "videoResult": str(result_path)}, additional_hashes=bound)
    log_progress("Transcription video: completed; aligned numeric hand inputs are ready.")
    return bundle_path


def _checkpoint(path, *, require_video):
    from .transcriber import video_model_config
    from .transcriber_runtime import INFERENCE_FORMAT, checkpoint_identity, load_checkpoint

    checkpoint = load_checkpoint(path, allow_inference=True)
    identity = checkpoint_identity(checkpoint)
    if require_video and "video" not in identity:
        raise HarnessError("--video requires a paired-trained joint checkpoint. Historical audio-only checkpoints cannot consume video.")
    if "video" in identity:
        video_model_config(identity["video"]["config"], allow_legacy=True)
    if checkpoint.get("format") != INFERENCE_FORMAT:
        if checkpoint["global_step"] <= 0:
            raise HarnessError("Transcription requires a trained checkpoint, not random initialization.")
        counts = checkpoint.get("resume_state", checkpoint["history"][-1] if checkpoint["history"] else {})
        if "video" in identity and counts.get("optimizer_updates") == 0:
            raise HarnessError("Joint transcription requires actual optimizer updates; this checkpoint only skipped unsupervised or zero-gradient batches.")
    return checkpoint


def _preflight(state):
    from .transcriber import inference_metadata
    from .transcriber_runtime import checkpoint_identity

    checkpoint = _checkpoint(state["options"]["checkpoint"], require_video=bool(state["options"]["video"]))
    conditioning_dim = checkpoint_identity(checkpoint)["model"]["conditioning_dim"]
    inference_metadata(read_json(state["options"]["metadata"]), conditioning_dim)
    return checkpoint


def _execute(directory, state, *, review_flags=None):
    from .transcriber import analyze_beats, argument_parser, export_gp, infer, log_progress

    state.update(status="running", actions=[])
    _save(directory, state)
    try:
        log_progress("Transcription preflight: validating musical settings and the trained checkpoint...")
        _preflight(state)
        log_progress("Transcription preflight: completed.")
        options = state["options"]
        if not options["audio"]:
            _stage(directory, state, "audio", lambda attempt: _extract_audio(attempt, state))
        audio_path = _audio_input(state)["path"]
        video_bundle = _prepare_video(directory, state, review_flags) if options["video"] else None
        if options["video"] and video_bundle is None:
            return _publish(directory, state)
        parser = argument_parser()

        def inference(attempt):
            path = attempt / "predictions.json"
            command = ["infer", "--data-root", options["data_root"], "--audio", audio_path,
                       "--metadata", options["metadata"], "--checkpoint", options["checkpoint"],
                       "--device", options["device"], "--output", str(path)]
            thresholds = [options["draft_profile"][name] for name in ("draft-percussion-threshold", "thumb-slap-threshold")
                         if options["draft_profile"][name] is not None]
            command.extend(("--percussion-threshold", str(min(.5, *thresholds))))
            if video_bundle:
                command.extend(("--video-bundle", video_bundle))
            infer(parser.parse_args(command))
            return {"predictions": path}

        predictions = _stage(directory, state, "inference", inference)["predictions"]

        def beats(attempt):
            path = attempt / "beat-evidence.json"
            analyze_beats(parser.parse_args(["analyze-beats", "--data-root", options["data_root"], "--audio", audio_path,
                                             "--checkpoint", options["beat_checkpoint"], "--device", options["beat_device"], "--output", str(path)]))
            return {"beatEvidence": path}

        evidence = _stage(directory, state, "beats", beats)["beatEvidence"]

        def export(attempt):
            outputs = {"fullVoices": attempt / "transcription.full-voices.gp",
                       "singleVoice": attempt / "transcription.single-voice.gp", "gpReport": attempt / "gp-output.json"}
            command = ["export-gp", "--data-root", options["data_root"], "--predictions", predictions,
                       "--beat-evidence", evidence, "--template", options["template"],
                       "--full-output", str(outputs["fullVoices"]), "--single-output", str(outputs["singleVoice"]),
                       "--report", str(outputs["gpReport"])]
            for name, value in options["draft_profile"].items():
                if value is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        command.append("--" + name)
                else:
                    command.extend(("--" + name, str(value)))
            export_gp(parser.parse_args(command))
            return outputs

        _stage(directory, state, "export", export)
        log_progress("Transcription: verifying final artifacts and saving the result report...")
        _check_identity(state)
        for stage in state["stages"].values():
            _check_hashes(stage["hashes"])
        state.update(status="ready", actions=[])
        result = _publish(directory, state)
        log_progress(f"Transcription: completed. Results: {directory / SUMMARY_NAME}")
        return result
    except (OSError, ValueError, RuntimeError) as error:
        for stage in state["stages"].values():
            if stage["status"] == "running":
                stage.update(status="failed", error=str(error))
        state.update(status="blocked", actions=[{"stage": "transcription", "reason": str(error),
                                               "resume": "Resolve the reported problem and rerun the same command. Changed inputs require a new output directory."}])
        _publish(directory, state)
        log_progress(f"Transcription: blocked - {error}")
        raise


def run_transcription(args):
    """Run the shared inference/beat/export APIs or resume unchanged completed stages."""
    from .transcriber import log_progress

    log_progress("Transcription setup: validating paths, options and checkpoint compatibility...")
    root, directory = _directory(args)
    options = _options(args, root, directory)
    log_progress("Transcription setup: input paths and options validated.")
    directory.mkdir(parents=True, exist_ok=True)
    with _lock(directory):
        if (directory / STATE_NAME).exists():
            log_progress("Transcription setup: verifying the saved job and completed stages...")
            state = _read_state(directory)
            if state["options"] != options:
                raise HarnessError("Transcription options changed. Use a new output directory.")
        else:
            if any(path.name != ".transcription.lock" for path in directory.iterdir()):
                raise HarnessError("New transcription output directory must be empty; existing files are not overwritten.")
            log_progress("Transcription setup: hashing source files and recording the new job...")
            state = {"schemaVersion": 1, "kind": "transcription-pipeline-state", "status": "pending",
                     "options": options, "identity": _identity(options), "stages": {}, "attempts": [], "actions": []}
            _save(directory, state)
        log_progress("Transcription setup: completed.")
        if state["status"] == "ready":
            log_progress("Transcription: already completed; returning the existing GP paths.")
            return _summary(directory, state)
        return _execute(directory, state)


def transcription_status(args):
    """Report only after rechecking all immutable inputs and completed artifacts."""
    _, directory = _directory(args)
    with _lock(directory):
        return _summary(directory, _read_state(directory))


def review_transcription(args):
    """Delegate explicit manual decisions to the existing isolated video worker."""
    _, directory = _directory(args)
    with _lock(directory):
        state = _read_state(directory)
        if not state["options"]["video"] or state["status"] != "needs-review":
            raise HarnessError("This job is not waiting for a video review.")
        flags = []
        if args.accept_shots:
            flags.append("--accept-shots")
        for seconds in args.add_cut:
            flags.extend(("--add-cut", str(seconds)))
        if args.alignment_offset is not None:
            flags.extend(("--alignment-offset", str(args.alignment_offset)))
        if not flags:
            raise HarnessError("Choose --accept-shots or --alignment-offset.")
        return _execute(directory, state, review_flags=flags or None)
