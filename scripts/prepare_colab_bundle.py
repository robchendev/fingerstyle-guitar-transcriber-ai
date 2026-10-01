"""Build a portable private corpus archive for Colab from macOS or Windows."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory

from .dataset_io import ROOT


TOP_LEVEL_FIELDS = {
    "schemaVersion", "kind", "records", "acceptConventions",
    "reviewMode", "reviewBudget",
}
RECORD_FIELDS = {
    "id", "groupId", "split", "gp", "video", "audio", "title", "clips",
    "reuse", "pluckingScreenSide", "voiceSupervisionPolicy",
}
REUSABLE_STAGES = {"alignment", "shots", "hands", "annotations", "geometry", "roles"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locate(root, value):
    path = Path(value).expanduser()
    if path.is_file() or path.is_dir():
        return path.resolve()
    text = str(value).replace("\\", "/")
    for marker in ("data/", "runs/"):
        index = text.find(marker)
        if index >= 0:
            candidate = root / text[index:]
            if candidate.is_file() or candidate.is_dir():
                return candidate.resolve()
    raise ValueError(f"Required private input does not exist: {value}")


def _relative(root, path):
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Private corpus input must be inside the repository: {path}") from error


def _copy_file(root, staging, value, *, destination=None):
    source = _locate(root, value)
    if not source.is_file():
        raise ValueError(f"Expected a file: {source}")
    relative = Path(destination) if destination is not None else Path(_relative(root, source))
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return relative.as_posix()


def _copy_directory(root, staging, value):
    source = _locate(root, value)
    if not source.is_dir():
        raise ValueError(f"Expected a directory: {source}")
    relative = Path(_relative(root, source))
    shutil.copytree(source, staging / relative, dirs_exist_ok=True)
    return relative.as_posix()


def _default_git_bundle(root, output):
    for arguments in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
        ["git", "ls-files", "--error-unmatch", "scripts/prepare_colab_bundle.py"],
    ):
        result = subprocess.run(
            arguments, cwd=root, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode:
            raise ValueError(
                "Commit tracked source changes before building the Colab archive."
            )
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True,
    ).strip()
    if not branch:
        raise ValueError("Create the Colab archive from a named Git branch, not detached HEAD.")
    subprocess.run(
        ["git", "bundle", "create", str(output), branch],
        cwd=root, check=True,
    )


def build_colab_bundle(batch_path, release_manifest, detector_model, output, *,
                       hand_model="runs/video-evidence/models/hand_landmarker.task",
                       fretboard_device="0", workers=1, root=ROOT,
                       git_bundle_creator=None):
    root = Path(root).resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite Colab archive: {output}")
    if output.suffix.lower() != ".tar":
        raise ValueError("Colab archive output must use the uncompressed .tar extension.")
    batch_path = _locate(root, batch_path)
    release_manifest = _locate(root, release_manifest)
    detector_model = _locate(root, detector_model)
    hand_model = _locate(root, hand_model)
    if not isinstance(fretboard_device, str) or not fretboard_device.strip():
        raise ValueError("fretboard_device must be a nonempty Ultralytics device string.")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer from 1 to 16.")
    try:
        raw = json.loads(batch_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid source batch manifest: {error}") from error
    if (
        not isinstance(raw, dict)
        or raw.get("schemaVersion") != 1
        or raw.get("kind") != "paired-preparation-batch"
        or not isinstance(raw.get("records"), list)
        or not raw["records"]
    ):
        raise ValueError("Expected a nonempty version-1 paired-preparation batch.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="fingerstyle-colab-") as temporary:
        staging = Path(temporary)
        portable = {
            key: raw[key] for key in TOP_LEVEL_FIELDS if key in raw
        }
        portable.update(
            workspace="data",
            releaseManifest=_relative(root, release_manifest),
            videoPython="__COLAB_PYTHON__",
            handModel="runs/video-evidence/models/hand_landmarker.task",
            fretboardModel="runs/fretboard-detector-platform/best.pt",
            fretboardDevice=fretboard_device,
            workers=workers,
        )
        _copy_directory(root, staging, release_manifest.parent)
        pairs = root / "data" / "pairs.json"
        if pairs.is_file():
            _copy_file(root, staging, pairs)
        _copy_file(
            root, staging, hand_model,
            destination=portable["handModel"],
        )
        _copy_file(
            root, staging, detector_model,
            destination=portable["fretboardModel"],
        )

        portable_records = []
        for raw_record in raw["records"]:
            if not isinstance(raw_record, dict):
                raise ValueError("Batch records must be objects.")
            record = {
                key: raw_record[key] for key in RECORD_FIELDS if key in raw_record
            }
            for name in ("id", "groupId", "split", "gp", "video", "pluckingScreenSide"):
                if name not in record:
                    raise ValueError(f"Batch record is missing {name}.")
            for name in ("gp", "video", "audio"):
                if name in record:
                    record[name] = _copy_file(root, staging, record[name])
            reuse = {}
            for stage, artifact in raw_record.get("reuse", {}).items():
                if stage not in REUSABLE_STAGES:
                    continue
                artifact = _locate(root, artifact)
                _copy_directory(root, staging, artifact.parent)
                reuse[stage] = _relative(root, artifact)
            record["reuse"] = reuse
            portable_records.append(record)
        portable["records"] = portable_records

        manifest_path = staging / "runs" / "colab-batch.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(portable, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        bundle_path = staging / "fingerstyle.bundle"
        (git_bundle_creator or _default_git_bundle)(root, bundle_path)

        with tarfile.open(output, "w") as archive:
            for path in sorted(staging.rglob("*")):
                archive.add(path, arcname=path.relative_to(staging), recursive=False)
    return {
        "output": str(output),
        "sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "records": len(portable["records"]),
        "batchManifest": "runs/colab-batch.json",
        "fretboardDevice": fretboard_device,
        "workers": workers,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--batch", required=True)
    result.add_argument("--release-manifest", required=True)
    result.add_argument("--detector-model", required=True)
    result.add_argument("--output", required=True)
    result.add_argument(
        "--hand-model",
        default="runs/video-evidence/models/hand_landmarker.task",
    )
    result.add_argument("--fretboard-device", default="0")
    result.add_argument("--workers", type=int, default=1)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    report = build_colab_bundle(
        args.batch, args.release_manifest, args.detector_model, args.output,
        hand_model=args.hand_model,
        fretboard_device=args.fretboard_device,
        workers=args.workers,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
