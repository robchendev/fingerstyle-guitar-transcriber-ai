# Google Colab fretboard preparation and joint training

This workflow transfers a private, reviewed corpus from macOS to a Colab GPU,
runs the seven-keypoint fretboard detector at the pipeline's 0.5-second cadence,
checks complete-video geometry, and trains the architecture-6 multimodal
transcriber from scratch.

The detector checkpoint must be an Ultralytics PyTorch `.pt` file whose
`kpt_shape` is `[7, 3]`. The detector annotation dataset is not needed for
transcriber training; retain it on the Mac for future detector retraining.

## Before using Colab

Final joint training requires all of the following:

- The filmed tuning and physical full-capo corrections are applied.
- The selected immutable release contains `downbeatConditioning`.
- GP/audio alignment and source-bound review are complete.
- The detector checkpoint is the best validation checkpoint, not merely the
  last training epoch.

It is safe to run the fretboard preparation and overlay review against an older
frozen release, but do not use that release for final joint training if it
predates the corrections or downbeat contract.

## Colab account and runtime

A Google account automatically has free Colab access. Open
[Google Colab](https://colab.research.google.com/), sign in, and create a new
notebook. Paid plans and current pricing are available on the
[official signup page](https://colab.research.google.com/signup).

In the notebook, choose **Runtime → Change runtime type → GPU**. Prefer an L4
or A100 when available. Free and paid managed runtimes have variable hardware
and lifetime limits. Google documents up to 12 hours for general managed
runtimes and up to 24 hours of continuous execution for Pro+ when compute units
remain. Use bounded training invocations and copy every resumable result to
Drive before ending a runtime.

Colab service and Google Drive storage are separate. The private corpus archive
may require additional Google Drive or Google One storage.

## 1. Build the portable archive on macOS

Update the branch and place the Ultralytics Platform checkpoint somewhere on
the Mac:

```bash
cd /path/to/fingerstyle-ai-transcriber
git checkout fretboard-modelling
git pull
```

The archive builder refuses tracked staged or unstaged changes because a Git
bundle contains commits, not the working tree. Commit intended code changes
before packaging. Untracked private data remains outside Git and is copied
through the explicit corpus inputs below.

Build the archive. Replace the batch, release and detector paths with the
reviewed versions on the Mac:

```bash
python3 -m scripts.prepare_colab_bundle \
  --batch runs/video-evidence/batch-manifests/corpus-hand-evidence-v4.json \
  --release-manifest data/releases/CORRECTED-RELEASE/manifest.json \
  --detector-model "$HOME/Downloads/best.pt" \
  --output "$HOME/Desktop/fingerstyle-colab-input.tar" \
  --fretboard-device 0 \
  --workers 1
```

The command:

- creates a Git bundle from the checked-out branch;
- rewrites Windows or macOS corpus paths to repository-relative paths;
- copies only selected GP, audio, video, release and reusable stage families;
- omits obsolete source-map and receipt fields;
- omits old `bundle` and `fretboard` reuse so schema-5 data is rebuilt;
- stores the detector as `runs/fretboard-detector-platform/best.pt`;
- configures one worker and GPU device `0`.

One worker is intentional. Multiple workers launch multiple Ultralytics
processes against the same GPU and can exhaust VRAM.

Check the archive size and save the printed SHA-256:

```bash
ls -lh "$HOME/Desktop/fingerstyle-colab-input.tar"
shasum -a 256 "$HOME/Desktop/fingerstyle-colab-input.tar"
```

Upload the TAR through Google Drive to:

```text
My Drive/fingerstyle-colab/fingerstyle-colab-input.tar
```

## 2. Mount Drive and restore locally

Create a Colab notebook and run:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Copy the archive to the runtime's local disk before extracting it. Running
video preparation directly through the Drive mount is slower and creates many
small Drive operations.

```bash
!rm -rf /content/fingerstyle-input /content/fingerstyle-ai-transcriber
!mkdir -p /content/fingerstyle-input
!cp "/content/drive/MyDrive/fingerstyle-colab/fingerstyle-colab-input.tar" /content/input.tar
!tar -xf /content/input.tar -C /content/fingerstyle-input

!git clone \
  --branch fretboard-modelling \
  /content/fingerstyle-input/fingerstyle.bundle \
  /content/fingerstyle-ai-transcriber

!rsync -a \
  --exclude fingerstyle.bundle \
  /content/fingerstyle-input/ \
  /content/fingerstyle-ai-transcriber/
```

```python
%cd /content/fingerstyle-ai-transcriber
```

## 3. Install Colab dependencies

```bash
!apt-get -qq update
!apt-get -qq install -y ffmpeg

!python -m pip install -q \
  -r requirements-training.txt \
  -r scripts/video-evidence/requirements.txt

!python -m pip install -q --upgrade ultralytics
```

Do not install `requirements-fretboard.txt` in Colab. It retains the historical
Ultralytics `8.3.102` pin and may not load a newer Platform or YOLO26
checkpoint.

Bind the portable manifest to the current Colab interpreter:

```python
import json
import sys
from pathlib import Path

path = Path("runs/colab-batch.json")
batch = json.loads(path.read_text())
batch["videoPython"] = sys.executable
path.write_text(json.dumps(batch, indent=2) + "\n")
```

Verify CUDA and the detector contract:

```python
import torch
import ultralytics
from ultralytics import YOLO

assert torch.cuda.is_available()
print("Torch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("runs/fretboard-detector-platform/best.pt")
print("Keypoint shape:", model.model.kpt_shape)
assert list(model.model.kpt_shape) == [7, 3]
```

```bash
!nvidia-smi
```

No source edit is required. `fretboardDevice: "0"` is a supported batch field
and is forwarded to every Ultralytics prediction.

## 4. Run the 0.5-second fretboard stage

```bash
%%bash
set -euo pipefail

python -m scripts.prepare_training_data batch \
  --manifest runs/colab-batch.json \
  --output-directory runs/video-evidence/batches/fretboard-colab-v1 \
  2>&1 | tee runs/fretboard-colab-preparation.log
```

The tracker attempts YOLO every 0.5 seconds, propagates accepted points with
optical flow, and attempts early reacquisition when flow is unavailable for six
frames. It may therefore call the detector more often than every 0.5 seconds.
The seventh, fret-5 landmark validates the geometry; the six outer landmarks
form the downstream fretboard representation.

Check status:

```bash
!python -m scripts.prepare_training_data batch-status \
  --manifest runs/colab-batch.json \
  --output-directory runs/video-evidence/batches/fretboard-colab-v1

!cat runs/video-evidence/batches/fretboard-colab-v1/next-actions.txt
```

Do not train until the batch status is `ready`.

## 5. Check corpus coverage and overlays

```python
import json
import statistics
from pathlib import Path

root = Path("runs/video-evidence/batches/fretboard-colab-v1")
reports = list(root.rglob("fretboard.json"))
rows = [json.loads(path.read_text()) for path in reports]

print("Fretboard reports:", len(rows))
print("Median geometry coverage:",
      statistics.median(row["geometryCoverage"] for row in rows))
print("Detector calls:", sum(row["detectorCalls"] for row in rows))
print("Accepted calls:", sum(row["acceptedDetectorCalls"] for row in rows))
print("Tracked frames:", sum(row["trackedFrames"] for row in rows))
```

Generate complete-video overlays for chosen pair IDs:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts/video-evidence").resolve()))
from fretboard_tracking import render_overlay

batch = json.loads(Path("runs/colab-batch.json").read_text())
records = {row["id"]: row for row in batch["records"]}
output_root = Path("runs/video-evidence/batches/fretboard-colab-v1")

sample_ids = ["tab-0001", "tab-0011", "tab-0072"]

for identifier in sample_ids:
    report = next((output_root / identifier).rglob("fretboard.json"))
    destination = Path("/content") / f"{identifier}-fretboard-overlay.mp4"
    render_overlay(records[identifier]["video"], report, destination)
    print(destination)
```

Download the MP4 files from Colab's **Files** panel. Check point identity,
scene-cut reacquisition, optical-flow stability, occlusion behavior and
close/wide/dark views before starting training.

## 6. Save prepared data to Drive

```bash
!tar -cf /content/fretboard-colab-prepared.tar \
  runs/video-evidence/batches/fretboard-colab-v1

!cp /content/fretboard-colab-prepared.tar \
  "/content/drive/MyDrive/fingerstyle-colab/fretboard-colab-prepared.tar"
```

The archive is the durable checkpoint for preparation. The Colab `/content`
filesystem disappears when its runtime is deleted.

## 7. Train the joint transcriber

This starts architecture-6 audio/hand/fretboard training from random
initialization with a 100-epoch ceiling:

```bash
%%bash
set -euo pipefail

python -m scripts.prepare_training_data batch-train \
  --manifest runs/colab-batch.json \
  --output-directory runs/video-evidence/batches/fretboard-colab-v1 \
  --epochs 100 \
  --device cuda \
  --cpu-threads 4 \
  --max-hours 8 \
  2>&1 | tee runs/joint-colab-training.log
```

Use `--max-hours 8` for a general managed runtime so there is time to save the
result before a possible 12-hour cutoff. Pro+ users with sufficient compute
units may choose a longer bounded invocation such as `--max-hours 20`.

The trainer saves `latest.pt`, reduces learning rate after four
non-improving validations, and stops after 12 non-improving decoded-event
validations. Repeating the identical command against the restored run resumes
the same optimizer and checkpoint.

After every invocation:

```bash
!tar -cf /content/fretboard-colab-run.tar \
  runs/video-evidence/batches/fretboard-colab-v1

!cp /content/fretboard-colab-run.tar \
  "/content/drive/MyDrive/fingerstyle-colab/fretboard-colab-run.tar"
```

On a new runtime, repeat setup, extract the original input TAR, then extract the
latest prepared/run TAR over the repository before repeating `batch-train`.

The selected final checkpoint is stored under:

```text
runs/video-evidence/batches/fretboard-colab-v1/training/joint-000/best-events.pt
```

Copy `best-events.pt`, `latest.pt`, `summary.json`, the generated training
configuration and the complete run archive back to Drive.
