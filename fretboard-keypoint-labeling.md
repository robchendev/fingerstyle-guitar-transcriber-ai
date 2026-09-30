# Fretboard keypoint labeling

This workflow samples native-resolution frames on Windows, supports annotation
in a local browser on macOS, and exports portable labels for YOLO training.
The dataset is stored under `data/`, which is excluded from Git.

Current branch status: this machine retains the original 400 annotated pilot
frames (322 train, 32 validation, 46 test). On the Mac, the owner removed 16
unusable frames and rearranged the remaining 384 to 336 train, 30 validation
and 18 test. Training on Ultralytics Platform reached 0.83 pose mAP50-95; that
updated dataset and checkpoint are not on this machine.

`data\dataset2` extends the local 400-frame pilot to a 2,000-frame annotation
set. Its explicit target is 1,750 train, 156 validation and 94 test frames,
which preserves the owner's 336/30/18 ratio. The 400 seed images and annotations
remain first and unchanged; 1,600 new frames follow. Unusable seed or new
images still count toward the 2,000 selection target and may be removed manually
on the Mac.

## Keypoints

Place these seven ordered points:

1. String 6 contact point at the nut
2. String 1 contact point at the nut
3. String 6 crossing the silver 12th-fret wire
4. String 1 crossing the silver 12th-fret wire
5. String 6 contact point at the bridge saddle
6. String 1 contact point at the bridge saddle
7. Center of the silver 5th-fret wire

Fret numbers refer to the silver fret wires, not the spaces between wires. Nut
and bridge points are the string contact points. The harness derives the nut,
fret-12-wire, and bridge-saddle centers from each outer-string pair. Equal
temperament places fret 5 at approximately 25.08% and fret 12 at exactly 50%
of the nut-to-saddle scale length.

Use **Visible** when the point can be placed directly and **Occluded** when its
position is identifiable despite an obstruction. Clear a point to mark it
unavailable when it is outside the frame or cannot be identified. Do not
estimate an off-screen nut or bridge.

`Complete` means the entire frame has been reviewed. It does not require all
seven points to be available.

Use **Disable this image** when blur, darkness, corruption or other source
quality makes the frame unsuitable for training. Disabled frames count as
reviewed in the UI, appear gray in the progress strip and are omitted from
YOLO exports. They are not exported as negative examples and can be re-enabled
later.

## Create the dataset on Windows

From the repository root:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation select --video-directory runs\video-evidence\sources --shots-directory runs\video-evidence\inspection --hands-directory runs\video-evidence --negative-fraction 0.05 --target-frames 1000 --candidates-per-video 24 --minimum-per-video 1 --maximum-per-video 12 --cache-directory data\fretboard-selection-cache --output data\fretboard-keypoints 2>&1 | Tee-Object -FilePath runs\fretboard-selection.log
```

Frames retain native resolution. Selection uses source-bound hand observations,
prefers two-hand playing frames, limits zero-hand negatives to 5%, uses
available shot reports, and deduplicates visual views across the corpus.
Candidate descriptors are cached per video for resumable selection.

To extend an existing annotated dataset after all authorized source videos and
their 32-frame candidate caches exist:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation extend --base-dataset data\fretboard-keypoints --output data\dataset2 --video-directory runs\video-evidence\sources --train-frames 1750 --validation-frames 156 --test-frames 94 --maximum-per-video 12 --cache-directory data\fretboard-selection-cache
```

The extension keeps every represented source video in its existing split,
assigns each previously unseen video deterministically in the requested ratio,
rejects candidates within 0.5 seconds of an existing frame from that video,
and selects visual diversity independently inside each split. It copies only
the completed seed annotations; all added frames begin incomplete in the same
custom annotation UI. The extension deliberately removes `data.yaml` and
generated labels so incomplete new images cannot be mistaken for negative
training examples. Run `export` only after reviewing the intended training set.

## Test the UI now on Windows

Create a small five-frame Yoyogi dataset:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation init --video runs\video-evidence\sources\tab-0001\source.mkv --frames-per-video 5 --output data\fretboard-keypoints-test
```

Start the local browser UI:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation serve --dataset data\fretboard-keypoints-test
```

The server binds only to `127.0.0.1` and opens
`http://127.0.0.1:8765/`. Stop it with `Ctrl+C`.

Controls:

- Mouse wheel: zoom
- Middle- or right-button drag: pan
- `1` through `7`: select a keypoint
- `Q`, `W`, `E`: available, occluded, unavailable
- Left/right arrows: previous and next frame
- Enter: complete and advance

## Stop and resume

Every point placement, clear operation, completion change, and note change is
saved to `annotations.json`. Writes use an atomic pending-file replacement.

It is safe to:

1. Leave frames incomplete.
2. Stop the server with `Ctrl+C`.
3. Shut down the computer.
4. Run the same `serve` command later.

The UI resumes from the existing dataset. Do not rerun `init` against the same
output directory; initialization deliberately refuses to overwrite it.

Back up the complete `data\fretboard-keypoints` folder periodically. The
sampled images, manifest, and annotations are all required.

## Annotate on macOS

Copy the repository working tree and the self-contained
`data/fretboard-keypoints` folder to the Mac. The original videos are not
required for annotation because sampled images are included.

Create an annotation environment from the repository root:

```bash
python3.12 -m venv .venv-annotation && source .venv-annotation/bin/activate && python -m pip install opencv-python
```

Start or resume the UI:

```bash
python -m scripts.fretboard_annotation serve --dataset data/fretboard-keypoints
```

Stop it with `Ctrl+C`, then copy the entire annotated dataset folder back to
the Windows repository.

## Export on Windows

After copying the annotated folder back:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation export --dataset data\fretboard-keypoints
```

This writes YOLO pose labels under `labels/` and a Windows-local `data.yaml`.
Only frames marked complete are exported. Reviewed frames with no available
keypoints are exported as negative examples.

The seven-keypoint labels use normalized coordinates, so moving the dataset
between macOS and Windows does not change them. Source images remain at native
resolution; model training can use 4K or a lower `imgsz` without relabeling.
Horizontal augmentation preserves each physical string identity; unlike
anatomical left/right landmarks, string 6 and string 1 must not exchange label
indices when an image is mirrored.

## Expected effort

A production target may reach approximately 800-1,150 diverse images:

- 300-400 initial seed frames
- 400-600 model-assisted corrections
- 100-150 independently reviewed validation frames

Seven-point labeling is expected to take roughly 12-24 total hours, spread
across resumable sessions. The selector removes repeated views before labeling.

Do not label the full target before testing whether the labels work. Stop after
the first 300-400 diverse frames and train a pilot detector. Add more labels
only for failure modes found on held-out videos.

## Next steps after manual annotation

### 1. Export completed frames on Windows (complete)

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation export --dataset data\fretboard-keypoints
```

The command writes `labels/` and `data.yaml`. Incomplete frames remain saved in
`annotations.json` but are not exported. Fully reviewed guitar-absent frames
are exported as negative examples.

### 2. Audit the exported dataset (complete)

Before training:

- Confirm all outer-string point definitions use the same physical string ordering.
- Inspect every validation image independently from training images.
- Keep all frames from one source video in one split.
- Check that unavailable off-screen points were not guessed.
- Back up the complete dataset folder.

The current exporter validates coordinate ranges, point order at visible
anchors, and degenerate pairs. A separate visual dataset-audit command is
still to be implemented before production training.

### 3. Train a pilot YOLO pose model (current)

The images and labels preserve native 4K coordinates. Full 4K training uses
`imgsz=3840`, the nano pose architecture and batch size one.

Install pinned training dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-fretboard.txt
```

Validate the request without training:

```powershell
.\.venv\Scripts\python.exe -m scripts.fretboard_training --dataset data\fretboard-keypoints --output runs\fretboard-detector --device 0 --dry-run
```

The historical M3 Max CPU fallback command is:

```bash
caffeinate .venv/bin/python -m scripts.fretboard_training --dataset data/fretboard-keypoints --output runs/fretboard-detector --image-size 3840 --batch-size 1 --epochs 200 --patience 30 --device cpu --workers 6 2>&1 | tee runs/fretboard-detector-training.log
```

On a Windows machine with an NVIDIA GPU, use:

```powershell
.\.venv\Scripts\python.exe -m scripts.fretboard_training --dataset data\fretboard-keypoints --output runs\fretboard-detector --image-size 3840 --batch-size 1 --epochs 200 --patience 30 --device 0 2>&1 | Tee-Object -FilePath runs\fretboard-detector-training.log
```

Resume the current macOS CPU run from Ultralytics `last.pt` with:

```bash
caffeinate .venv/bin/python -m scripts.fretboard_training --dataset data/fretboard-keypoints --output runs/fretboard-detector --image-size 3840 --batch-size 1 --epochs 200 --patience 30 --device cpu --workers 6 --resume runs/fretboard-detector/weights/last.pt 2>&1 | tee runs/fretboard-detector-resume.log
```

Windows/NVIDIA resume:

```powershell
.\.venv\Scripts\python.exe -m scripts.fretboard_training --dataset data\fretboard-keypoints --output runs\fretboard-detector --image-size 3840 --batch-size 1 --epochs 200 --patience 30 --device 0 --resume runs\fretboard-detector\weights\last.pt 2>&1 | Tee-Object -FilePath runs\fretboard-detector-resume.log
```

### 4. Evaluate on held-out videos

Do not judge the model from annotation images or random neighboring frames.
Run it on complete videos that were excluded from training and review:

- Detection coverage on frames where the fretboard is visible
- Nut, fret-12, and bridge endpoint accuracy
- Correct string-6/string-1 orientation
- Close, medium, and wide camera views
- Partial and occluded fretboards
- Scene-cut reacquisition
- Per-frame stability after optical-flow tracking
- Six-string and inferred-fret overlays

The detector is ready for corpus use only when its visual overlays are
trustworthy on held-out videos. A high YOLO confidence score alone is not
sufficient.

### 5. Add hard examples instead of more duplicates

After the pilot:

1. Run it over unlabelled candidate frames.
2. Collect low-confidence detections, rejected geometry, missed fretboards,
   scene-cut failures, and unstable tracks.
3. Cluster similar failures.
4. Label representative failures in the same UI.
5. Retrain and compare against the unchanged held-out set.

Generate resumable model predictions for the selected dataset:

```powershell
.\.venv\Scripts\python.exe -m scripts.fretboard_annotation prefill --dataset data\fretboard-keypoints --model runs\fretboard-detector\weights\best.pt --device 0 --image-size 3840 --confidence 0.01
```

Then open the normal annotation UI. Predicted frames are purple in the progress
bar. The UI pre-fills all model-proposed points, outlines the original proposal
with dashed circles, and records each completed review as `accepted` when
unchanged or `corrected` after edits. Misses and partial predictions remain
reviewable, and every save is resumable through `predictions.json` and
`annotations.json`.

Repeat until additional labels no longer improve held-out video behavior.

### 6. Adopt the trained checkpoint

The final local checkpoint must be recorded with:

- SHA-256 digest
- Keypoint order and visibility convention
- Training dataset digest
- Ultralytics and PyTorch versions
- Training configuration
- Held-out evaluation report

Do not overwrite an earlier checkpoint or reinterpret a seven-point checkpoint
as the experimental 40-point model.

### 7. Process the video corpus

The production geometry pass will:

- Reset on every scene cut
- Use multiscale YOLO anchors
- Track geometry on intervening frames with optical flow
- Infer fret positions from nut, fret 12, and bridge geometry
- Infer six string paths from the outer-string anchors
- Retain MediaPipe hands on every frame
- Mask unavailable or invalid geometry
- Cache each completed video for safe resume

Add these top-level fields to the batch manifest:

```json
{"videoPython":"scripts\\video-evidence\\.venv\\Scripts\\python.exe","fretboardModel":"runs\\fretboard-detector\\weights\\best.pt","workers":16}
```

Install detector inference dependencies into the video environment:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m pip install -r requirements-fretboard.txt
```

Run corpus preparation yourself with live output and a saved log:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_training_data batch --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1 2>&1 | Tee-Object -FilePath runs\fretboard-corpus-preparation.log
```

Repeat the same command to resume completed stages.

### 8. Retrain and compare the transcriber

After geometry bundles are complete:

1. Regenerate the paired-video index.
2. Verify geometry and hand coverage for train and validation splits.
3. Train the joint audio/video transcriber from scratch.
4. Select the best checkpoint from held-out decoded-event metrics.
5. Compare audio-only, hands-only, and hands-plus-fretboard models on the same
   unseen songs.

Fretboard geometry is useful only if it improves held-out transcription; do
not assume higher visual coverage automatically improves tablature accuracy.

Generate and preflight the new configuration:

```powershell
.\.venv\Scripts\python.exe -m scripts.transcriber config --manifest data\releases\dataset-v1\manifest.json --video-index runs\video-evidence\batches\dataset-v1\paired-index.json --output runs\joint-v5-config.json
```

```powershell
.\.venv\Scripts\python.exe -m scripts.transcriber preflight --config runs\joint-v5-config.json --forward --output runs\joint-v5-preflight.json 2>&1 | Tee-Object -FilePath runs\joint-v5-preflight.log
```

Run training yourself. The default is a 100-epoch ceiling with decoded-event
early stopping and learning-rate reduction:

```powershell
.\.venv\Scripts\python.exe -m scripts.transcriber train --config runs\joint-v5-config.json --run-dir runs\joint-v5-training 2>&1 | Tee-Object -FilePath runs\joint-v5-training.log
```

Resume from the exact latest checkpoint:

```powershell
.\.venv\Scripts\python.exe -m scripts.transcriber train --config runs\joint-v5-config.json --run-dir runs\joint-v5-training --resume runs\joint-v5-training\latest.pt 2>&1 | Tee-Object -FilePath runs\joint-v5-resume.log
```

For ablations, copy the configuration to separate files and set
`video.model.experiment_mode` to `original-hands`, `role-gates`, or `geometry`.
Use a separate audio-only configuration without `video`. Never reuse a run
directory across modes.
