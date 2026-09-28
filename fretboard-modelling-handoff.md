# Fretboard modelling handoff

## Status

- Branch: `fretboard-modelling`
- Local only; nothing pushed
- Implementation plan: complete
- Current stage: 4K detector training on the M3 Max CPU
- Full 400-frame seven-point annotation: complete
- Full root suite: 715 passed
- Focused vision suites: passed
- Untracked plan: `_plan-fretboard-modelling.md`

## Implemented

- Native-resolution diverse-frame selection
- Resumable Windows/macOS browser annotation UI
- Seven-point YOLO pose export with six outer geometry anchors
- Reproducible 4K detector training wrapper
- Multiscale scene-cut-aware detector acquisition
- Per-frame optical-flow tracking
- Deterministic fret and six-string reconstruction
- Schema-5 D233 paired-video bundles
- Architecture-6 role-aware joint model
- Trusted author-created GP voice supervision
- Plucking-thumb/code-voice evaluation metrics
- Exact audio fallback
- Full-voice and single-voice GP exports
- Sixteen-video resumable corpus preparation
- 100-epoch ceiling
- Decoded-event early stopping
- Learning-rate reduction
- Audio, original-hands, role-gates, and geometry experiment modes
- Historical architecture-4/5 inference loading
- Required first-full-bar downbeat conditioning for new 16-feature models
- Historical 12-feature checkpoint inference compatibility
- Identity-preserving horizontal augmentation for physical string keypoints

## Private artifacts

The following are Git-ignored:

- `data\fretboard-keypoints`
- `runs\fretboard-selection.log`
- `data\fretboard-selection-cache`
- Local detector weights and training runs
- Prepared corpus bundles and training outputs

The completed balanced pilot is at `data\fretboard-keypoints` and contains:

- 400 frames
- 202 represented source videos
- 322 train frames
- 32 validation frames
- 46 test frames
- 260 two-hand frames
- 120 one-hand frames
- 20 intentional zero-hand negatives
- No source video crossing splits
- Native source resolutions
- Song titles, source timestamps, and timestamped YouTube links

The previous two-hand-heavy pilot is archived at
`data\fretboard-keypoints-two-hand-backup`. Per-video candidate caches remain
under `data\fretboard-selection-cache` for future selection runs.

## Annotation contract

Label seven points:

1. Nut contact — low E / string 6
2. Nut contact — high E / string 1
3. Silver fret-12 wire — low E / string 6
4. Silver fret-12 wire — high E / string 1
5. Bridge saddle contact — low E / string 6
6. Bridge saddle contact — high E / string 1
7. Silver fret-5 wire — center

Fret labels mean the metal wire, not the fret space.

Each point may be available, occluded, or unavailable. Complete means the
frame was fully reviewed; off-screen points may remain unavailable.

## Remaining required steps

### 1. Annotate the pilot dataset (complete)

Windows:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation serve --dataset data\fretboard-keypoints
```

macOS:

```bash
python -m scripts.fretboard_annotation serve --dataset data/fretboard-keypoints
```

The UI autosaves and resumes through `annotations.json`.

### 2. Export completed annotations (complete)

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation export --dataset data\fretboard-keypoints
```

### 3. Install detector dependencies

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-fretboard.txt
```

macOS:

```bash
.venv/bin/python -m pip install -r requirements-fretboard.txt
```

### 4. Validate the detector-training request

Windows/NVIDIA:

```powershell
.\.venv\Scripts\python.exe -m scripts.fretboard_training --dataset data\fretboard-keypoints --output runs\fretboard-detector --device 0 --dry-run
```

macOS CPU:

```bash
.venv/bin/python -m scripts.fretboard_training --dataset data/fretboard-keypoints --output runs/fretboard-detector --device cpu --dry-run
```

### 5. Train the 4K detector (current)

The current run uses the M3 Max CPU because pinned Ultralytics `8.3.102`
explicitly warns that Apple MPS has a known pose bug in training, validation and
prediction. Do not switch this pose run to `mps`.

Current macOS command:

```bash
caffeinate .venv/bin/python -m scripts.fretboard_training --dataset data/fretboard-keypoints --output runs/fretboard-detector --image-size 3840 --batch-size 1 --epochs 200 --patience 30 --device cpu --workers 6 2>&1 | tee runs/fretboard-detector-training.log
```

Windows/NVIDIA command:

```powershell
.\.venv\Scripts\python.exe -m scripts.fretboard_training --dataset data\fretboard-keypoints --output runs\fretboard-detector --image-size 3840 --batch-size 1 --epochs 200 --patience 30 --device 0 2>&1 | Tee-Object -FilePath runs\fretboard-detector-training.log
```

Windows/NVIDIA resume:

```powershell
.\.venv\Scripts\python.exe -m scripts.fretboard_training --dataset data\fretboard-keypoints --output runs\fretboard-detector --image-size 3840 --batch-size 1 --epochs 200 --patience 30 --device 0 --resume runs\fretboard-detector\weights\last.pt 2>&1 | Tee-Object -FilePath runs\fretboard-detector-resume.log
```

macOS CPU resume:

```bash
caffeinate .venv/bin/python -m scripts.fretboard_training --dataset data/fretboard-keypoints --output runs/fretboard-detector --image-size 3840 --batch-size 1 --epochs 200 --patience 30 --device cpu --workers 6 --resume runs/fretboard-detector/weights/last.pt 2>&1 | tee runs/fretboard-detector-resume.log
```

### 6. Review held-out complete-video overlays

Check:

- Nut, fret-12, and bridge endpoint accuracy
- String-6/string-1 orientation
- Close, medium, wide, partial, and occluded views
- Scene-cut reacquisition
- Optical-flow stability
- Missing-geometry masks

Do not accept the detector from confidence scores alone.

### 7. Add hard examples

- Select missed, rejected, unstable, and low-confidence views.
- Deduplicate repeated failures.
- Generate proposals with `python -m scripts.fretboard_annotation prefill`.
- Review purple predicted frames in the normal annotation UI.
- Accept unchanged predictions or adjust points before completing the frame.
- Retrain against the unchanged held-out split.
- Stop when held-out behavior no longer improves.

### 8. Configure corpus preparation

Add these top-level batch manifest fields:

```json
{"videoPython":"scripts\\video-evidence\\.venv\\Scripts\\python.exe","fretboardModel":"runs\\fretboard-detector\\weights\\best.pt","workers":16}
```

Install detector dependencies into the video environment:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m pip install -r requirements-fretboard.txt
```

### 9. Prepare the full corpus

Run this manually:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_training_data batch --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1 2>&1 | Tee-Object -FilePath runs\fretboard-corpus-preparation.log
```

Repeat the same command to reuse completed stages. Follow explicit review or
reset instructions from the log.

### 10. Generate and preflight joint training

```powershell
.\.venv\Scripts\python.exe -m scripts.transcriber config --manifest data\releases\dataset-v1\manifest.json --video-index runs\video-evidence\batches\dataset-v1\paired-index.json --output runs\joint-v5-config.json
```

```powershell
.\.venv\Scripts\python.exe -m scripts.transcriber preflight --config runs\joint-v5-config.json --forward --output runs\joint-v5-preflight.json 2>&1 | Tee-Object -FilePath runs\joint-v5-preflight.log
```

### 11. Train the joint model

Run this manually:

```powershell
.\.venv\Scripts\python.exe -m scripts.transcriber train --config runs\joint-v5-config.json --run-dir runs\joint-v5-training 2>&1 | Tee-Object -FilePath runs\joint-v5-training.log
```

Resume:

```powershell
.\.venv\Scripts\python.exe -m scripts.transcriber train --config runs\joint-v5-config.json --run-dir runs\joint-v5-training --resume runs\joint-v5-training\latest.pt 2>&1 | Tee-Object -FilePath runs\joint-v5-resume.log
```

### 12. Run ablations

Use separate configuration and run directories:

- Audio only: omit `video`
- Original hands: `video.model.experiment_mode = "original-hands"`
- Role gates: `video.model.experiment_mode = "role-gates"`
- Full geometry: `video.model.experiment_mode = "geometry"`

Keep splits, seeds, and budgets identical. Select from held-out decoded-event
metrics, not training loss.

## Optional work not yet implemented

- Automatic hard-example queue generation from a trained detector
- A plucking-thumb decoder preference

## Voice convention

- Code voice 0 / displayed voice 1: primarily upper/non-thumb material
- Code voice 1 / displayed voice 2: primarily plucking-thumb bass
- All current training GP files are trusted author-created scores
- No flattened GP files are present in training
- Fretting-thumb evidence does not imply a voice
- Both full-voice and single-voice exports remain enabled

## Completion criteria

- Held-out detector overlays are trustworthy.
- Geometry never crosses cuts or unavailable evidence.
- Schema-5 bundles pass validation.
- Joint training starts from random initialization.
- Full geometry improves held-out transcription without unacceptable
  regressions.
