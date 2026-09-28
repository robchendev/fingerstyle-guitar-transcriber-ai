# Local training

Complete the [setup instructions](../README.md#setup), activate the root Python environment with `.\.venv\Scripts\Activate.ps1`, and run these commands from the repository root.

Use matching `.gp` scores and already-trimmed local performance videos. Audio is extracted automatically. If you supply an `audio` file instead, it must match the performance. Videos with multiple audio tracks or timing errors require a separate audio file and alignment review.

**Important: Every video must start exactly on the first beat of the song at timestamp 0.** Trim any lead-in, silence or count-in before preparing the dataset. Keep the audio and video synchronized, including any separately supplied audio.

Released training targets record the first full-bar downbeat separately. For a
pickup, it is derived from the normalized GP pickup measure and reviewed
score-to-audio alignment; otherwise it is the first measure downbeat. New joint
training conditions each frame on cyclic beat and bar phase from this anchor.

### Correct filmed tuning and capo before release

The source GP must describe the tuning and physical full-capo fret visible in
the performance. Do not preserve a convenience normalization such as replacing
half-step-down/capo-2 with standard/capo-1. These settings have the same
sounding pitch at a given fret, but they do not have the same physical
fretboard position and therefore conflict with video supervision.

For the Eddie corpus, use the
[tuning/capo correction handoff](CAPO_NORMALIZATION_HANDOFF.md). Apply reviewed
corrections to each owned `raw.gp`, invalidate its existing preparation, repeat
score/alignment review and publish a new immutable release. Never edit frozen
release payloads or generated `normalized.gp` files in place.

## Prepare and review

Save this example as `runs\batch.json` and replace its file paths and plucking-hand screen sides. Paths can be absolute or relative to the repository. Set aside separate groups for validation, which evaluates the model during training. Keep related arrangements and copies of the same recording in the same group and split.

```json
{
  "schemaVersion": 1,
  "kind": "paired-preparation-batch",
  "workspace": "data",
  "releaseVersion": "dataset-v1",
  "acceptConventions": true,
  "handModel": "runs\\video-evidence\\models\\hand_landmarker.task",
  "records": [
    {"id": "pair-a", "groupId": "group-a", "split": "train", "gp": "inputs\\pair-a.gp", "video": "inputs\\pair-a-trimmed.mp4", "pluckingScreenSide": "left", "voiceSupervisionPolicy": "native-multivoice"},
    {"id": "pair-b", "groupId": "group-b", "split": "validation", "gp": "inputs\\pair-b.gp", "video": "inputs\\pair-b-trimmed.mp4", "pluckingScreenSide": "right", "voiceSupervisionPolicy": "native-multivoice"}
  ]
}
```

Set `acceptConventions` to `true` only if your scores use `O` for wrist thump, plain `X` for thumb slap and ghost `X` for percussive hit. Unrecognized markings stay marked as unknown. Review `rules.json` for notation differences. After changing imported files or rules, run `invalidate --id ID --reason TEXT`, then prepare and review again. Previously released datasets stay unchanged.

```powershell
python -m scripts.prepare_training_data batch --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1
python -m scripts.prepare_training_data batch-status --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1
python -m scripts.prepare_training_data --workspace data review --id pair-a --cue first-attack --cue end
```

Check `next-actions.txt`, each pair's `review-report.json`, `notation.json` and `normalized.gp` against the recording. Approve usable audio ranges in seconds and adjust the anchors matching score positions to audio times. Replace the example times below with your reviewed timings. Add anchors to correct timing drift or use `--exclude-range START:END` to leave out a section. `--acknowledge-uncertainty` accepts the reported unknowns without resolving them.

```powershell
python -m scripts.prepare_training_data batch-review --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1 --id pair-a --reviewer reviewer --accept-score --range 0.4:16.4 --anchor 1=0.4 --anchor end=16.4 --acknowledge-uncertainty
python -m scripts.prepare_training_data batch-review --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1 --id pair-b --reviewer reviewer --accept-score --range 0.4:16.4 --anchor 1=0.4 --anchor end=16.4 --acknowledge-uncertainty
python -m scripts.prepare_training_data batch-release --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1 --reviewer reviewer
python -m scripts.prepare_training_data batch --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1
```

Both training and validation need approved, usable sections. Follow any requests for audio/video alignment or camera-shot review using `batch-review --alignment-offset SECONDS` or `batch-review --accept-shots`, then rerun the batch. A `ready` result means preparation is complete, not that the labels are musically correct. Video training also needs usable hand-tracking results.

## Train from scratch

```powershell
python -m scripts.transcriber config --manifest data\releases\dataset-v1\manifest.json --video-index runs\video-evidence\batches\dataset-v1\paired-index.json --output runs\joint-config.json
python -m scripts.transcriber preflight --config runs\joint-config.json --forward --output runs\joint-preflight.json
python -m scripts.transcriber train --config runs\joint-config.json --run-dir runs\joint-training
```

Set the device, thread count, batch size and number of epochs in the generated configuration before training. `preflight` checks the setup. Use `--max-hours` to set a time limit.

Validation needs usable labels for notes or percussion and at least one supported technique, such as a bend, hammer-on, slide or grace note. Training cannot select its best model from notes-only validation data, even if `preflight` passes.

### Voice supervision and role-biased video

This corpus contains original author-created GP files, not flattened training
exports. Existing and new records therefore default to trusted
`native-multivoice` supervision. Code voice 0 (displayed voice 1) normally
contains upper/non-thumb material; code voice 1 (displayed voice 2) normally
contains plucking-thumb bass material. Future imported flattened files may
explicitly use `flattened-or-unknown`, which masks only voice loss while
retaining other trusted labels.

Video architecture 5 keeps the schema-4 D194 numeric input unchanged and adds
separate learnable fretting/plucking feature-group scales. Fretting starts
biased toward non-thumb fingertip positions; plucking starts biased toward
thumb and fingertip motion. Anonymous hands remain neutral. These are
initialization priors, not fixed weights or confidence estimates. Frames with
no usable video use the acoustic model output exactly and do not update visual
fusion parameters.

String/fret proximity, landmark reliability and any thumb-to-voice decoder
preference require separately versioned inputs or independently validated
evidence and are not part of architecture 5.

New fretboard-aware training uses video architecture 6 with schema-5 D233
inputs. D194 remains the unchanged prefix. Appended channels contain masked
fingertip scale/string/fret evidence, geometry-relative velocity and geometry
quality. New joint training starts from random initialization. Historical
architecture-4/5 checkpoints remain inference-only.

## Label a fretboard detector

The local annotation UI samples source frames without resizing them. Its six
ordered points are the outer sixth- and first-string positions at the nut,
twelfth fret and bridge. A reviewed frame may mark any point unavailable when
it is outside the shot; do not estimate hidden bridge or nut positions.

Create a gitignored dataset from explicit videos or a directory tree:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation init --video-directory runs\video-evidence\sources --frames-per-video 5 --output data\fretboard-keypoints
```

Start the localhost-only browser UI. It autosaves each edit and resumes the
same dataset:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation serve --dataset data\fretboard-keypoints
```

Use the mouse wheel to zoom, middle/right drag to pan, keys `1` through `7` to
select a point, `Q`/`W`/`E` for available/occluded/unavailable status, and
Enter to complete and advance. `Complete` means the frame was fully reviewed,
not that all seven landmarks are visible.

Export completed annotations to YOLO pose labels and `data.yaml`:

```powershell
.\scripts\video-evidence\.venv\Scripts\python.exe -m scripts.fretboard_annotation export --dataset data\fretboard-keypoints
```

Images retain their native resolution and labels use normalized coordinates.
The eventual training `imgsz` is a separate model-training choice; native 4K
sources may be trained at 4K or downscaled without relabeling.

### Reading fretboard-detector training output

Ultralytics reports five training losses; lower is better for each:

- `box_loss`: fretboard bounding-box localization (`box` is shorthand).
- `pose_loss`: the seven keypoint coordinate error (`pose` is shorthand for
  the keypoint arrangement).
- `kobj_loss`: keypoint objectness, meaning whether each keypoint is present.
- `cls_loss`: fretboard class/classification confidence (`cls` is shorthand).
- `dfl_loss`: Distribution Focal Loss, which refines the probability
  distributions used for bounding-box edges.

The terms solve different subproblems and need not improve together. Large
early classification drops, short plateaus and later box/pose improvement are
normal. Judge trends over multiple epochs, not one batch or three epochs.

After an epoch, the validation table reports box and pose precision (`P`),
recall (`R`), `mAP50` and `mAP50-95`; higher is better. Pose `mAP50-95` is the
main summary for the seven landmarks. The run directory also contains
`results.csv` and `results.png` with training and validation losses. Improving
training loss with persistently declining validation mAP indicates
overfitting.

Alternatively, `python -m scripts.prepare_training_data batch-train --manifest runs\batch.json --output-directory runs\video-evidence\batches\dataset-v1 --epochs 20 --device cpu --cpu-threads 4` prepares data and trains the audio/video model, pausing when review is needed. Rerunning the same command resumes training or reuses a completed result.

For audio-only training, use `python -m scripts.prepare_training_data` with `init`, `add --id ID --group GROUP --gp FILE --audio FILE`, `prepare --accept-conventions`, and `review`. Create the dataset with `release --version dataset-v1 --validation-group GROUP --reviewer reviewer --authorize-release`. Then run `python -m scripts.transcriber config --manifest data\releases\dataset-v1\manifest.json --output runs\audio-config.json` without `--video-index`, and use the same preflight/train commands with that configuration.

`models\transcriber.pt` is for transcription, **not resuming training**. Start new training without `--resume`. To continue an interrupted run, use `python -m scripts.transcriber train --config runs\joint-config.json --run-dir runs\joint-training --resume runs\joint-training\latest.pt`. Keep that run's source files, released dataset and prepared video data unchanged.
