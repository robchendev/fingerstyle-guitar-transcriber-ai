# Fingerstyle Guitar Transcriber AI

Turn fingerstyle guitar performances into editable Guitar Pro `.gp` drafts using audio and hand tracking. Run the included PyTorch model locally or train one on your own performances and scores.

Use trimmed local videos. Training also needs matching single-track, six-string scores in modern `.gp` format. For transcription, supply the tuning before applying a capo, capo fret, tempo, beat unit and time signature. Local recordings, scores and outputs are excluded from Git.

## Demo

Source performance: [YouTube video](https://www.youtube.com/watch?v=4yivUKdHd4A). This video was not part of the model's training data.

Performance video, hand tracking and the audio/video features used by the model:

https://github.com/user-attachments/assets/4c803626-4edb-4f6f-accf-187b5abe6597

Generated Guitar Pro playback for the same passage, bars 25-36:

https://github.com/user-attachments/assets/232375f6-d8c2-4dfa-8a6a-5bf320e48e82

**User-configurable settings used for this demo**

| Parameter | Value |
| --- | --- |
| Note cutoff (`--note-cutoff`) | `0.78`, keeping scores strictly above the cutoff |
| Thumb-slap X cutoff (`--x-cutoff`) | `0.10` |
| Other percussion cutoff | `0.80` |
| Brush, arpeggio and pick-stroke cutoffs | `0.995` for detection, `0.98` for including individual notes |
| Note effects, connections and grace-note cutoffs | `0.99` |
| Harmonics | Excluded |
| Tuning before capo, strings 6 to 1 | `B1 F#2 B2 F#3 B3 E4` |
| Capo | Fret `1` |
| Tempo | `122 BPM`, quarter-note beat (`1/4`) |
| Time signature | `4/4` |
| Plucking hand on screen | Left |
| Rhythm policy | `fingerstyle` |

The bundled model in [`models/`](models/) struggles with complex melodies and was trained on a limited dataset. Training on a larger, high-quality dataset may improve results; see the [training guide](docs/TRAINING.md).

## Setup

Use Python 3.12 and install FFmpeg/FFprobe on `PATH`. Run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt -r requirements-rhythm.txt
.\.venv\Scripts\python.exe -m pip install --no-deps beat-this==1.1.0
python -m venv scripts\video-evidence\.venv
.\scripts\video-evidence\.venv\Scripts\python.exe -m pip install -r scripts\video-evidence\requirements.txt
.\scripts\video-evidence\.venv\Scripts\python.exe scripts\video-evidence\cli.py provision-hands
New-Item -ItemType Directory -Force .tools\models
Invoke-WebRequest 'https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt' -OutFile '.tools\models\beat-this-final0.ckpt'
```

The included transcription model is at [`models/transcriber.pt`](models/transcriber.pt). Setup also downloads MediaPipe for hand tracking and Beat This! for beat detection. Processing runs locally.

## Transcribe

**Important: The video must start exactly on the first beat of the song at timestamp 0.** Trim any lead-in, silence or count-in before running transcription, keeping the audio and video synchronized.

Supply a GP or GPT template for the page layout. Exported files clear the template's artist, arranger and lyricist credits.

```powershell
.\.venv\Scripts\python.exe .\transcribe_video.py --video 'data\inputs\performance.mp4' --checkpoint 'models\transcriber.pt' --template 'data\template.gpt' --beat-checkpoint '.tools\models\beat-this-final0.ckpt' --plucking-screen-side left --note-cutoff 0.8 --x-cutoff 0.3 --output-directory 'runs\transcription' --tuning E2 A2 D3 G3 B3 E4 --capo 0 --bpm 120 --beat-unit 1/4 --time-signature 4/4 --first-full-bar-downbeat 0
```

List tuning from string 6 to string 1. Set the plucking side to its position on screen, not the player's handedness. The command extracts audio, tracks hands and exports full-voice and single-voice GP drafts. It reports progress and any timing corrections needed. Repeat the same command to resume; changed inputs require a new output directory.

Set `--first-full-bar-downbeat` to the audio timestamp of beat 1 in the first
complete bar after any pickup. Use `0` when the recording starts on that
downbeat. New downbeat-conditioned checkpoints require this input; historical
checkpoints ignore it.

Adjust export cutoffs without rerunning the model or processing the video:

```powershell
.\.venv\Scripts\python.exe .\reexport_gp.py --source-run 'runs\transcription' --note-cutoff 0.85 --x-cutoff 0.35 --output-directory 'runs\reexport'
```

Both commands support `--dry-run` to preview their settings. A cutoff of `0.8` does not mean 80% accuracy. Lowering a re-export cutoff cannot recover notes already discarded by the model.

## Train

**Important: Every training video must start exactly on the first beat of the song at timestamp 0.** Trim the lead-in before preparing the dataset, keeping the audio and video synchronized.

See [training with your own videos and scores](docs/TRAINING.md) for data preparation, review and training. To transcribe with your trained model, pass its `best-events.pt` or `latest.pt` to `--checkpoint`. The bundled model cannot resume training.

The Eddie corpus has a dedicated
[tuning/capo correction handoff](docs/CAPO_NORMALIZATION_HANDOFF.md) for
replacing convenience-normalized GP settings with the physical tuning and capo
filmed in each training video before rebuilding the release.

To transfer a reviewed private corpus and detector checkpoint from macOS for
GPU preparation and joint training, use the
[Google Colab workflow](docs/COLAB_TRAINING.md).

## Limits

Review and edit the generated notes, fingerings, rhythms and techniques. Hand tracking measures posture and movement, not exact string or fret contact.

Partial capos, retuning during a recording and multi-instrument scores are unsupported. Single-voice drafts simplify overlapping parts. Full-voice drafts retain overlapping note durations, but may group them differently from a manually written score.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
.\scripts\video-evidence\.venv\Scripts\python.exe -m unittest discover -s tests\video -t .
```

## License

[MIT](LICENSE).
