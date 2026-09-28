"""Local, channel-preserving features and explicit musical conditioning."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

from .gp_events import validate_provided_timing


class HarnessError(ValueError):
    """The harness cannot safely use the supplied input or configuration."""


BASE_CONDITIONING_DIM = 12
DOWNBEAT_CONDITIONING_DIM = 16


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 22050
    n_fft: int = 4096
    hop_length: int = 441
    n_mels: int = 96
    f_min: float = 35.0
    f_max: float = 10000.0

    def __post_init__(self):
        for name in ("sample_rate", "n_fft", "hop_length", "n_mels"):
            if type(getattr(self, name)) is not int:
                raise HarnessError(f"{name} must be an integer.")
        if not 8000 <= self.sample_rate <= 96000 or not 512 <= self.n_fft <= 16384 or not 1 <= self.hop_length <= self.n_fft or not 16 <= self.n_mels <= 256:
            raise HarnessError("Unsupported feature dimensions.")
        if type(self.f_min) not in (int, float) or type(self.f_max) not in (int, float) or not math.isfinite(self.f_min) or not math.isfinite(self.f_max) or not 0 < self.f_min < self.f_max <= self.sample_rate / 2:
            raise HarnessError("Feature frequency bounds must lie inside Nyquist.")

    @property
    def hop_seconds(self):
        return self.hop_length / self.sample_rate


def read_audio_window(path, start, stop, *, sample_rate=None, channels=None, sample_count=None):
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
        raise HarnessError("Audio windows require nonnegative start and exclusive stop samples.")
    with sf.SoundFile(Path(path)) as stream:
        if stop > len(stream):
            raise HarnessError("Audio window exceeds the retained clip.")
        for actual, expected, name in ((stream.samplerate, sample_rate, "sample rate"), (stream.channels, channels, "channels"), (len(stream), sample_count, "sample count")):
            if expected is not None and actual != expected:
                raise HarnessError(f"Audio {name} changed after manifest validation.")
        if not 1 <= stream.channels <= 8:
            raise HarnessError("Audio must contain one to eight channels.")
        rate = stream.samplerate
        stream.seek(start)
        samples = stream.read(stop - start, dtype="float32", always_2d=True)
    if len(samples) != stop - start or not np.isfinite(samples).all():
        raise HarnessError("Audio decoder returned incomplete or nonfinite PCM.")
    return samples, rate


def mel_bank(config):
    frequencies = torch.linspace(0, config.sample_rate / 2, config.n_fft // 2 + 1)
    low = 2595 * math.log10(1 + config.f_min / 700)
    high = 2595 * math.log10(1 + config.f_max / 700)
    boundaries = 700 * (torch.pow(10., torch.linspace(low, high, config.n_mels + 2) / 2595) - 1)
    left, center, right = boundaries[:-2], boundaries[1:-1], boundaries[2:]
    rising = (frequencies[:, None] - left) / (center - left)
    falling = (right - frequencies[:, None]) / (right - center)
    return torch.minimum(rising, falling).clamp_min(0) * (2 / (right - left))


def audio_features(samples, sample_rate, config=FeatureConfig()):
    if not isinstance(samples, np.ndarray) or samples.ndim != 2 or samples.dtype.kind != "f" or not len(samples) or not 1 <= samples.shape[1] <= 8:
        raise HarnessError("Expected nonempty floating audio samples by channels.")
    if type(sample_rate) is not int or sample_rate <= 0 or not np.isfinite(samples).all():
        raise HarnessError("Invalid sample rate or nonfinite audio.")
    if len(samples) / sample_rate > 30:
        raise HarnessError("Feature extraction is windowed; each call is limited to 30 seconds.")
    if sample_rate != config.sample_rate:
        divisor = math.gcd(sample_rate, config.sample_rate)
        samples = resample_poly(samples, config.sample_rate // divisor, sample_rate // divisor, axis=0).astype(np.float32)
    waveform = torch.from_numpy(np.array(samples.T, dtype=np.float32, copy=True))
    window = torch.hann_window(config.n_fft)
    spectrum = torch.stft(waveform, config.n_fft, config.hop_length, window=window, center=True, pad_mode="constant", return_complex=True)
    # Combining channel power retains opposite-phase stereo evidence.
    power = spectrum.abs().square().mean(dim=0) / window.sum().square()
    frames = math.ceil(len(samples) / config.hop_length)
    mel = power[:, :frames].T @ mel_bank(config)
    log_mel = 10 * torch.log10(mel.clamp_min(1e-10))
    features = (log_mel - log_mel.mean()) / log_mel.std(unbiased=False).clamp_min(1e-5)
    if not torch.isfinite(features).all():
        raise HarnessError("Feature extraction produced nonfinite values.")
    times = np.arange(frames, dtype=np.float64) * config.hop_seconds
    return features, times


def _quarter_positions(tempo_events, positions):
    positions = np.asarray(positions, dtype=np.float64)
    event_positions = np.array([event["position"] for event in tempo_events], dtype=np.float64)
    quarter_bpms = np.array([
        float(validate_provided_timing(event, [4, 4]))
        for event in tempo_events
    ])
    cumulative = np.zeros(len(tempo_events), dtype=np.float64)
    for index in range(1, len(tempo_events)):
        duration = event_positions[index] - event_positions[index - 1]
        previous = quarter_bpms[index - 1]
        following = quarter_bpms[index] if tempo_events[index - 1].get("linear", False) else previous
        cumulative[index] = cumulative[index - 1] + duration * (previous + following) / 120
    indices = np.searchsorted(event_positions, positions, side="right") - 1
    elapsed = positions - event_positions[indices]
    bpm = quarter_bpms[indices]
    slopes = np.zeros(len(tempo_events), dtype=np.float64)
    for index, event in enumerate(tempo_events[:-1]):
        if event.get("linear", False):
            slopes[index] = (quarter_bpms[index + 1] - quarter_bpms[index]) / (
                event_positions[index + 1] - event_positions[index]
            )
    return cumulative[indices] + (bpm * elapsed + .5 * slopes[indices] * elapsed ** 2) / 60


def _phase_features(
    meter_events, positions, quarter_positions, meter_quarters,
    first_downbeat_position, first_downbeat_quarter,
):
    meter_positions = np.array([event["position"] for event in meter_events], dtype=np.float64)
    meter_indices = np.searchsorted(meter_positions, positions, side="right") - 1
    anchors = np.where(
        meter_quarters[meter_indices] > first_downbeat_quarter,
        meter_quarters[meter_indices],
        first_downbeat_quarter,
    )
    relative = quarter_positions - anchors
    beat_lengths = np.empty(len(positions), dtype=np.float64)
    bar_lengths = np.empty(len(positions), dtype=np.float64)
    for index, meter_index in enumerate(meter_indices):
        numerator, denominator = meter_events[int(meter_index)]["timeSignature"]
        beat_lengths[index] = 4 / denominator
        bar_lengths[index] = numerator * beat_lengths[index]
    phases = np.column_stack((
        np.sin(2 * np.pi * relative / beat_lengths),
        np.cos(2 * np.pi * relative / beat_lengths),
        np.sin(2 * np.pi * relative / bar_lengths),
        np.cos(2 * np.pi * relative / bar_lengths),
    ))
    if not np.isfinite(first_downbeat_position) or first_downbeat_position < 0 or not np.isfinite(phases).all():
        raise HarnessError("First full-bar downbeat must be a finite nonnegative timestamp.")
    return phases.astype(np.float32)


def conditioning_features(
    tuning, capo, tempo_events, meter_events, positions, *,
    first_full_bar_downbeat=None, score_quarters=None, first_full_bar_quarter=None,
):
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 1 or not len(positions) or not np.isfinite(positions).all() or np.any(np.diff(positions) < 0):
        raise HarnessError("Conditioning positions must be finite and nondecreasing.")
    if not isinstance(tuning, list) or len(tuning) != 6 or any(type(value) is not int or not 0 <= value <= 127 for value in tuning) or type(capo) is not int or not 0 <= capo <= 24 or any(value + capo > 127 for value in tuning):
        raise HarnessError("Provide six explicit pre-capo MIDI pitches and a valid full capo.")
    if not tempo_events or not meter_events:
        raise HarnessError("Explicit tempo and meter schedules are required.")
    for events in (tempo_events, meter_events):
        coordinates = [event["position"] for event in events]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in coordinates) or any(a >= b for a, b in zip(coordinates, coordinates[1:])) or coordinates[0] > positions[0]:
            raise HarnessError("Timing events need an initial value and strictly increasing finite positions.")
    for event in meter_events:
        validate_provided_timing(tempo_events[0], event["timeSignature"])
    bpm = []
    for i, event in enumerate(tempo_events):
        bpm.append(float(validate_provided_timing(event, meter_events[0]["timeSignature"])))
        if type(event.get("linear", False)) is not bool or event.get("linear", False) and i + 1 == len(tempo_events):
            raise HarnessError("A tempo ramp requires an explicit later endpoint.")
    tempo_positions = np.array([event["position"] for event in tempo_events])
    meter_positions = np.array([event["position"] for event in meter_events])
    ti = np.searchsorted(tempo_positions, positions, side="right") - 1
    mi = np.searchsorted(meter_positions, positions, side="right") - 1
    quarter_bpm = np.array(bpm)[ti]
    for index, event in enumerate(tempo_events[:-1]):
        if event.get("linear", False):
            mask = ti == index
            ratio = (positions[mask] - tempo_positions[index]) / (tempo_positions[index + 1] - tempo_positions[index])
            quarter_bpm[mask] = bpm[index] + ratio * (bpm[index + 1] - bpm[index])
    include_phase = first_full_bar_downbeat is not None
    if include_phase:
        if (
            type(first_full_bar_downbeat) not in (int, float)
            or not math.isfinite(first_full_bar_downbeat)
            or first_full_bar_downbeat < 0
        ):
            raise HarnessError("First full-bar downbeat must be a finite nonnegative timestamp.")
        first_full_bar_downbeat = float(first_full_bar_downbeat)
        if score_quarters is None:
            phase_quarters = _quarter_positions(tempo_events, positions)
            anchor_quarter = float(_quarter_positions(tempo_events, [first_full_bar_downbeat])[0])
            meter_quarters = _quarter_positions(
                tempo_events, [event["position"] for event in meter_events],
            )
        else:
            phase_quarters = np.asarray(score_quarters, dtype=np.float64)
            if (
                phase_quarters.shape != positions.shape
                or not np.isfinite(phase_quarters).all()
                or np.any(np.diff(phase_quarters) < 0)
                or type(first_full_bar_quarter) not in (int, float)
                or not math.isfinite(first_full_bar_quarter)
                or first_full_bar_quarter < 0
            ):
                raise HarnessError("Aligned score quarters and first full-bar score position are required.")
            anchor_quarter = float(first_full_bar_quarter)
            meter_quarters = np.array(
                [event["position"] for event in meter_events], dtype=np.float64,
            )
        phase = _phase_features(
            meter_events, positions, phase_quarters, meter_quarters,
            first_full_bar_downbeat, anchor_quarter,
        )
    elif score_quarters is not None or first_full_bar_quarter is not None:
        raise HarnessError("Score phase inputs require a first full-bar downbeat timestamp.")
    values = np.empty(
        (len(positions), DOWNBEAT_CONDITIONING_DIM if include_phase else BASE_CONDITIONING_DIM),
        dtype=np.float32,
    )
    values[:, :6] = (np.array(tuning) - 60) / 24
    values[:, 6] = capo / 12
    values[:, 7] = np.log2(quarter_bpm / 120)
    for index, (tempo_index, meter_index) in enumerate(zip(ti, mi)):
        unit = tempo_events[int(tempo_index)]["beatUnit"]
        meter = meter_events[int(meter_index)]["timeSignature"]
        values[index, 8:12] = unit[0] / 8, unit[1] / 16, meter[0] / 12, meter[1] / 16
    if include_phase:
        values[:, BASE_CONDITIONING_DIM:] = phase
    return torch.from_numpy(values)
