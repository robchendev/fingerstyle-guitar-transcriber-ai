from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf
import torch

from scripts.transcriber_audio import FeatureConfig, HarnessError, audio_features, conditioning_features, read_audio_window
from scripts.dataset_io import ROOT


class TranscriberAudioTests(unittest.TestCase):
    def test_frame_centers_low_bass_and_antiphase_stereo_are_preserved(self):
        config = FeatureConfig()
        times = np.arange(config.sample_rate) / config.sample_rate
        wave = np.sin(2 * np.pi * 49 * times).astype(np.float32)
        stereo = np.column_stack((wave, -wave))
        before = stereo.copy()
        features, frames = audio_features(stereo, config.sample_rate, config)
        mono, _ = audio_features(wave[:, None], config.sample_rate, config)
        self.assertEqual(features.shape, (50, 96))
        self.assertEqual(frames[-1], .98)
        self.assertGreater(float(features.std()), .9)
        torch.testing.assert_close(features, mono)
        np.testing.assert_array_equal(stereo, before)

    def test_native_sample_window_is_read_once_without_source_crop_or_downmix(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "synthetic.flac"
            original = np.column_stack((np.linspace(-.5, .5, 4800), np.linspace(.25, -.25, 4800))).astype(np.float32)
            sf.write(path, original, 48000, subtype="PCM_24")
            samples, rate = read_audio_window(path, 100, 2300, sample_rate=48000, channels=2, sample_count=4800)
            self.assertEqual(samples.shape, (2200, 2))
            self.assertEqual(rate, 48000)
            np.testing.assert_allclose(samples, original[100:2300], atol=2e-7)
            with self.assertRaises(HarnessError):
                read_audio_window(path, 0, 5000)
            features, _ = audio_features(samples, rate)
            self.assertEqual(features.shape[1], 96)

    def test_conditioning_keeps_beat_units_and_meter_changes_without_score_phase(self):
        tempos = [{"position": 0, "bpm": 40, "beatUnit": [3, 8]}, {"position": 2, "bpm": 120, "beatUnit": [1, 4]}]
        meters = [{"position": 0, "timeSignature": [6, 8]}, {"position": 3, "timeSignature": [4, 4]}]
        features = conditioning_features([40, 45, 50, 55, 59, 64], 2, tempos, meters, [0, 1, 2, 3])
        self.assertEqual(features.shape, (4, 12))
        self.assertEqual(features[0, 7], -1)
        self.assertEqual(features[2, 7], 0)
        self.assertNotEqual(features[1, 8], features[2, 8])
        self.assertNotEqual(features[2, 11], features[3, 11])
        shifted = conditioning_features([40, 45, 50, 55, 59, 64], 2, tempos, meters, [.5, 1.5, 2.5, 3.5])
        torch.testing.assert_close(features, shifted)

    def test_reviewed_downbeat_adds_cyclic_beat_and_bar_phase(self):
        tempo = [{"position": 0, "bpm": 60, "beatUnit": [1, 4]}]
        meter = [{"position": 0, "timeSignature": [4, 4]}]
        values = conditioning_features(
            [40, 45, 50, 55, 59, 64], 0, tempo, meter, [0, .5, 1, 2],
            first_full_bar_downbeat=1,
        )
        self.assertEqual(values.shape, (4, 16))
        torch.testing.assert_close(values[2, 12:], torch.tensor([0., 1., 0., 1.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(values[0, 12:], torch.tensor([0., 1., -1., 0.]), atol=1e-6, rtol=0)
        aligned = conditioning_features(
            [40, 45, 50, 55, 59, 64], 0, tempo, meter, [0, .5, 1, 2],
            first_full_bar_downbeat=3.25,
            score_quarters=[0, .5, 1, 2],
            first_full_bar_quarter=1,
        )
        torch.testing.assert_close(values[:, 12:], aligned[:, 12:], atol=1e-6, rtol=0)

    def test_nominal_ramps_and_invalid_inputs_are_explicit(self):
        tempo = [{"position": 0, "bpm": 60, "beatUnit": [1, 4], "linear": True}, {"position": 4, "bpm": 120, "beatUnit": [1, 4]}]
        values = conditioning_features([40, 45, 50, 55, 59, 64], 0, tempo, [{"position": 0, "timeSignature": [4, 4]}], [0, 2, 4])
        self.assertAlmostEqual(float(values[1, 7]), np.log2(.75), places=6)
        phased = conditioning_features(
            [40, 45, 50, 55, 59, 64], 0, tempo,
            [{"position": 0, "timeSignature": [4, 4]}, {"position": 4, "timeSignature": [3, 4]}],
            [0, 2, 4], first_full_bar_downbeat=0,
        )
        torch.testing.assert_close(phased[1, 12:14], torch.tensor([0., -1.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(phased[2, 12:], torch.tensor([0., 1., 0., 1.]), atol=1e-6, rtol=0)
        with self.assertRaises(HarnessError):
            conditioning_features([40] * 6, 0, tempo[:1], [{"position": 0, "timeSignature": [4, 4]}], [0])
        with self.assertRaises(HarnessError):
            conditioning_features([40] * 6, 0, tempo, [{"position": 0, "timeSignature": [4, 4]}], [0], first_full_bar_downbeat=-1)
        with self.assertRaises(HarnessError):
            audio_features(np.full((100, 1), np.nan, dtype=np.float32), 22050)
        with self.assertRaises(HarnessError):
            FeatureConfig(f_min=0)


if __name__ == "__main__":
    unittest.main()
