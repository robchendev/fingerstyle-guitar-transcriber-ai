from dataclasses import FrozenInstanceError, asdict
import json
import math
import unittest

import torch

from scripts.transcriber_model import (
    FingerstyleTranscriber, HARMONIC_FRETS, HARMONIC_TYPES, LOSS_STAT_KEYS,
    LOSS_WEIGHTS, ModelConfig, PERCUSSION_TYPES, PRESENCE_CALIBRATION,
    decode_events, masked_loss,
)


def synthetic_outputs(batch=2, frames=5, *, requires_grad=False, architecture_version=1):
    shapes = {
        "note_onset_logits": (6,),
        "fret_logits": (6, 37),
        "pitch_logits": (6, 128),
        "voice_logits": (6, 4),
        "duration_log": (6,),
        "harmonic_logits": (6,),
        "harmonic_kind_logits": (6, 4),
        "harmonic_node_logits": (6, 6),
        "percussion_logits": (3,),
    }
    if architecture_version >= 2:
        shapes.update({
            "technique_logits": (4,),
            "technique_direction_logits": (4, 2),
            "technique_strings_logits": (4, 6),
        })
    if architecture_version >= 3:
        shapes.update({
            "connection_logits": (6, 5 if architecture_version == 4 else 10),
            "note_technique_logits": (6, 8 if architecture_version == 4 else 4),
            "bend_curve": (6, 7),
        })
    if architecture_version == 4:
        shapes.update({
            "grace_logits": (6,), "grace_fret_logits": (6, 37),
            "grace_mode_logits": (6, 2), "grace_transition_logits": (6, 5),
        })
    return {name: torch.zeros(batch, frames, *shape, requires_grad=requires_grad) for name, shape in shapes.items()}


def synthetic_targets(batch=2, frames=5, *, architecture_version=1):
    names = ("note_onset", "fret", "pitch", "voice", "duration_log", "harmonic",
             "harmonic_kind", "harmonic_node", "percussion")
    categorical = ("fret", "pitch", "voice", "harmonic_kind", "harmonic_node")
    targets = {
        name: torch.full((batch, frames, 3 if name == "percussion" else 6),
                         -999 if name in categorical else float("nan"),
                         dtype=torch.long if name in categorical else torch.float32)
        for name in names
    }
    if architecture_version >= 2:
        targets["technique"] = torch.full((batch, frames, 4), float("nan"))
        targets["technique_direction"] = torch.full((batch, frames, 4), -999, dtype=torch.long)
        targets["technique_strings"] = torch.full((batch, frames, 4, 6), float("nan"))
    masks = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in targets.items()}
    return targets, masks, torch.ones(batch, frames, dtype=torch.bool)


def event_outputs(frames, *, architecture_version=1):
    outputs = {name: value[0] for name, value in synthetic_outputs(1, frames, architecture_version=architecture_version).items()}
    for name in ("note_onset_logits", "harmonic_logits", "percussion_logits"):
        outputs[name].fill_(-12)
    if "technique_logits" in outputs:
        outputs["technique_logits"].fill_(-12)
        outputs["technique_strings_logits"].fill_(-12)
    if "note_technique_logits" in outputs:
        outputs["note_technique_logits"].fill_(-12)
    if "grace_logits" in outputs:
        outputs["grace_logits"].fill_(-12)
    outputs["duration_log"].fill_(math.log1p(1))
    return outputs


def choose_category(outputs, name, frame, axis, category):
    outputs[name][frame, axis].fill_(-12)
    outputs[name][frame, axis, category] = 12


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_config_is_validated_frozen_and_serializable(self):
        config = ModelConfig()
        self.assertEqual(asdict(config), {
            "architecture_version": 1, "n_mels": 96, "conditioning_dim": 12, "hidden_size": 128,
            "recurrent_layers": 2, "max_fret": 36, "max_voices": 4, "dropout": .1,
        })
        self.assertEqual(ModelConfig(**json.loads(json.dumps(asdict(config)))), config)
        with self.assertRaises(FrozenInstanceError):
            config.max_fret = 10
        self.assertEqual(ModelConfig(conditioning_dim=16).conditioning_dim, 16)
        for options in ({"architecture_version": 0}, {"architecture_version": 5}, {"n_mels": 0}, {"conditioning_dim": 11}, {"conditioning_dim": 13}, {"hidden_size": 0},
                        {"recurrent_layers": -1}, {"max_fret": -1}, {"max_fret": 128},
                        {"max_voices": 0}, {"dropout": 1}, {"dropout": -1}, {"dropout": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ModelConfig(**options)
        for options in ({"architecture_version": True}, {"n_mels": True}, {"max_fret": 2.5}, {"dropout": "0.1"}):
            with self.subTest(options=options), self.assertRaises(TypeError):
                ModelConfig(**options)
        with self.assertRaises(TypeError):
            FingerstyleTranscriber({})

    def test_variable_shapes_and_finite_backward(self):
        for batch, frames, mels in ((1, 1, 1), (3, 9, 11), (2, 4, 96)):
            with self.subTest(batch=batch, frames=frames, mels=mels):
                config = ModelConfig(n_mels=mels, hidden_size=8, recurrent_layers=2,
                                     max_fret=8, max_voices=3, dropout=0)
                model = FingerstyleTranscriber(config)
                features = torch.randn(batch, frames, mels, requires_grad=True)
                conditioning = torch.randn(batch, frames, 12, requires_grad=True)
                lengths = torch.arange(batch) % frames + 1
                outputs = model(features, conditioning, lengths)
                expected = {
                    "note_onset_logits": (6,), "fret_logits": (6, 9),
                    "pitch_logits": (6, 128), "voice_logits": (6, 3),
                    "duration_log": (6,), "harmonic_logits": (6,),
                    "harmonic_kind_logits": (6, 4), "harmonic_node_logits": (6, 6),
                    "percussion_logits": (3,),
                }
                self.assertEqual(set(outputs), set(expected))
                for name, tail in expected.items():
                    self.assertEqual(outputs[name].shape, (batch, frames, *tail))
                    self.assertTrue(torch.isfinite(outputs[name]).all())
                self.assertTrue((outputs["duration_log"] >= 0).all())
                sum(value.square().mean() for value in outputs.values()).backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertTrue(torch.isfinite(conditioning.grad).all())
                for row, length in enumerate(lengths):
                    self.assertEqual(torch.count_nonzero(features.grad[row, length:]).item(), 0)
                    self.assertEqual(torch.count_nonzero(conditioning.grad[row, length:]).item(), 0)

    def test_pilot_architecture_keeps_parameter_count_and_strict_state_loading(self):
        config = ModelConfig()
        model = FingerstyleTranscriber(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 891623)
        self.assertEqual(model.head_shapes["percussion_logits"], (3,))
        clone = FingerstyleTranscriber(config)
        clone.load_state_dict(model.state_dict(), strict=True)
        features, conditioning = torch.zeros(1, 3, 96), torch.zeros(1, 3, 12)
        model.eval()
        clone.eval()
        with torch.no_grad():
            original, restored = model(features, conditioning), clone(features, conditioning)
        for name in original:
            self.assertTrue(torch.equal(original[name], restored[name]), name)

    def test_encode_decode_preserves_original_checkpoint_keys_and_forward_values(self):
        from torch.nn import functional as F
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        for version in (1, 2, 3, 4):
            with self.subTest(version=version):
                model = FingerstyleTranscriber(ModelConfig(
                    architecture_version=version, n_mels=9, hidden_size=8,
                    recurrent_layers=2, dropout=0,
                )).eval()
                expected_keys = {
                    f"{module}.{parameter}" for module in ("conv1", "conv2", "projection.0")
                    for parameter in ("weight", "bias")
                }
                expected_keys.update(
                    f"recurrent.{parameter}_l{layer}{direction}"
                    for parameter in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
                    for layer in range(2) for direction in ("", "_reverse")
                )
                expected_keys.update(
                    f"heads.{name}.{parameter}" for name in model.head_shapes for parameter in ("weight", "bias")
                )
                self.assertEqual(set(model.state_dict()), expected_keys)
                features, conditioning = torch.randn(2, 5, 9), torch.randn(2, 5, 12)
                lengths = torch.tensor([3, 5])
                valid = torch.arange(5)[None, :] < lengths[:, None]
                hidden = features.masked_fill(~valid[:, :, None], 0)[:, None, :, :]
                hidden = F.gelu(model.conv1(hidden)).masked_fill(~valid[:, None, :, None], 0)
                hidden = F.gelu(model.conv2(hidden)).masked_fill(~valid[:, None, :, None], 0)
                hidden = model.projection(hidden.permute(0, 2, 1, 3).flatten(2))
                hidden = torch.cat((hidden, conditioning.masked_fill(~valid[:, :, None], 0)), dim=-1)
                packed, _ = model.recurrent(pack_padded_sequence(
                    hidden, lengths, batch_first=True, enforce_sorted=False,
                ))
                hidden, _ = pad_packed_sequence(packed, batch_first=True, total_length=5)
                reference = {}
                for name, head in model.heads.items():
                    values = head(hidden).reshape(2, 5, *model.head_shapes[name])
                    if name == "duration_log":
                        values = F.softplus(values)
                    mask = valid.reshape(2, 5, *([1] * len(model.head_shapes[name])))
                    reference[name] = values.masked_fill(~mask, 0)
                encoded, mask = model.encode(features, conditioning, lengths)
                self.assertTrue(torch.equal(encoded, hidden))
                self.assertTrue(torch.equal(mask, valid))
                for outputs in (
                    model(features, conditioning, lengths), model.decode_hidden(encoded, mask),
                ):
                    for name in reference:
                        self.assertTrue(torch.equal(outputs[name], reference[name]), name)

    def test_decode_hidden_rejects_invalid_shared_feature_inputs(self):
        model = FingerstyleTranscriber(ModelConfig(n_mels=8, hidden_size=4, recurrent_layers=1))
        hidden, valid = torch.zeros(2, 3, 8), torch.ones(2, 3, dtype=torch.bool)
        for value, mask, error in (
            ([], valid, TypeError), (hidden[:, 0], valid, ValueError),
            (hidden[:, :0], valid[:, :0], ValueError), (hidden[:, :, :7], valid, ValueError),
            (hidden.double(), valid, TypeError), (hidden, valid.float(), TypeError),
            (hidden, valid[:, :2], ValueError), (hidden + float("nan"), valid, ValueError),
        ):
            with self.subTest(error=error), self.assertRaises(error):
                model.decode_hidden(value, mask)

    def test_v2_adds_technique_heads_without_changing_v1_state_contract(self):
        config = ModelConfig(architecture_version=2, hidden_size=8, recurrent_layers=1, dropout=0)
        model = FingerstyleTranscriber(config)
        self.assertEqual(model.head_shapes["technique_logits"], (4,))
        self.assertEqual(model.head_shapes["technique_direction_logits"], (4, 2))
        self.assertEqual(model.head_shapes["technique_strings_logits"], (4, 6))
        lengths = torch.tensor([5, 3])
        outputs = model(torch.zeros(2, 5, 96), torch.zeros(2, 5, 12), lengths)
        targets, masks, valid = synthetic_targets(2, 5, architecture_version=2)
        valid[1, 3:] = False
        targets["technique"].zero_()
        masks["technique"].fill_(True)
        targets["technique"][0, 1, 0] = 1
        targets["technique_direction"][0, 1, 0] = 1
        masks["technique_direction"][0, 1, 0] = True
        targets["technique_strings"][0, 1, 0] = torch.tensor([1., 0., 1., 0., 0., 1.])
        masks["technique_strings"][0, 1, 0] = True
        loss, stats = masked_loss(outputs, targets, masks, valid)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["technique_positive"]["count"], 1)
        self.assertEqual(stats["technique_negative"]["count"], 31)
        self.assertEqual(stats["technique_direction"]["count"], 1)
        self.assertEqual(stats["technique_strings_positive"]["count"], 3)
        self.assertEqual(stats["technique_strings_negative"]["count"], 3)

    def test_v3_adds_connection_and_note_technique_heads(self):
        config = ModelConfig(architecture_version=3, hidden_size=8, recurrent_layers=1, dropout=0)
        model = FingerstyleTranscriber(config)
        self.assertEqual(model.head_shapes["connection_logits"], (6, 10))
        self.assertEqual(model.head_shapes["note_technique_logits"], (6, 4))
        self.assertEqual(model.head_shapes["bend_curve"], (6, 7))
        outputs = model(torch.zeros(1, 3, 96), torch.zeros(1, 3, 12))
        targets, masks, valid = synthetic_targets(1, 3, architecture_version=2)
        targets["connection"] = torch.zeros(1, 3, 6, dtype=torch.long)
        masks["connection"] = torch.zeros_like(targets["connection"], dtype=torch.bool)
        targets["note_technique"] = torch.zeros(1, 3, 6, 4)
        masks["note_technique"] = torch.zeros_like(targets["note_technique"], dtype=torch.bool)
        targets["bend_curve"] = torch.zeros(1, 3, 6, 7)
        masks["bend_curve"] = torch.zeros_like(targets["bend_curve"], dtype=torch.bool)
        targets["connection"][0, 1, 0] = 1
        masks["connection"][0, 1, 0] = True
        targets["note_technique"][0, 1, 0, :2] = 1
        masks["note_technique"][0, 1, 0] = True
        targets["bend_curve"][0, 1, 0] = torch.tensor([0, 0, .12, .12, .12, .99, .25])
        masks["bend_curve"][0, 1, 0] = True
        loss, stats = masked_loss(outputs, targets, masks, valid)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["connection_positive"]["count"], 1)
        self.assertEqual(stats["connection_negative"]["count"], 0)
        self.assertEqual(stats["note_technique_positive"]["count"], 2)
        self.assertEqual(stats["note_technique_negative"]["count"], 2)
        self.assertEqual(stats["bend_curve"]["count"], 7)

    def test_v3_loss_balances_legato_positives_against_many_negatives(self):
        outputs = synthetic_outputs(1, 20, requires_grad=True, architecture_version=3)
        targets, masks, valid = synthetic_targets(1, 20, architecture_version=2)
        targets["connection"] = torch.zeros(1, 20, 6, dtype=torch.long)
        masks["connection"] = torch.ones_like(targets["connection"], dtype=torch.bool)
        targets["connection"][0, 0, 0] = 1
        targets["note_technique"] = torch.zeros(1, 20, 6, 4)
        masks["note_technique"] = torch.ones_like(targets["note_technique"], dtype=torch.bool)
        targets["note_technique"][0, 0, 0, 0] = 1
        targets["bend_curve"] = torch.zeros(1, 20, 6, 7)
        masks["bend_curve"] = torch.zeros_like(targets["bend_curve"], dtype=torch.bool)
        loss, stats = masked_loss(outputs, targets, masks, valid)
        self.assertAlmostEqual(loss.item(), math.log(10) + math.log(2), places=6)
        self.assertEqual(stats["connection_positive"]["count"], 1)
        self.assertEqual(stats["connection_negative"]["count"], 119)
        self.assertEqual(stats["note_technique_positive"]["count"], 1)
        self.assertEqual(stats["note_technique_negative"]["count"], 479)
        loss.backward()
        self.assertLess(outputs["connection_logits"].grad[0, 0, 0, 1], 0)
        self.assertLess(outputs["note_technique_logits"].grad[0, 0, 0, 0], 0)

    def test_padding_does_not_change_real_frames(self):
        model = FingerstyleTranscriber(ModelConfig(n_mels=9, hidden_size=8, recurrent_layers=1, dropout=0)).eval()
        features = torch.randn(2, 8, 9)
        conditioning = torch.randn(2, 8, 12)
        padded = model(features, conditioning, torch.tensor([3, 8]))
        short = model(features[:1, :3], conditioning[:1, :3])
        changed_features, changed_conditioning = features.clone(), conditioning.clone()
        changed_features[0, 3:] = 1000
        changed_conditioning[0, 3:] = -1000
        changed = model(changed_features, changed_conditioning, torch.tensor([3, 8]))
        for name in padded:
            torch.testing.assert_close(padded[name][0, :3], short[name][0], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(padded[name], changed[name])
            self.assertEqual(torch.count_nonzero(padded[name][0, 3:]).item(), 0)

    def test_conditioning_and_frequency_order_affect_outputs(self):
        torch.manual_seed(7)
        model = FingerstyleTranscriber(ModelConfig(n_mels=12, hidden_size=8, recurrent_layers=1, dropout=0)).eval()
        features = torch.zeros(1, 5, 12)
        features[:, :, 1] = 4
        conditioning = torch.zeros(1, 5, 12)
        baseline = model(features, conditioning)["pitch_logits"]
        conditioned = model(features, conditioning + 1)["pitch_logits"]
        permuted = model(features.flip(-1), conditioning)["pitch_logits"]
        self.assertGreater((baseline - conditioned).abs().max().item(), 1e-5)
        self.assertGreater((baseline - permuted).abs().max().item(), 1e-5)

    def test_invalid_forward_inputs_fail_explicitly(self):
        model = FingerstyleTranscriber(ModelConfig(n_mels=8, hidden_size=4, recurrent_layers=1))
        features, conditioning = torch.zeros(2, 3, 8), torch.zeros(2, 3, 12)
        for feature, condition, length, error in (
            ([], conditioning, None, TypeError),
            (features[:, 0], conditioning, None, ValueError),
            (features[:, :0], conditioning[:, :0], None, ValueError),
            (features[:, :, :7], conditioning, None, ValueError),
            (features, conditioning[:, :, :11], None, ValueError),
            (features.long(), conditioning, None, TypeError),
            (features, conditioning.double(), None, TypeError),
            (features, conditioning, torch.tensor([1., 2.]), TypeError),
            (features, conditioning, torch.tensor([1]), ValueError),
            (features, conditioning, torch.tensor([0, 2]), ValueError),
            (features, conditioning, torch.tensor([4, 2]), ValueError),
            (features + float("nan"), conditioning, None, ValueError),
            (features, conditioning + float("inf"), None, ValueError),
        ):
            with self.subTest(error=error, length=length), self.assertRaises(error):
                model(feature, condition, length)


class MaskedLossTests(unittest.TestCase):
    def test_all_masked_is_differentiable_zero_with_no_priors(self):
        outputs = synthetic_outputs(requires_grad=True)
        targets, masks, valid = synthetic_targets()
        loss, stats = masked_loss(outputs, targets, masks, valid)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(set(stats), set(LOSS_STAT_KEYS))
        self.assertTrue(all(value == {"sum": 0.0, "count": 0} for value in stats.values()))
        loss.backward()
        for value in outputs.values():
            self.assertIsNotNone(value.grad)
            self.assertEqual(torch.count_nonzero(value.grad).item(), 0)

    def test_unknown_and_padded_targets_have_zero_gradient(self):
        outputs = synthetic_outputs(requires_grad=True)
        targets, masks, valid = synthetic_targets()
        valid[1, 2:] = False
        for name in targets:
            targets[name][0, 1, 0] = 1
            masks[name][0, 1, 0] = True
            masks[name][1, 2:] = True
        loss, stats = masked_loss(outputs, targets, masks, valid, sparsity_weight=0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, value in outputs.items():
            expected = torch.zeros_like(value, dtype=torch.bool)
            expected[0, 1, 0] = True
            self.assertEqual(torch.count_nonzero(value.grad[~expected]).item(), 0, name)
            self.assertGreater(torch.count_nonzero(value.grad[expected]).item(), 0, name)
        self.assertEqual(stats["note_onset_positive"]["count"], 1)
        self.assertEqual(stats["note_onset_negative"]["count"], 0)

    def test_note_onset_balances_known_sides_not_label_counts(self):
        outputs = synthetic_outputs(1, 3)
        targets, masks, valid = synthetic_targets(1, 3)
        targets["note_onset"].fill_(0)
        masks["note_onset"].fill_(True)
        targets["note_onset"][0, 0, 0] = 1
        outputs["note_onset_logits"].fill_(1)
        outputs["note_onset_logits"][0, 0, 0] = 2
        loss, stats = masked_loss(outputs, targets, masks, valid)
        positive = torch.nn.functional.softplus(torch.tensor(-2.)).item()
        negative = torch.nn.functional.softplus(torch.tensor(1.)).item()
        self.assertAlmostEqual(loss.item(), (positive + negative) / 2, places=6)
        self.assertAlmostEqual(stats["note_onset_positive"]["sum"], positive, places=6)
        self.assertEqual(stats["note_onset_positive"]["count"], 1)
        self.assertEqual(stats["note_onset_negative"]["count"], 17)
        masks["note_onset"] = targets["note_onset"].bool()
        positive_only, stats = masked_loss(outputs, targets, masks, valid)
        self.assertAlmostEqual(positive_only.item(), positive, places=6)
        self.assertEqual(stats["note_onset_negative"]["count"], 0)
        masks["note_onset"] = ~targets["note_onset"].bool()
        negative_only, stats = masked_loss(outputs, targets, masks, valid)
        self.assertAlmostEqual(negative_only.item(), negative, places=6)
        self.assertEqual(stats["note_onset_positive"]["count"], 0)

    def test_harmonic_still_rejects_masked_negatives(self):
        outputs = synthetic_outputs()
        targets, masks, valid = synthetic_targets()
        targets["harmonic"][0, 0, 0] = 0
        masks["harmonic"][0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "positive-only"):
            masked_loss(outputs, targets, masks, valid)

    def test_percussion_observed_negatives_train_but_unknowns_and_padding_do_not(self):
        outputs = synthetic_outputs(1, 4, requires_grad=True)
        targets, masks, valid = synthetic_targets(1, 4)
        targets["percussion"][0, :2, 0] = torch.tensor([1., 0.])
        masks["percussion"][0, :2, 0] = True
        targets["percussion"][0, 2, 0] = 0
        masks["percussion"][0, 3] = True
        valid[0, 3] = False
        loss, stats = masked_loss(outputs, targets, masks, valid)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        for name in ("percussion_positive", "percussion_negative"):
            self.assertEqual(stats[name]["count"], 1)
            self.assertAlmostEqual(stats[name]["sum"], math.log(2), places=6)
        self.assertEqual(stats["percussion_sparsity"]["count"], 0)
        loss.backward()
        gradient = outputs["percussion_logits"].grad
        self.assertAlmostEqual(gradient[0, 0, 0].item(), -5 / 12, places=6)
        self.assertAlmostEqual(gradient[0, 1, 0].item(), 1 / 12, places=6)
        self.assertEqual(torch.count_nonzero(gradient).item(), 2)

    def test_constant_all_positive_percussion_has_strong_negative_loss_and_gradient(self):
        outputs = synthetic_outputs(1, 20)
        targets, masks, valid = synthetic_targets(1, 20)
        logit = torch.tensor(8., requires_grad=True)
        outputs["percussion_logits"] = logit.expand(1, 20, 3)
        targets["percussion"].zero_()
        masks["percussion"].fill_(True)
        targets["percussion"][0, 0, 0] = 1
        loss, stats = masked_loss(outputs, targets, masks, valid)
        self.assertEqual(stats["percussion_positive"]["count"], 1)
        self.assertEqual(stats["percussion_negative"]["count"], 59)
        self.assertEqual(stats["percussion_sparsity"]["count"], 0)
        self.assertGreater(loss.item(), 7)
        expected = (5 * stats["percussion_positive"]["sum"] + stats["percussion_negative"]["sum"]) / (5 + 59)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertGreater(logit.grad.item(), .9)

    def test_negative_only_percussion_batches_update_without_a_sparsity_prior(self):
        outputs = synthetic_outputs(1, 3, requires_grad=True)
        targets, masks, valid = synthetic_targets(1, 3)
        targets["percussion"][0, 1, 2] = 0
        masks["percussion"][0, 1, 2] = True
        optimizer = torch.optim.SGD([outputs["percussion_logits"]], lr=.1)
        loss, stats = masked_loss(outputs, targets, masks, valid)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        self.assertEqual(stats["percussion_positive"]["count"], 0)
        self.assertEqual(stats["percussion_negative"]["count"], 1)
        self.assertEqual(stats["percussion_sparsity"]["count"], 0)
        loss.backward()
        self.assertEqual(outputs["percussion_logits"].grad[0, 1, 2].item(), .5)
        optimizer.step()
        self.assertLess(outputs["percussion_logits"][0, 1, 2].item(), 0)
        self.assertEqual(torch.count_nonzero(outputs["percussion_logits"]).item(), 1)

    def test_positive_only_percussion_retains_exact_original_loss_and_gradient(self):
        outputs = synthetic_outputs(2, 4, requires_grad=True)
        outputs["percussion_logits"] = torch.linspace(-2, 3, 24).reshape(2, 4, 3).requires_grad_()
        targets, masks, valid = synthetic_targets(2, 4)
        valid[1, 2:] = False
        for row, frame, axis in ((0, 1, 0), (0, 2, 2), (1, 0, 2)):
            targets["percussion"][row, frame, axis] = 1
            masks["percussion"][row, frame, axis] = True
        prediction = outputs["percussion_logits"]
        values = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[masks["percussion"]], targets["percussion"][masks["percussion"]], reduction="none",
        )
        prior_mask = valid[..., None] & torch.tensor([True, False, True])
        prior = prediction[prior_mask].sigmoid()
        original_loss = values.sum() / values.numel() + .02 * (prior.sum() / prior.numel())
        loss, stats = masked_loss(outputs, targets, masks, valid)
        self.assertTrue(torch.equal(loss, original_loss))
        self.assertEqual(stats["percussion_negative"], {"sum": 0., "count": 0})
        original_gradient, = torch.autograd.grad(original_loss, prediction)
        actual_gradient, = torch.autograd.grad(loss, prediction)
        self.assertTrue(torch.equal(actual_gradient, original_gradient))

    def test_percussion_prior_excludes_only_classes_with_observed_negatives(self):
        outputs = synthetic_outputs(1, 4, requires_grad=True)
        targets, masks, valid = synthetic_targets(1, 4)
        valid[0, 3] = False
        for frame, axis, label in ((0, 0, 1), (1, 0, 0), (1, 1, 1), (0, 2, 0), (3, 1, 0)):
            targets["percussion"][0, frame, axis] = label
            masks["percussion"][0, frame, axis] = True
        targets["percussion"][0, 2, 1] = 0
        with_prior, stats = masked_loss(outputs, targets, masks, valid, sparsity_weight=.2)
        without_prior, _ = masked_loss(outputs, targets, masks, valid, sparsity_weight=0)
        self.assertAlmostEqual((with_prior - without_prior).item(), .1, places=6)
        self.assertEqual(stats["percussion_sparsity"], {"sum": 1.5, "count": 3})
        with_prior.backward()
        gradient = outputs["percussion_logits"].grad
        self.assertGreater(gradient[0, 2, 1].item(), 0)
        self.assertEqual(gradient[0, 2, 0].item(), 0)
        self.assertEqual(gradient[0, 2, 2].item(), 0)
        self.assertEqual(torch.count_nonzero(gradient[0, 3]).item(), 0)

    def test_priors_are_separate_and_only_use_evidenced_classes_and_known_attacks(self):
        outputs = synthetic_outputs(1, 4, requires_grad=True)
        targets, masks, valid = synthetic_targets(1, 4)
        valid[0, 3] = False
        for frame in (0, 2):
            targets["note_onset"][0, frame, 0] = 1
            masks["note_onset"][0, frame, 0] = True
        targets["harmonic"][0, 0, 0] = 1
        masks["harmonic"][0, 0, 0] = True
        targets["percussion"][0, 1, 1] = 1
        masks["percussion"][0, 1, 1] = True
        original_masks = {name: value.clone() for name, value in masks.items()}
        with_prior, stats = masked_loss(outputs, targets, masks, valid, sparsity_weight=.2)
        without_prior, _ = masked_loss(outputs, targets, masks, valid, sparsity_weight=0)
        self.assertAlmostEqual((with_prior - without_prior).item(), .2, places=6)
        self.assertEqual(stats["harmonic_sparsity"], {"sum": 1.0, "count": 2})
        self.assertEqual(stats["percussion_sparsity"], {"sum": 1.5, "count": 3})
        self.assertEqual(stats["harmonic_positive"]["count"], 1)
        self.assertEqual(stats["percussion_positive"]["count"], 1)
        self.assertEqual(stats["note_onset_negative"]["count"], 0)
        with_prior.backward()
        self.assertGreater(outputs["harmonic_logits"].grad[0, 2, 0].item(), 0)
        self.assertEqual(outputs["harmonic_logits"].grad[0, 1, 0].item(), 0)
        self.assertGreater(outputs["percussion_logits"].grad[0, 0, 1].item(), 0)
        self.assertEqual(torch.count_nonzero(outputs["percussion_logits"].grad[:, :, [0, 2]]).item(), 0)
        self.assertEqual(torch.count_nonzero(outputs["percussion_logits"].grad[:, 3:]).item(), 0)
        for name in masks:
            torch.testing.assert_close(masks[name], original_masks[name])

    def test_no_positive_evidence_disables_presence_priors(self):
        outputs = synthetic_outputs(requires_grad=True)
        targets, masks, valid = synthetic_targets()
        targets["note_onset"][0, 0, 0] = 1
        masks["note_onset"][0, 0, 0] = True
        loss, stats = masked_loss(outputs, targets, masks, valid)
        loss.backward()
        for name in ("harmonic", "percussion"):
            self.assertEqual(stats[f"{name}_positive"]["count"], 0)
            self.assertEqual(stats[f"{name}_sparsity"]["count"], 0)
            self.assertEqual(torch.count_nonzero(outputs[f"{name}_logits"].grad).item(), 0)
        targets["harmonic"][0, 0, 0] = 1
        masks["harmonic"][0, 0, 0] = True
        all_padded, stats = masked_loss(outputs, targets, masks, torch.zeros_like(valid))
        self.assertEqual(all_padded.item(), 0)
        self.assertTrue(all(value["count"] == 0 for value in stats.values()))

    def test_category_collision_masks_remain_independent(self):
        outputs = synthetic_outputs(1, 1, requires_grad=True)
        targets, masks, valid = synthetic_targets(1, 1)
        for name in ("note_onset", "harmonic"):
            targets[name][0, 0, 0] = 1
            masks[name][0, 0, 0] = True
        targets["pitch"][0, 0, 0] = 60
        masks["pitch"][0, 0, 0] = True
        # A collision can leave pitch known while fret/voice/kind/node remain
        # unknown. Their sentinel labels must never reach cross entropy.
        loss, stats = masked_loss(outputs, targets, masks, valid, sparsity_weight=0)
        loss.backward()
        self.assertEqual(stats["pitch"]["count"], 1)
        for name in ("fret", "voice", "harmonic_kind", "harmonic_node"):
            self.assertEqual(stats[name]["count"], 0)
            self.assertEqual(torch.count_nonzero(outputs[f"{name}_logits"].grad).item(), 0)

    def test_sums_and_counts_reconstruct_component_objective(self):
        outputs = synthetic_outputs(1, 2)
        targets, masks, valid = synthetic_targets(1, 2)
        for name in targets:
            targets[name][0, 0, 0] = 1
            masks[name][0, 0, 0] = True
        targets["note_onset"][0, 1, 0] = 0
        masks["note_onset"][0, 1, 0] = True
        targets["percussion"][0, 1, 0] = 0
        masks["percussion"][0, 1, 0] = True
        loss, stats = masked_loss(outputs, targets, masks, valid)
        means = {name: stat["sum"] / stat["count"] if stat["count"] else 0 for name, stat in stats.items()}
        reconstructed = LOSS_WEIGHTS["note_onset"] * (means["note_onset_positive"] + means["note_onset_negative"]) / 2
        percussion = ("percussion_positive", "percussion_negative")
        reconstructed += sum(
            weight * means[name] for name, weight in LOSS_WEIGHTS.items() if name not in ("note_onset", *percussion)
        )
        reconstructed += sum(LOSS_WEIGHTS[name] * stats[name]["sum"] for name in percussion) / sum(
            LOSS_WEIGHTS[name] * stats[name]["count"] for name in percussion
        )
        reconstructed += .02 * (means["harmonic_sparsity"] + means["percussion_sparsity"])
        self.assertAlmostEqual(loss.item(), reconstructed, places=5)
        for stat in stats.values():
            self.assertIsInstance(stat["sum"], float)
            self.assertIsInstance(stat["count"], int)

    def test_invalid_loss_inputs_are_explicit(self):
        outputs = synthetic_outputs()
        targets, masks, valid = synthetic_targets()
        for weight in (-1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                masked_loss(outputs, targets, masks, valid, sparsity_weight=weight)
        with self.assertRaises(TypeError):
            masked_loss(outputs, targets, masks, valid.float())
        with self.assertRaises(ValueError):
            masked_loss(outputs, targets, masks, valid[:, :1])
        with self.assertRaises(ValueError):
            masked_loss({}, targets, masks, valid)
        with self.assertRaises(ValueError):
            masked_loss(outputs, {}, masks, valid)
        with self.assertRaises(ValueError):
            masked_loss(outputs, targets, {}, valid)
        for name, invalid, error in (("fret", -1, ValueError), ("pitch", 128, ValueError),
                                     ("voice", 4, ValueError), ("harmonic_kind", 4, ValueError),
                                     ("harmonic_node", 6, ValueError), ("duration_log", -1, ValueError),
                                     ("note_onset", .5, ValueError), ("harmonic", float("inf"), ValueError),
                                     ("percussion", .5, ValueError), ("percussion", -1, ValueError),
                                     ("percussion", float("nan"), ValueError)):
            local_targets, local_masks, _ = synthetic_targets()
            local_targets[name][0, 0, 0] = invalid
            local_masks[name][0, 0, 0] = True
            with self.subTest(name=name), self.assertRaises(error):
                masked_loss(outputs, local_targets, local_masks, valid)
        wrong_masks = dict(masks, fret=masks["fret"].float())
        wrong_targets = dict(targets, pitch=targets["pitch"].float())
        with self.assertRaises(TypeError):
            masked_loss(outputs, targets, wrong_masks, valid)
        with self.assertRaises(TypeError):
            masked_loss(outputs, wrong_targets, masks, valid)


class EventDecoderTests(unittest.TestCase):
    tuning = (40, 45, 50, 55, 59, 64)

    def decode(self, outputs, seconds, **kwargs):
        return decode_events(outputs, seconds, tuning=self.tuning, capo=0, **kwargs)

    def test_voiced_notes_and_percussion_coexist_without_carrier_fingering(self):
        outputs = event_outputs(4)
        outputs["note_onset_logits"][1, [0, 5]] = 8
        outputs["percussion_logits"][1, :2] = 8
        choose_category(outputs, "fret_logits", 1, 0, 3)
        choose_category(outputs, "pitch_logits", 1, 0, 43)
        choose_category(outputs, "voice_logits", 1, 0, 1)
        choose_category(outputs, "pitch_logits", 1, 5, 64)
        decoded = self.decode(outputs, [2, 2.02, 2.04, 2.06])
        self.assertEqual(len(decoded["notes"]), 2)
        self.assertEqual([note["string"] for note in decoded["notes"]], [6, 1])
        self.assertEqual([note["voiceIndex"] for note in decoded["notes"]], [1, 0])
        self.assertAlmostEqual(decoded["notes"][0]["notatedDurationQuarter"], 1, places=6)
        self.assertIsNone(decoded["notes"][0]["harmonic"])
        self.assertEqual(decoded["notes"][0]["uncertainty"], [])
        self.assertEqual([item["technique"] for item in decoded["percussion"]], list(PERCUSSION_TYPES[:2]))
        for item in decoded["percussion"]:
            self.assertEqual(item["onsetSeconds"], 2.02)
            self.assertEqual(item["presenceCalibration"], PRESENCE_CALIBRATION)
            self.assertNotIn("string", item)
            self.assertNotIn("fret", item)
        json.dumps(decoded, allow_nan=False)

    def test_v2_decodes_strum_direction_and_complete_string_membership(self):
        outputs = event_outputs(3, architecture_version=2)
        outputs["technique_logits"][1, 0] = 8
        outputs["technique_direction_logits"][1, 0, 1] = 8
        outputs["technique_strings_logits"][1, 0, [0, 2, 5]] = 8
        for axis, pitch in ((0, 40), (2, 50), (5, 64)):
            choose_category(outputs, "pitch_logits", 1, axis, pitch)
        decoded = self.decode(outputs, [0, .02, .04])
        self.assertEqual(decoded["techniques"][0]["technique"], "brush")
        self.assertEqual(decoded["techniques"][0]["direction"], "Up")
        self.assertEqual(decoded["techniques"][0]["strings"], [6, 4, 1])
        self.assertEqual(set(decoded["techniques"][0]["stringMembershipConfidence"]), {"1", "2", "3", "4", "5", "6"})
        self.assertEqual(
            [(note["string"], note["soundingPitchMidi"]) for note in decoded["notes"]],
            [(6, 40), (4, 50), (1, 64)],
        )
        self.assertTrue(all("technique_membership_completed_attack" in note["uncertainty"] for note in decoded["notes"]))
        self.assertTrue(all(note["completionParent"] == {
            "technique": decoded["techniques"][0]["technique"], "onsetSeconds": .02,
        } for note in decoded["notes"]))
        outputs["note_onset_logits"][1, 0] = 8
        independent = self.decode(outputs, [0, .02, .04])
        self.assertNotIn("completionParent", next(note for note in independent["notes"] if note["string"] == 6))

    def test_v3_decodes_connection_note_techniques_and_bend_curve(self):
        outputs = event_outputs(3, architecture_version=3)
        outputs["note_onset_logits"][1, 0] = 8
        choose_category(outputs, "pitch_logits", 1, 0, 40)
        choose_category(outputs, "connection_logits", 1, 0, 1)
        outputs["note_technique_logits"][1, 0, [0, 2, 3]] = 8
        outputs["bend_curve"][1, 0] = torch.tensor([0, 0, .12, .12, .12, .99, .25])
        note = self.decode(outputs, [0, .02, .04])["notes"][0]
        self.assertEqual(note["connection"], "hammer_on")
        self.assertGreater(note["connectionConfidence"], .99)
        self.assertGreater(note["noteTechniques"]["bend"], .99)
        self.assertGreater(note["noteTechniques"]["left_hand_tap"], .99)
        self.assertAlmostEqual(note["bendCurve"]["DestinationOffset"], 99)
        self.assertAlmostEqual(note["bendCurve"]["DestinationValue"], 25)

    def test_plateaus_repeated_attacks_and_late_audio_are_not_truncated(self):
        outputs = event_outputs(14)
        scores = torch.tensor([8, 8, -8, 9, -8, 9, -8, -8, 8, 8, 8, -8, -8, 10.])
        outputs["note_onset_logits"][:, 0] = scores
        outputs["percussion_logits"][:, 2] = scores
        seconds = [0., .01, .02, .03, .04, .05, .06, .07, 10., 10.01, 10.02, 10.03, 100., 101.]
        decoded = self.decode(outputs, seconds, min_gap_seconds=.04)
        expected = [.03, 10., 101.]
        self.assertEqual([note["onsetSeconds"] for note in decoded["notes"]], expected)
        self.assertEqual([item["onsetSeconds"] for item in decoded["percussion"]], expected)
        self.assertEqual(decoded, self.decode(outputs, seconds, min_gap_seconds=.04))

    def test_entire_plateau_is_one_attack_and_nms_is_per_string(self):
        outputs = event_outputs(6)
        outputs["note_onset_logits"][:, 0] = 5
        outputs["note_onset_logits"][[1, 4], 1] = 5
        decoded = self.decode(outputs, [4, 4.01, 4.02, 4.03, 4.2, 4.21])
        self.assertEqual([(note["onsetSeconds"], note["string"]) for note in decoded["notes"]],
                         [(4., 6), (4.01, 5), (4.2, 5)])

    def test_harmonic_sounding_pitch_uses_kind_node_tuning_and_capo(self):
        self.assertEqual(HARMONIC_FRETS, (5, 7, 9, 12, 19, 24))
        offsets = (24, 19, 28, 12, 19, 24)
        for kind_index, kind in enumerate(HARMONIC_TYPES):
            for node_index, offset in enumerate(offsets):
                with self.subTest(kind=kind, node=HARMONIC_FRETS[node_index]):
                    outputs = event_outputs(1)
                    outputs["note_onset_logits"][0, 0] = 8
                    outputs["harmonic_logits"][0, 0] = 8
                    choose_category(outputs, "fret_logits", 0, 0, 7)
                    choose_category(outputs, "harmonic_kind_logits", 0, 0, kind_index)
                    choose_category(outputs, "harmonic_node_logits", 0, 0, node_index)
                    expected = 40 + 2 + offset + (0 if kind == "Natural" else 7)
                    choose_category(outputs, "pitch_logits", 0, 0, expected)
                    note = decode_events(outputs, [0.], tuning=self.tuning, capo=2)["notes"][0]
                    self.assertEqual(note["soundingPitchMidi"], expected)
                    self.assertEqual(note["fretBasePitchMidi"], 49)
                    self.assertEqual(note["expectedSoundingPitchMidi"], expected)
                    self.assertEqual(note["harmonic"]["type"], kind)
                    self.assertEqual(note["harmonic"]["fret"], HARMONIC_FRETS[node_index])
                    self.assertNotIn("sounding_pitch_fret_mismatch", note["uncertainty"])
                    self.assertNotIn("sounding_pitch_harmonic_mismatch", note["uncertainty"])
                    choose_category(outputs, "pitch_logits", 0, 0, expected - 1)
                    inconsistent = decode_events(outputs, [0.], tuning=self.tuning, capo=2)["notes"][0]
                    self.assertEqual(inconsistent["soundingPitchMidi"], expected - 1)
                    self.assertIn("sounding_pitch_harmonic_mismatch", inconsistent["uncertainty"])

    def test_pitch_mismatch_is_reported_not_repaired_and_duration_is_bounded(self):
        outputs = event_outputs(1)
        outputs["note_onset_logits"][0, 0] = 8
        outputs["duration_log"][0, 0] = 1000
        choose_category(outputs, "fret_logits", 0, 0, 3)
        choose_category(outputs, "pitch_logits", 0, 0, 99)
        note = self.decode(outputs, [20.])["notes"][0]
        self.assertEqual(note["fret"], 3)
        self.assertEqual(note["soundingPitchMidi"], 99)
        self.assertEqual(note["notatedDurationQuarter"], 64)
        self.assertEqual(note["uncertainty"], ["sounding_pitch_fret_mismatch", "duration_clipped"])
        json.dumps(self.decode(outputs, [20.]), allow_nan=False)

    def test_empty_timeline_and_below_threshold_are_not_true_negatives(self):
        decoded = self.decode(event_outputs(0), [])
        self.assertEqual(decoded["notes"], [])
        self.assertEqual(decoded["percussion"], [])
        self.assertIn("not a confirmed negative", decoded["policy"]["presenceAbsence"])
        self.assertEqual(PRESENCE_CALIBRATION, "unvalidated-model-scores")
        self.assertEqual(self.decode(event_outputs(3), torch.arange(3.))["notes"], [])

    def test_decoder_invalid_inputs_fail_explicitly(self):
        outputs = event_outputs(3)
        for options in ({"onset_threshold": -1}, {"onset_threshold": 1.01},
                        {"percussion_threshold": float("nan")}, {"harmonic_threshold": 2},
                        {"min_gap_seconds": -1}, {"max_duration_quarter": 0},
                        {"max_duration_quarter": float("inf")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.decode(outputs, [0, 1, 2], **options)
        with self.assertRaises(TypeError):
            self.decode(outputs, [0, 1, 2], onset_threshold=True)
        for seconds, error in (([0, 0, 1], ValueError), ([0, 2, 1], ValueError),
                               ([-1, 0, 1], ValueError), ([0, 1], ValueError),
                               ([0, float("nan"), 2], ValueError), ("abc", TypeError),
                               ([False, 1, 2], TypeError), (torch.ones(3, dtype=torch.bool), TypeError)):
            with self.subTest(seconds=seconds), self.assertRaises(error):
                self.decode(outputs, seconds)
        for tuning, capo, error in ((self.tuning[:5], 0, ValueError), (self.tuning, -1, ValueError),
                                    ([40.] * 6, 0, TypeError), ([127] * 6, 1, ValueError),
                                    (self.tuning, True, TypeError), ("abcdef", 0, TypeError)):
            with self.subTest(tuning=tuning, capo=capo), self.assertRaises(error):
                decode_events(outputs, [0, 1, 2], tuning=tuning, capo=capo)
        for name in outputs:
            malformed = dict(outputs)
            malformed[name] = outputs[name][:2]
            with self.subTest(head=name), self.assertRaises(ValueError):
                self.decode(malformed, [0, 1, 2])
        for replacement, error in ((torch.zeros(3, 6, dtype=torch.long), TypeError),
                                   (torch.full((3, 6), float("nan")), ValueError)):
            malformed = dict(outputs, note_onset_logits=replacement)
            with self.assertRaises(error):
                self.decode(malformed, [0, 1, 2])
        outputs["duration_log"][0, 0] = -1
        with self.assertRaises(ValueError):
            self.decode(outputs, [0, 1, 2])
        with self.assertRaises(ValueError):
            self.decode(synthetic_outputs(1, 3), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
