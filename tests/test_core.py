"""Unexecuted unit tests; no model weights or network access required."""

import unittest

import torch
from torch import nn

from neuron_steering import Target, calibrate, discover_layers, steer


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.tensor([[1., 0., 2.], [0., 2., 0.]]))

    def forward(self, activation):
        return self.down_proj(activation)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()

    def forward(self, hidden):
        # A deterministic block-shaped fixture, not a Qwen replica.
        return self.mlp(torch.cat((hidden, hidden[..., :1]), dim=-1)) + hidden


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block()])

    def forward(self, input_ids, attention_mask, **_kwargs):
        hidden = torch.stack((input_ids.float(), input_ids.float()), dim=-1)
        return self.layers[0](hidden)


class SteeringTests(unittest.TestCase):
    def setUp(self):
        self.model = Model().eval()
        self.layers = discover_layers(self.model)
        self.target = Target(0, 1, -1.)

    def test_projection_difference_and_exception_cleanup(self):
        projection = self.layers[0].mlp.down_proj
        activation = torch.ones(1, 2, 3)
        baseline = projection(activation)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with steer(self.layers, {self.target: -3.}):
                expected = (-3. * projection.weight[:, 1]).expand_as(baseline)
                torch.testing.assert_close(projection(activation) - baseline, expected)
                raise RuntimeError("intentional")
        torch.testing.assert_close(projection(activation), baseline)
        self.assertFalse(projection._forward_pre_hooks)

    def test_calibration_ignores_padding_and_first_valid_token(self):
        inputs = {"input_ids": torch.tensor([[100, 1, 2, 3]]),
                  "attention_mask": torch.tensor([[0, 1, 1, 1]])}
        # For token value v, block output is [4v, 3v]. Remaining norms: 10, 15.
        # torch.median selects 10; the target weight column has norm 2.
        doses = calibrate(self.model, self.layers, [self.target], inputs)
        self.assertAlmostEqual(doses[self.target], -5.)
        self.assertFalse(self.layers[0]._forward_hooks)

    def test_empty_calibration_tokens_rejected(self):
        inputs = {"input_ids": torch.tensor([[1]]),
                  "attention_mask": torch.tensor([[1]])}
        with self.assertRaisesRegex(ValueError, "No calibration tokens"):
            calibrate(self.model, self.layers, [self.target], inputs)

    def test_cached_single_token_shape_supported_by_intervention(self):
        projection = self.layers[0].mlp.down_proj
        activation = torch.ones(1, 1, 3)
        baseline = projection(activation)
        with steer(self.layers, {self.target: -2.}):
            expected = (-2. * projection.weight[:, 1]).expand_as(baseline)
            torch.testing.assert_close(projection(activation) - baseline, expected)


if __name__ == "__main__":
    unittest.main()
