"""Calibration uses block outputs; intervention modifies down_proj inputs."""

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class Target:
    """Both indices are zero-based; strength is a signed coefficient."""

    layer: int
    neuron: int
    strength: float


def discover_layers(model: nn.Module) -> dict[int, nn.Module]:
    """Find one dense decoder stack by structure, without guessing wrapper paths."""
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not len(module):
            continue
        if all(isinstance(getattr(getattr(block, "mlp", None), "down_proj", None),
                          nn.Linear) for block in module):
            candidates.append((name, module))
    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise ValueError(f"Expected one dense decoder stack; found {names}. "
                         "Quantized, fused, MoE and sharded projections are unsupported.")
    return dict(enumerate(candidates[0][1]))


def _validate(layers: Mapping[int, nn.Module], targets: Sequence[Target]) -> None:
    if not targets:
        raise ValueError("At least one target is required.")
    seen = set()
    for target in targets:
        if target.layer not in layers:
            raise ValueError(f"Unknown layer: {target.layer}")
        projection = layers[target.layer].mlp.down_proj
        if not isinstance(projection, nn.Linear):
            raise TypeError("Only unquantized torch.nn.Linear projections are supported.")
        if not 0 <= target.neuron < projection.in_features:
            raise ValueError(f"Neuron out of range: {target}")
        if not math.isfinite(target.strength):
            raise ValueError("Strength must be finite.")
        key = (target.layer, target.neuron)
        if key in seen:
            raise ValueError(f"Duplicate target: {key}")
        seen.add(key)


@torch.inference_mode()
def calibrate(model: nn.Module, layers: Mapping[int, nn.Module],
              targets: Sequence[Target], inputs: Mapping[str, torch.Tensor],
              *, skip_first_token: bool = True) -> dict[Target, float]:
    """Measure an unsteered, single-prompt forward pass and return fixed doses.

    Median means torch.median (lower middle value for an even sample count).
    The first *valid* token is optionally excluded, independently of padding.
    Do not call with steering hooks installed or with cached inputs.
    """
    _validate(layers, targets)
    if model.training:
        raise ValueError("Call model.eval() before calibration.")
    ids, mask = inputs.get("input_ids"), inputs.get("attention_mask")
    if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Calibration requires a single text prompt, shape [1, tokens].")
    if mask is None or mask.shape != ids.shape:
        raise ValueError("An attention_mask matching input_ids is required.")
    if "past_key_values" in inputs or "inputs_embeds" in inputs:
        raise ValueError("Cached inputs and inputs_embeds are unsupported.")
    valid = mask.bool().clone()
    positions = valid[0].nonzero(as_tuple=True)[0]
    if skip_first_token and positions.numel():
        valid[0, positions[0]] = False
    if not valid.any().item():
        raise ValueError("No calibration tokens remain; use a longer prompt.")
    scales = {}
    handles = []

    def make_hook(index):
        def capture(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.shape[:2] != ids.shape:
                raise ValueError("Decoder output must have shape [1, tokens, hidden].")
            norms = hidden.detach().float().norm(dim=-1)
            scale = norms[valid.to(norms.device)].median().item()
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid residual scale at layer {index}: {scale}")
            scales[index] = scale
        return capture

    try:
        for index in sorted({target.layer for target in targets}):
            handles.append(layers[index].register_forward_hook(make_hook(index)))
        kwargs = dict(inputs)
        kwargs.update(use_cache=False, output_hidden_states=False,
                      output_attentions=False)
        model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()

    doses = {}
    for target in targets:
        weight = layers[target.layer].mlp.down_proj.weight
        norm = weight[:, target.neuron].detach().float().norm().item()
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"Invalid output direction norm: {target}")
        if target.layer not in scales:
            raise ValueError(f"Layer {target.layer} was not called during calibration.")
        dose = target.strength * scales[target.layer] / norm
        if not math.isfinite(dose):
            raise ValueError(f"Nonfinite dose: {target}")
        doses[target] = dose
    return doses


@contextmanager
def steer(layers: Mapping[int, nn.Module], doses: Mapping[Target, float]):
    """Apply fixed additions to every processed token, including cached decoding.

    Prompt padding positions are also modified but excluded from calibration.
    Use only for inference, without concurrent calls on the same model.
    Hooks are removed even if generation raises an exception.
    """
    _validate(layers, list(doses))
    grouped = {}
    for target, dose in doses.items():
        if not math.isfinite(dose):
            raise ValueError("Doses must be finite.")
        grouped.setdefault(target.layer, []).append((target.neuron, dose))
    handles = []

    def make_hook(edits):
        def modify(_module, args):
            activation = args[0]
            changed = activation.clone()
            for neuron, dose in edits:
                changed[..., neuron] += dose
            if not torch.isfinite(changed).all().item():
                raise FloatingPointError("Steered activation overflowed; reduce strength.")
            return (changed, *args[1:])
        return modify

    try:
        for index, edits in grouped.items():
            handles.append(layers[index].mlp.down_proj.register_forward_pre_hook(
                make_hook(edits)))
        yield
    finally:
        for handle in handles:
            handle.remove()
