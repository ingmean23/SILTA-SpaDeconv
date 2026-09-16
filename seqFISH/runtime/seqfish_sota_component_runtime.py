"""Isolated structural-latent routing for the seqFISH component search."""

from __future__ import annotations

import json
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F


ROUTES = {"mu", "sampled_h", "sampled_g", "sampled_hg"}
_SETTINGS: Dict[str, Any] = {}


def normalize_settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    route = str(config.get("seqfish_cooperative_input_route", "mu")).lower()
    if route not in ROUTES:
        raise ValueError(f"Unknown seqFISH cooperative route: {route}")
    h_scale = float(config.get("seqfish_h_sampling_scale", 1.0))
    g_scale = float(config.get("seqfish_g_sampling_scale", 1.0))
    alpha = float(config.get("seqfish_joint_adapter_alpha", 0.02))
    interaction = float(
        config.get("seqfish_joint_adapter_interaction_weight", 0.10))
    if h_scale < 0.0 or g_scale < 0.0:
        raise ValueError("Structural sampling scales must be non-negative")
    if alpha < 0.0:
        raise ValueError("Joint adapter alpha must be non-negative")
    return {
        "route": route,
        "h_sampling_scale": h_scale,
        "g_sampling_scale": g_scale,
        "joint_adapter": bool(
            config.get("seqfish_joint_adapter_enabled", False)),
        "joint_adapter_alpha": alpha,
        "joint_adapter_interaction_weight": interaction,
    }


def _buffer(module: torch.nn.Module, name: str) -> None:
    if not hasattr(module, name):
        module.register_buffer(
            name, torch.tensor(0.0, dtype=torch.float64), persistent=True)


def _increment(module: torch.nn.Module, name: str) -> None:
    with torch.no_grad():
        getattr(module, name).add_(1.0)


def _assign(module: torch.nn.Module, name: str, value: float) -> None:
    with torch.no_grad():
        getattr(module, name).fill_(float(value))


def _scaled_sample(
    mu: torch.Tensor, sampled: torch.Tensor, scale: float,
) -> torch.Tensor:
    return mu + float(scale) * (sampled - mu)


def attach(model: torch.nn.Module, settings: Mapping[str, Any]) -> None:
    if getattr(model, "_seqfish_sota_component_attached", False):
        return
    route = str(settings["route"])
    joint = bool(settings["joint_adapter"])
    h_scale = float(settings["h_sampling_scale"])
    g_scale = float(settings["g_sampling_scale"])

    for name in (
        "seqfish_route_forward_count",
        "seqfish_route_train_forward_count",
        "seqfish_sampled_h_train_count",
        "seqfish_sampled_g_train_count",
        "seqfish_last_h_delta_mean_abs",
        "seqfish_last_g_delta_mean_abs",
        "seqfish_joint_adapter_forward_count",
        "seqfish_joint_adapter_train_forward_count",
        "seqfish_joint_adapter_last_delta_abs_mean",
    ):
        _buffer(model, name)

    original_h = model._reparameterize_h
    original_g = model._reparameterize_g

    def remember_h(h_mu: torch.Tensor, h_logvar: torch.Tensor) -> torch.Tensor:
        sampled = _scaled_sample(h_mu, original_h(h_mu, h_logvar), h_scale)
        model._seqfish_route_h = sampled
        delta = float((sampled.detach() - h_mu.detach()).abs().mean().item())
        _assign(model, "seqfish_last_h_delta_mean_abs", delta)
        if model.training and delta > 0.0:
            _increment(model, "seqfish_sampled_h_train_count")
        return sampled

    def remember_g(g_mu: torch.Tensor, g_logvar: torch.Tensor) -> torch.Tensor:
        sampled = _scaled_sample(g_mu, original_g(g_mu, g_logvar), g_scale)
        model._seqfish_route_g = sampled
        delta = float((sampled.detach() - g_mu.detach()).abs().mean().item())
        _assign(model, "seqfish_last_g_delta_mean_abs", delta)
        if model.training and delta > 0.0:
            _increment(model, "seqfish_sampled_g_train_count")
        return sampled

    model._reparameterize_h = remember_h
    model._reparameterize_g = remember_g

    def cooperative_inputs(module, args, kwargs):
        del module
        if len(args) < 4:
            raise RuntimeError("Unexpected cooperative fusion call signature")
        h_value, g_value = args[0], args[1]
        if route in {"sampled_h", "sampled_hg"}:
            h_value = getattr(model, "_seqfish_route_h", None)
        if route in {"sampled_g", "sampled_hg"}:
            g_value = getattr(model, "_seqfish_route_g", None)
        if h_value is None or g_value is None:
            raise RuntimeError(
                "Structural latents were not prepared before cooperative fusion")
        _increment(model, "seqfish_route_forward_count")
        if model.training:
            _increment(model, "seqfish_route_train_forward_count")
        return (h_value, g_value, *args[2:]), kwargs

    model.cooperative_decon.register_forward_pre_hook(
        cooperative_inputs, with_kwargs=True)

    if joint:
        from utils.dacg_readout_adapter import BoundedReadoutAdapter

        model.seqfish_joint_adapter = BoundedReadoutAdapter(
            fusion_dim=int(model.cooperative_decon.fusion_dim),
            num_cell_types=int(model.num_cell_types),
            alpha=float(settings["joint_adapter_alpha"]),
            interaction_weight=float(
                settings["joint_adapter_interaction_weight"]),
        )

    model._seqfish_joint_route = route
    model._seqfish_h_sampling_scale = h_scale
    model._seqfish_g_sampling_scale = g_scale
    model._seqfish_joint_adapter_enabled = joint
    model._seqfish_sota_component_attached = True


def install(settings: Mapping[str, Any]) -> None:
    global _SETTINGS
    normalized = normalize_settings(settings)
    if _SETTINGS and normalized != _SETTINGS:
        raise RuntimeError(
            f"SeqFISH component runtime already configured as {_SETTINGS}")
    _SETTINGS = normalized

    from models.DACG_model import DACGModel

    if getattr(DACGModel, "_seqfish_sota_component_class_patch", False):
        return
    original_init = DACGModel.__init__
    original_forward = DACGModel.forward

    @wraps(original_init)
    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        values = dict(kwargs)
        architecture = str(values.get("decon_architecture", "concat")).lower()
        requested_h = bool(values.get("use_h_stochastic", False))
        requested_g = bool(values.get("use_g_stochastic", False))
        if architecture.startswith("coop_"):
            values["use_h_stochastic"] = False
            values["use_g_stochastic"] = False
        original_init(self, *args, **values)
        if architecture.startswith("coop_"):
            self.use_h_stochastic = requested_h
            self.use_g_stochastic = requested_g
            attach(self, _SETTINGS)

    @wraps(original_forward)
    def patched_forward(self: Any, *args: Any, **kwargs: Any):
        output = original_forward(self, *args, **kwargs)
        adapter = getattr(self, "seqfish_joint_adapter", None)
        if adapter is None:
            return output
        if not all(key in output for key in (
                "decon_logits", "coop_h0", "coop_g0")):
            raise RuntimeError(
                "Joint adapter requires cooperative forward outputs")
        adapted = adapter(
            output["decon_logits"], output["coop_h0"], output["coop_g0"])
        logits = adapted["logits"]
        temperature = self.decon_temp.clamp(min=1.0, max=5.0)
        output["pre_joint_decon_logits"] = output["decon_logits"]
        output["decon_logits"] = logits
        output["decon"] = F.softmax(logits / temperature, dim=-1)
        output["joint_adapter_delta"] = adapted["delta"]
        output["joint_adapter_g_value"] = adapted["g_value"]
        output["joint_adapter_interaction"] = adapted["interaction"]
        _increment(self, "seqfish_joint_adapter_forward_count")
        _assign(
            self, "seqfish_joint_adapter_last_delta_abs_mean",
            float(adapted["delta"].detach().abs().mean().item()),
        )
        if self.training:
            _increment(self, "seqfish_joint_adapter_train_forward_count")
        return output

    DACGModel.__init__ = patched_init
    DACGModel.forward = patched_forward
    DACGModel._seqfish_sota_component_class_patch = True


def install_from_cli() -> None:
    if "-i" not in sys.argv:
        raise RuntimeError("SeqFISH component runtime requires -i CONFIG")
    path = Path(sys.argv[sys.argv.index("-i") + 1])
    install(json.loads(path.read_text(encoding="utf-8-sig")))


def audit(model: torch.nn.Module) -> Dict[str, Any]:
    adapter = getattr(model, "seqfish_joint_adapter", None)
    return {
        "route": str(getattr(model, "_seqfish_joint_route", "missing")),
        "patch_attached": bool(
            getattr(model, "_seqfish_sota_component_attached", False)),
        "h_sampling_scale": float(
            getattr(model, "_seqfish_h_sampling_scale", 0.0)),
        "g_sampling_scale": float(
            getattr(model, "_seqfish_g_sampling_scale", 0.0)),
        "joint_adapter_enabled": bool(
            getattr(model, "_seqfish_joint_adapter_enabled", False)),
        "route_forward_count": int(float(
            getattr(model, "seqfish_route_forward_count", 0.0))),
        "route_train_forward_count": int(float(
            getattr(model, "seqfish_route_train_forward_count", 0.0))),
        "sampled_h_train_count": int(float(
            getattr(model, "seqfish_sampled_h_train_count", 0.0))),
        "sampled_g_train_count": int(float(
            getattr(model, "seqfish_sampled_g_train_count", 0.0))),
        "last_h_delta_mean_abs": float(
            getattr(model, "seqfish_last_h_delta_mean_abs", 0.0)),
        "last_g_delta_mean_abs": float(
            getattr(model, "seqfish_last_g_delta_mean_abs", 0.0)),
        "joint_adapter_train_forward_count": int(float(getattr(
            model, "seqfish_joint_adapter_train_forward_count", 0.0))),
        "joint_adapter_output_weight_norm": (
            float(adapter.output_projection.weight.detach().norm().cpu())
            if adapter is not None else 0.0),
    }


def assert_training_contract(model: torch.nn.Module) -> Dict[str, Any]:
    value = audit(model)
    if not value["patch_attached"] or value["route_train_forward_count"] <= 0:
        raise RuntimeError(f"Cooperative route did not run: {value}")
    route = value["route"]
    if route in {"sampled_h", "sampled_hg"} and (
            value["h_sampling_scale"] > 0
            and value["sampled_h_train_count"] <= 0):
        raise RuntimeError(f"h was never sampled: {value}")
    if route in {"sampled_g", "sampled_hg"} and (
            value["g_sampling_scale"] > 0
            and value["sampled_g_train_count"] <= 0):
        raise RuntimeError(f"g was never sampled: {value}")
    if value["joint_adapter_enabled"] and (
            value["joint_adapter_train_forward_count"] <= 0
            or value["joint_adapter_output_weight_norm"] <= 0.0):
        raise RuntimeError(f"Joint adapter was not optimized: {value}")
    return value
