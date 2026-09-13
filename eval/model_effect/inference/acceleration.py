"""Optional inference acceleration helpers for fixed-shape CUDA workloads."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune-no-cudagraphs",
    "max-autotune",
)
FP8_MODES = ("dynamic",)
_CUDAGRAPH_MODES = {"reduce-overhead", "max-autotune"}
_DISABLED_VALUES = {"", "0", "false", "none", "off"}


def normalize_optional_mode(value, *, choices: tuple[str, ...], name: str) -> str | None:
    if value is None or value is False:
        return None
    if value is True:
        value = choices[0]
    normalized = str(value).strip().lower()
    if normalized in _DISABLED_VALUES:
        return None
    if normalized not in choices:
        raise ValueError(f"Unsupported {name}={value!r}; choose from: {', '.join(choices)}")
    return normalized


def prepare_allocator_for_compile(compile_mode: str | None) -> bool:
    """Disable expandable segments before CUDA init when CUDA graphs are requested."""
    if compile_mode not in _CUDAGRAPH_MODES:
        return False

    changed = False
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        raw = os.environ.get(key)
        if not raw:
            continue
        entries = [item.strip() for item in raw.split(",") if item.strip()]
        kept = [
            item for item in entries
            if item.lower().replace(" ", "") != "expandable_segments:true"
        ]
        if len(kept) == len(entries):
            continue

        torch_module = sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_initialized():
            raise RuntimeError(
                "CUDA is already initialized with expandable_segments enabled; "
                "construct StudentEngine with compile_mode before the first CUDA operation"
            )
        changed = True
        if kept:
            os.environ[key] = ",".join(kept)
        else:
            os.environ.pop(key, None)
    return changed


def _eligible_fp8_linear(module, fqn: str) -> bool:
    import torch

    if not isinstance(module, torch.nn.Linear):
        return False
    if not fqn.startswith("backbone.aggregator."):
        return False
    return (
        module.in_features >= 256
        and module.out_features >= 256
        and module.in_features % 16 == 0
        and module.out_features % 16 == 0
    )


@dataclass(frozen=True)
class FP8Conversion:
    module_names: tuple[str, ...]
    torchao_version: str


def apply_dynamic_fp8(model, device) -> FP8Conversion:
    """Quantize large aggregator Linear layers while keeping all output heads in BF16."""
    import torch

    if device.type != "cuda":
        raise RuntimeError("FP8 inference requires a CUDA device")
    if torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError(
            f"FP8 inference requires CUDA capability >= 8.9, got "
            f"{torch.cuda.get_device_capability(device)} on {device}"
        )
    try:
        import torchao
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            quantize_,
        )
    except Exception as exc:
        raise RuntimeError(
            "FP8 mode requires a torchao release compatible with the installed PyTorch"
        ) from exc

    eligible = [
        (name, module)
        for name, module in model.named_modules()
        if _eligible_fp8_linear(module, name)
    ]
    if not eligible:
        raise RuntimeError("No eligible aggregator Linear layers were found for FP8")

    # torchao preserves the original Linear dtype for its output. BF16 keeps
    # residual paths in the existing autocast dtype instead of promoting to FP32.
    for _name, module in eligible:
        module.to(dtype=torch.bfloat16)

    eligible_ids = {id(module) for _name, module in eligible}
    quantize_(
        model,
        Float8DynamicActivationFloat8WeightConfig(),
        filter_fn=lambda module, _fqn: id(module) in eligible_ids,
    )
    return FP8Conversion(
        module_names=tuple(name for name, _module in eligible),
        torchao_version=str(getattr(torchao, "__version__", "unknown")),
    )


def compile_hotspots(model, compile_mode: str) -> tuple[str, ...]:
    """Compile fixed-shape blocks without tracing the Python/stateful model loop."""
    import torch

    aggregator = model.backbone.aggregator
    compiled = []
    for index, block in enumerate(aggregator.frame_blocks):
        aggregator.frame_blocks[index] = torch.compile(
            block, mode=compile_mode, fullgraph=False, dynamic=False
        )
        compiled.append(f"backbone.aggregator.frame_blocks.{index}")

    patch_blocks = getattr(aggregator.patch_embed, "blocks", None)
    if patch_blocks is not None:
        for index, block in enumerate(patch_blocks):
            patch_blocks[index] = torch.compile(
                block, mode=compile_mode, fullgraph=False, dynamic=False
            )
            compiled.append(f"backbone.aggregator.patch_embed.blocks.{index}")

    # FlashInfer streaming exposes stateless pieces around its paged attention.
    # SDPABlock receives a changing global_idx and recompiles if the whole block
    # is traced, so the SDPA fixed-window path intentionally skips global blocks.
    if not getattr(aggregator, "use_sdpa", False):
        for index, block in enumerate(aggregator.global_blocks):
            if hasattr(block, "attn_pre"):
                block.attn_pre = torch.compile(
                    block.attn_pre, mode=compile_mode, fullgraph=False, dynamic=False
                )
                compiled.append(f"backbone.aggregator.global_blocks.{index}.attn_pre")
            if hasattr(block, "ffn_residual"):
                block.ffn_residual = torch.compile(
                    block.ffn_residual, mode=compile_mode, fullgraph=False, dynamic=False
                )
                compiled.append(f"backbone.aggregator.global_blocks.{index}.ffn_residual")
            block.attn.proj = torch.compile(
                block.attn.proj, mode=compile_mode, fullgraph=False, dynamic=False
            )
            compiled.append(f"backbone.aggregator.global_blocks.{index}.attn.proj")
    return tuple(compiled)
