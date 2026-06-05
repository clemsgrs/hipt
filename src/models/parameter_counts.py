from __future__ import annotations

import logging
from collections.abc import Callable

import torch.nn as nn


def count_trainable_parameters(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def trainable_parameter_breakdown(model: nn.Module) -> dict[str, int]:
    """Return trainable parameter counts for the main HIPT components."""
    second_transformer_modules = [
        getattr(model, name, None)
        for name in (
            "global_phi",
            "global_transformer",
            "global_attn_pool",
            "global_rho",
        )
    ]

    breakdown = {
        "total": count_trainable_parameters(model),
        "vit_region": count_trainable_parameters(getattr(model, "vit_region", None)),
        "second_transformer_block": sum(
            count_trainable_parameters(module) for module in second_transformer_modules
        ),
    }

    if hasattr(model, "dctm_head"):
        breakdown["classifier_head_dctm"] = count_trainable_parameters(model.dctm_head)
    elif hasattr(model, "classifier"):
        breakdown["classifier_head_bin"] = count_trainable_parameters(model.classifier)
    else:
        breakdown["classifier_head"] = 0

    return breakdown


def format_trainable_parameter_breakdown(model: nn.Module) -> str:
    breakdown = trainable_parameter_breakdown(model)
    parts = [f"{name}={count:,}" for name, count in breakdown.items()]
    return "Trainable parameters: " + "; ".join(parts)


def log_trainable_parameter_breakdown(
    model: nn.Module,
    logger: logging.Logger,
    *,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    message = format_trainable_parameter_breakdown(model)
    if log_fn is not None:
        log_fn(message)
    else:
        logger.info(message)
