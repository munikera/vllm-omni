# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Autotune FlashInfer's MXFP8 GEMM for a diffusion pipeline at startup.

vLLM's LLM engine tunes FlashInfer kernels during kernel warmup
(``vllm.model_executor.warmup.kernel_warmup.flashinfer_autotune``). The diffusion runner has no such
step, so ``mm_mxfp8`` (``FlashInferCutlassMxfp8LinearKernel``) runs FlashInfer's default tactic. That
tactic is shape-sensitive: on SM120 (RTX PRO 6000 / 5000 Blackwell), for Wan2.2 A14B's 15360 x 5120
and 13824 x 5120 GEMMs it falls to about 73 TFLOPS between roughly 20K and 64K rows, against about
385 TFLOPS tuned, which makes 832x480x81 video about 2.4x slower end to end.

FlashInfer keys tuning results by the weight shape and, per tuned call, covers the M buckets up to
that call's M. The engine's startup dummy request is small (512x512), so tuning it is not enough.
This module runs one explicit call per distinct (N, K) with a synthetic activation of
``VLLM_OMNI_FLASHINFER_AUTOTUNE_MAX_TOKENS`` rows inside ``flashinfer.autotuner.autotune(True)``,
using each layer's own weight and scales.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion import envs

logger = init_logger(__name__)

_MXFP8_KERNEL = "FlashInferCutlassMxfp8LinearKernel"


def collect_mxfp8_shapes(
    modules: Iterable[torch.nn.Module],
) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    """Distinct (N, K) weight shapes of FlashInfer-cutlass MXFP8 linear layers, with one layer's weight and scale."""
    shapes: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for module in modules:
        kernel = getattr(getattr(module, "quant_method", None), "kernel", None)
        if type(kernel).__name__ != _MXFP8_KERNEL:
            continue
        weight = getattr(module, "weight", None)
        scale = getattr(module, "weight_scale", None)
        if weight is None or scale is None or weight.dim() != 2:
            continue
        shapes.setdefault((int(weight.shape[0]), int(weight.shape[1])), (weight, scale))
    return shapes


def autotune_flashinfer_mxfp8(pipeline: torch.nn.Module) -> int:
    """Tune ``mm_mxfp8`` for every MXFP8 GEMM shape of ``pipeline``. Returns the number of shapes tuned."""
    if not envs.VLLM_OMNI_FLASHINFER_AUTOTUNE:
        return 0
    shapes = collect_mxfp8_shapes(pipeline.modules())
    if not shapes:
        return 0
    try:
        import vllm.utils.flashinfer  # noqa: F401  (registers torch.ops.vllm.mm_mxfp8)
        from flashinfer.autotuner import autotune
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    except ImportError as exc:
        logger.info("FlashInfer MXFP8 autotune skipped: %s", exc)
        return 0

    max_tokens = envs.VLLM_OMNI_FLASHINFER_AUTOTUNE_MAX_TOKENS
    start = time.perf_counter()
    with torch.inference_mode(), autotune(True):
        for (_, k), (weight, scale) in shapes.items():
            x = torch.randn(max_tokens, k, device=weight.device, dtype=torch.bfloat16)
            x_q, x_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            del x
            torch.ops.vllm.mm_mxfp8(x_q, weight.t(), x_scale, scale, torch.bfloat16, "cutlass")
            del x_q, x_scale
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    logger.info(
        "FlashInfer MXFP8 autotune: %d GEMM shapes %s up to %d tokens in %.1f s",
        len(shapes),
        sorted(shapes),
        max_tokens,
        time.perf_counter() - start,
    )
    return len(shapes)
