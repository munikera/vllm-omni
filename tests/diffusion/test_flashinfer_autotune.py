# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

import vllm_omni.diffusion.worker.flashinfer_autotune as fi_autotune

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


class FlashInferCutlassMxfp8LinearKernel:
    """Stand-in with the same class name as vLLM's kernel, which is what the collector matches."""


class OtherKernel:
    pass


def _linear(n, k, kernel_cls):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.zeros(n, k), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.zeros(n, k // 32), requires_grad=False)
    layer.quant_method = type("Method", (), {})()
    layer.quant_method.kernel = kernel_cls()
    return layer


def _pipeline(*layers):
    model = torch.nn.Module()
    for i, layer in enumerate(layers):
        model.add_module(f"l{i}", layer)
    return model


def test_collect_mxfp8_shapes_dedups_and_filters():
    pipeline = _pipeline(
        _linear(15360, 5120, FlashInferCutlassMxfp8LinearKernel),
        _linear(15360, 5120, FlashInferCutlassMxfp8LinearKernel),
        _linear(5120, 13824, FlashInferCutlassMxfp8LinearKernel),
        _linear(4096, 4096, OtherKernel),
        torch.nn.Linear(8, 8),
    )
    shapes = fi_autotune.collect_mxfp8_shapes(pipeline.modules())
    assert sorted(shapes) == [(5120, 13824), (15360, 5120)]


def test_autotune_disabled_by_env(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLASHINFER_AUTOTUNE", "0")
    pipeline = _pipeline(_linear(15360, 5120, FlashInferCutlassMxfp8LinearKernel))
    assert fi_autotune.autotune_flashinfer_mxfp8(pipeline) == 0


def test_autotune_skips_pipeline_without_mxfp8(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLASHINFER_AUTOTUNE", "1")
    assert fi_autotune.autotune_flashinfer_mxfp8(_pipeline(torch.nn.Linear(8, 8))) == 0
