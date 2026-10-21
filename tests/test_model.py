"""Smoke tests for the WaveFusion-Net model. Run with `pytest` from the repository root."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavefusion import WaveFusionNet  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return WaveFusionNet().eval()


def test_parameter_count_matches_paper(model):
    assert sum(p.numel() for p in model.parameters()) == 9_484_659


@pytest.mark.parametrize("height,width", [(64, 64), (70, 45), (96, 128)])
def test_output_has_input_shape(model, height, width):
    x = torch.rand(1, 3, height, width)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_dwt_roundtrip():
    from wavefusion.model import DWT, IDWT

    x = torch.rand(1, 3, 32, 32)
    assert torch.allclose(IDWT()(*DWT()(x)), x, atol=1e-5)
