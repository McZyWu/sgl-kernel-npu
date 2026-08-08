import pytest
import torch

from sgl_kernel_npu.fla.l2norm import l2norm_fwd


DEVICE = "npu"
EPS = 1e-6


def _reference_l2norm(x: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    x_fp32 = x.float()
    squared_sum = torch.sum(x_fp32 * x_fp32, dim=-1, keepdim=True)
    return (x_fp32 / torch.sqrt(squared_sum + EPS)).to(output_dtype)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 128), torch.bfloat16),
        ((35, 128), torch.bfloat16),
        ((109, 128), torch.bfloat16),
        ((110, 128), torch.bfloat16),
        ((2, 17, 128), torch.bfloat16),
        ((4, 344, 128), torch.bfloat16),
        ((110, 128), torch.float16),
    ],
)
def test_l2norm_fwd_kimi_k3_shapes(shape: tuple[int, ...], dtype: torch.dtype):
    torch.manual_seed(42)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)

    actual = l2norm_fwd(x, eps=EPS)
    expected = _reference_l2norm(x, dtype)

    assert actual.shape == x.shape
    assert actual.dtype == dtype
    atol = 5e-4 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("tokens", [1, 109, 110, 1376])
def test_l2norm_fwd_float32_output(tokens: int):
    torch.manual_seed(42)
    x = torch.randn((tokens, 128), device=DEVICE, dtype=torch.bfloat16)

    actual = l2norm_fwd(x, eps=EPS, output_dtype=torch.float32)
    expected = _reference_l2norm(x, torch.float32)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_l2norm_fwd_is_repeatable_at_old_tile_boundary():
    torch.manual_seed(42)
    x = torch.randn((110, 128), device=DEVICE, dtype=torch.bfloat16)

    first = l2norm_fwd(x, eps=EPS)
    second = l2norm_fwd(x, eps=EPS)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
