import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.activation.situ import situ


def _situ_reference_fp32(
    x: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    gate, up = x.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def _situ_reference(
    x: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    return _situ_reference_fp32(x, beta, linear_beta).to(x.dtype)


@pytest.mark.parametrize(
    "shape",
    [
        (7, 6144),
        (3, 67584),
        (2, 3, 1024),
    ],
)
@pytest.mark.parametrize("linear_beta", [25.0, None])
def test_situ_without_group_list(
    shape: tuple[int, ...], linear_beta: float | None
) -> None:
    beta = 4.0
    x = torch.randn(shape, dtype=torch.bfloat16, device="npu")

    actual, scale = situ(
        x,
        need_quant=False,
        beta=beta,
        linear_beta=linear_beta,
    )
    expected = _situ_reference(x, beta, linear_beta)

    assert scale is None
    assert actual.shape == shape[:-1] + (shape[-1] // 2,)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)


def test_situ_grouped_interface_is_unchanged() -> None:
    beta = 4.0
    linear_beta = 25.0
    x = torch.randn((8, 6144), dtype=torch.bfloat16, device="npu")
    group_list = torch.tensor([2, 0, 3], dtype=torch.int64, device="npu")

    actual, scale = situ(
        x,
        group_list,
        1,
        need_quant=False,
        beta=beta,
        linear_beta=linear_beta,
    )
    expected = _situ_reference(x, beta, linear_beta)

    assert scale is None
    torch.testing.assert_close(actual[:5], expected[:5], atol=2e-2, rtol=1e-2)


def test_situ_grouped_quantization_is_unchanged() -> None:
    beta = 4.0
    linear_beta = 25.0
    x = torch.randn((8, 6144), dtype=torch.bfloat16, device="npu")
    group_list = torch.tensor([2, 0, 3], dtype=torch.int64, device="npu")

    actual, scale = situ(
        x,
        group_list,
        1,
        need_quant=True,
        beta=beta,
        linear_beta=linear_beta,
    )
    expected_value = _situ_reference_fp32(x[:5], beta, linear_beta)
    expected_scale = torch.clamp(expected_value.abs().amax(dim=-1) / 127.0, min=1e-30)
    expected = torch.floor(expected_value / expected_scale[:, None] + 0.5)
    expected = expected.clamp(-128, 127).to(torch.int8)

    assert scale is not None
    assert torch.max(torch.abs(actual[:5].int() - expected.int())) <= 1
    torch.testing.assert_close(scale[:5], expected_scale, atol=1e-5, rtol=5e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
