import pytest
import torch
import torch.nn.functional as F
from sgl_kernel_npu.fla.kda_gate import _kda_gate_fwd_kernel, fused_kda_gate_npu


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 32])
@pytest.mark.parametrize("lower_bound", [None, -0.1])
@torch.no_grad()
def test_kda_gate_dynamic_tokens(tokens: int, lower_bound: float | None):
    heads, head_dim = 4, 64
    gate = torch.randn(tokens, heads * head_dim, device="npu", dtype=torch.bfloat16)
    A_log = torch.randn(heads, device="npu", dtype=torch.float32) * 0.1
    gate_bias = torch.randn(heads * head_dim, device="npu", dtype=torch.bfloat16)

    actual = fused_kda_gate_npu(
        gate,
        A_log,
        head_dim,
        gate_bias=gate_bias,
        lower_bound=lower_bound,
    )

    gate_fp32 = gate.float().view(tokens, heads, head_dim)
    gate_fp32 += gate_bias.float().view(1, heads, head_dim)
    exp_A = torch.exp(A_log).view(1, heads, 1)
    if lower_bound is None:
        expected = -exp_A * F.softplus(gate_fp32)
    else:
        expected = lower_bound * torch.sigmoid(exp_A * gate_fp32)

    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


def test_kda_gate_disables_token_specialization():
    assert "T" in _kda_gate_fwd_kernel.fn.do_not_specialize
