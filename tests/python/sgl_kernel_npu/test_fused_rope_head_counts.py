import itertools

import pytest
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.norm.fused_rope_qk_mqa import fused_rope_qk_mqa


def _reference(x, cache, dim, neox):
    value = x.float().cpu()
    cs = cache.float().cpu()
    cos, sin = cs[:, : dim // 2, None], cs[:, dim // 2 :, None]
    cos, sin = cos.transpose(1, 2), sin.transpose(1, 2)
    rotary = value[..., :dim]
    left = rotary[..., : dim // 2] if neox else rotary[..., ::2]
    right = rotary[..., dim // 2 :] if neox else rotary[..., 1::2]
    result = value.clone()
    if neox:
        result[..., : dim // 2] = left * cos - right * sin
        result[..., dim // 2 : dim] = left * sin + right * cos
    else:
        result[..., :dim:2] = left * cos - right * sin
        result[..., 1:dim:2] = left * sin + right * cos
    return result.to(x.dtype)


@pytest.mark.parametrize("heads_q,heads_k", [(4, 1), (4, 2), (5, 3), (10, 6), (20, 10)])
@pytest.mark.parametrize("neox,head_dim,rotary_dim", [(True, 64, 64), (False, 128, 64)])
@torch.inference_mode()
def test_rope_arbitrary_heads_and_dynamic_graph_inputs(
    heads_q, heads_k, neox, head_dim, rotary_dim
):
    tokens = 5
    # Keep sliced head and row strides; output padding is never a valid head.
    q_store = torch.randn(
        tokens, heads_q + 1, head_dim, device="npu", dtype=torch.bfloat16
    )
    k_store = torch.randn(
        tokens, heads_k + 1, head_dim, device="npu", dtype=torch.bfloat16
    )
    q, k = q_store[:, :heads_q], k_store[:, :heads_k]
    cache = torch.randn(tokens, rotary_dim, device="npu")
    fused_rope_qk_mqa(q, k, cache, rotary_dim, neox)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        oq, ok = fused_rope_qk_mqa(q, k, cache, rotary_dim, neox)
    for _ in range(2):
        q.normal_()
        k.normal_()
        cache.normal_()
        graph.replay()
        for result, source in ((oq, q), (ok, k)):
            torch.testing.assert_close(
                result.cpu(),
                _reference(source, cache, rotary_dim, neox),
                rtol=0.008,
                atol=0.008,
            )
            torch.testing.assert_close(
                result[..., rotary_dim:], source[..., rotary_dim:], rtol=0, atol=0
            )


if __name__ == "__main__":
    for (heads_q, heads_k), (neox, head_dim, rotary_dim) in itertools.product(
        [(4, 1), (4, 2), (5, 3), (10, 6), (20, 10)],
        [(True, 64, 64), (False, 128, 64)],
    ):
        test_rope_arbitrary_heads_and_dynamic_graph_inputs(
            heads_q, heads_k, neox, head_dim, rotary_dim
        )
