import math

import pytest
import torch
import triton
import triton.language as tl

from sgl_kernel_npu.kimi_k3.attn_residual import mix_fused


BLOCK_H = 32
HIDDEN_SIZE = 7168
MAX_ROWS = 16
EPS = 1e-6


@triton.jit
def _score_reference_kernel(
    prefix_ptr,
    bank_ptr,
    combined_weight_ptr,
    scores_ptr,
    num_valid_blocks,
    eps,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    token = tl.program_id(0)
    row = tl.program_id(1)
    if row > num_valid_blocks:
        return

    squared_sum = 0.0
    dot = 0.0
    for hidden_start in tl.static_range(0, H, BLOCK_SIZE_H):
        hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE_H)
        if row < num_valid_blocks:
            value = tl.load(
                bank_ptr
                + token * stride_bm
                + row * stride_bb
                + hidden_offsets
            ).to(tl.float32)
        else:
            value = tl.load(
                prefix_ptr + token * stride_pm + hidden_offsets
            ).to(tl.float32)
        combined_weight = tl.load(combined_weight_ptr + hidden_offsets)
        squared_sum += tl.sum(value * value)
        dot += tl.sum(value * combined_weight)

    inverse_rms = 1.0 / tl.sqrt(squared_sum / H + eps)
    tl.store(scores_ptr + token * stride_sm + row, dot * inverse_rms)


@triton.jit
def _combine_reference_kernel(
    prefix_ptr,
    bank_ptr,
    scores_ptr,
    output_ptr,
    num_valid_blocks,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    BLOCK_SIZE_H: tl.constexpr,
    NUM_ROWS: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    row_offsets = tl.arange(0, NUM_ROWS)
    valid_rows = row_offsets <= num_valid_blocks
    scores = tl.load(
        scores_ptr + token * stride_sm + row_offsets,
        mask=valid_rows,
        other=float("-inf"),
    )
    scores_max = tl.max(scores)
    exp_scores = tl.where(valid_rows, tl.exp(scores - scores_max), 0.0)
    probabilities = exp_scores / tl.sum(exp_scores)

    output = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    for row in range(0, num_valid_blocks + 1):
        if row < num_valid_blocks:
            value = tl.load(
                bank_ptr
                + token * stride_bm
                + row * stride_bb
                + hidden_offsets
            ).to(tl.float32)
        else:
            value = tl.load(
                prefix_ptr + token * stride_pm + hidden_offsets
            ).to(tl.float32)
        probability = tl.sum(tl.where(row_offsets == row, probabilities, 0.0))
        output += probability * value

    tl.store(
        output_ptr + token * stride_om + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
    )


def _score_combine_reference(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    num_valid_blocks: int,
    combined_weight: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden_size = prefix_sum.shape
    scores = torch.empty(
        (num_tokens, MAX_ROWS), dtype=torch.float32, device=prefix_sum.device
    )
    _score_reference_kernel[(num_tokens, num_valid_blocks + 1)](
        prefix_sum,
        bank,
        combined_weight,
        scores,
        num_valid_blocks,
        EPS,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        H=hidden_size,
        BLOCK_SIZE_H=BLOCK_H,
        num_warps=4,
    )

    output = torch.empty_like(prefix_sum)
    _combine_reference_kernel[(num_tokens, hidden_size // BLOCK_H)](
        prefix_sum,
        bank,
        scores,
        output,
        num_valid_blocks,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        output.stride(0),
        BLOCK_SIZE_H=BLOCK_H,
        NUM_ROWS=MAX_ROWS,
        num_warps=4,
    )
    return output


@pytest.mark.parametrize(
    ("num_tokens", "num_valid_blocks"),
    [(1, 1), (4, 4), (17, 8)],
)
def test_mix_fused_matches_score_combine_reduction(
    num_tokens: int, num_valid_blocks: int
):
    torch.manual_seed(42)
    prefix_sum = torch.randn(
        (num_tokens, HIDDEN_SIZE), device="npu", dtype=torch.bfloat16
    )
    bank = torch.randn(
        (num_tokens, 8, HIDDEN_SIZE), device="npu", dtype=torch.bfloat16
    )
    combined_weight = torch.randn(
        (HIDDEN_SIZE,), device="npu", dtype=torch.float32
    ) / math.sqrt(HIDDEN_SIZE)

    expected = _score_combine_reference(
        prefix_sum, bank, num_valid_blocks, combined_weight
    )
    actual = mix_fused(
        prefix_sum,
        bank,
        num_valid_blocks,
        combined_weight,
        EPS,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
