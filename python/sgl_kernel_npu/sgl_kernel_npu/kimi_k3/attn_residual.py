import torch
import triton
import triton.language as tl

from sgl_kernel_npu.utils.triton_utils import get_device_properties

# Keep these identical to the score + combine pipeline. Both constants affect
# the FP32 reduction tree and can change the final BF16 mixture.
_BLOCK_H = 32
_MAX_ROWS = 16


@triton.jit(do_not_specialize=["N", "B"])
def _mix_fused_kernel(
    prefix_ptr,
    bank_ptr,
    cw_ptr,
    out_ptr,
    N,
    B,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_om,
    H: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    block_size = (N - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    token_start = pid * block_size
    if token_start >= N:
        return
    token_end = tl.minimum(token_start + block_size, N)

    row_offsets = tl.arange(0, MAX_ROWS)

    for token in range(token_start, token_end):
        scores = tl.full([MAX_ROWS], -float("inf"), dtype=tl.float32)
        for row in range(B + 1):
            squared_sum = 0.0
            dot = 0.0
            for hidden_start in tl.range(
                0,
                H,
                BLOCK_H,
                loop_unroll_factor=1,
                disallow_acc_multi_buffer=True,
            ):
                hidden_offsets = hidden_start + tl.arange(0, BLOCK_H)
                if row < B:
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
                combined_weight = tl.load(cw_ptr + hidden_offsets)
                squared_sum += tl.sum(value * value)
                dot += tl.sum(value * combined_weight)
            inverse_rms = 1.0 / tl.sqrt(squared_sum / H + EPS)
            score = dot * inverse_rms
            scores = tl.where(row_offsets == row, score, scores)

        scores_max = tl.max(scores)
        exp_scores = tl.exp(scores - scores_max)
        probabilities = exp_scores / tl.sum(exp_scores)

        hidden_offsets = tl.arange(0, H)
        output = tl.zeros([H], dtype=tl.float32)
        for row in range(B + 1):
            if row < B:
                value = tl.load(
                    bank_ptr + token * stride_bm + row * stride_bb + hidden_offsets
                ).to(tl.float32)
            else:
                value = tl.load(prefix_ptr + token * stride_pm + hidden_offsets).to(
                    tl.float32
                )
            probability = tl.sum(tl.where(row_offsets == row, probabilities, 0.0))
            output += probability * value

        tl.store(
            out_ptr + token * stride_om + hidden_offsets,
            output.to(out_ptr.dtype.element_ty),
        )


def mix_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    num_valid_blocks: int,
    combined_weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """Ascend Kimi-K3 attention-residual score and combine pipeline.

    Keep scoring, softmax, and mixing in one persistent vector-core kernel to
    avoid materializing the score matrix and launching a second kernel.  N and
    B stay dynamic so prefill/decode shapes reuse the same compilation.
    """
    num_tokens, hidden_size = prefix_sum.shape
    if num_tokens == 0:
        return prefix_sum
    if not 0 <= num_valid_blocks <= bank.shape[1]:
        raise ValueError("num_valid_blocks must fit within the residual bank")
    if num_valid_blocks >= _MAX_ROWS:
        raise ValueError(f"num_valid_blocks must be less than {_MAX_ROWS}")
    if hidden_size % _BLOCK_H:
        raise ValueError(f"hidden size {hidden_size} must be divisible by {_BLOCK_H}")

    output = torch.empty_like(prefix_sum)
    _, num_vector_cores = get_device_properties()
    _mix_fused_kernel[(num_vector_cores,)](
        prefix_sum,
        bank,
        combined_weight,
        output,
        num_tokens,
        num_valid_blocks,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        output.stride(0),
        H=hidden_size,
        EPS=variance_epsilon,
        NUM_CORES=num_vector_cores,
        BLOCK_H=_BLOCK_H,
        MAX_ROWS=_MAX_ROWS,
        multibuffer=True,
    )
    return output
