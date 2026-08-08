import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from sgl_kernel_npu.utils.triton_utils import get_device_properties
from triton.language.extra.cann import libdevice


@triton.jit(do_not_specialize=["N_ROWS"])
def _situ_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALIGNED: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    N_ROWS,
    NUM_CORES: tl.constexpr,
    HAS_GROUP_LIST: tl.constexpr,
    BETA: tl.constexpr,
    INV_BETA: tl.constexpr,
    DO_LINEAR_BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    INV_LINEAR_BETA: tl.constexpr,
    NEED_QUANT: tl.constexpr,
):
    if HAS_GROUP_LIST:
        if GROUP_LIST_TYPE == 0:
            total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
        else:
            offsets = tl.arange(0, NUM_EXPERTS_ALIGNED)
            mask = offsets < NUM_EXPERTS
            counts = tl.load(group_list_ptr + offsets, mask=mask, other=0).to(tl.int32)
            total_rows = tl.sum(counts)
    else:
        total_rows = N_ROWS

    rows_per_core = (total_rows - 1) // NUM_CORES + 1
    row_begin = tl.program_id(0) * rows_per_core
    if row_begin >= total_rows:
        return
    row_end = tl.minimum(row_begin + rows_per_core, total_rows)

    for row in range(row_begin, row_end):
        row_offset = row.to(tl.int64) * TOTAL_COLS
        if NEED_QUANT:
            cols = tl.arange(0, HALF_COLS)
            gate = tl.load(x_ptr + row_offset + cols).to(tl.float32)
            up = tl.load(x_ptr + row_offset + HALF_COLS + cols).to(tl.float32)
            gate = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
            if DO_LINEAR_BETA:
                up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
            value = gate * up
            scale = tl.maximum(tl.max(tl.abs(value)) / 127.0, 1e-30)
            tl.store(scale_ptr + row.to(tl.int64), scale.to(scale_ptr.dtype.element_ty))
            for col_begin in range(0, HALF_COLS, COL_BLOCK_SIZE):
                block = al.extract_slice(
                    value,
                    offsets=(col_begin,),
                    sizes=(COL_BLOCK_SIZE,),
                    strides=(1,),
                )
                block = tl.floor(block.to(tl.float32) / scale + 0.5)
                block = tl.clamp(block, -128, 127).to(tl.int8)
                block_cols = col_begin + tl.arange(0, COL_BLOCK_SIZE)
                tl.store(
                    out_ptr + row.to(tl.int64) * HALF_COLS + block_cols,
                    block.to(out_ptr.dtype.element_ty),
                    mask=block_cols < HALF_COLS,
                )
        else:
            cols = tl.arange(0, COL_BLOCK_SIZE)
            for col_begin in range(0, HALF_COLS, COL_BLOCK_SIZE):
                block_cols = col_begin + cols
                mask = block_cols < HALF_COLS
                gate = tl.load(
                    x_ptr + row_offset + block_cols, mask=mask, other=0.0
                ).to(tl.float32)
                up = tl.load(
                    x_ptr + row_offset + HALF_COLS + block_cols,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                gate = BETA * libdevice.tanh(gate * INV_BETA) * tl.sigmoid(gate)
                if DO_LINEAR_BETA:
                    up = LINEAR_BETA * libdevice.tanh(up * INV_LINEAR_BETA)
                value = gate * up
                tl.store(
                    out_ptr + row.to(tl.int64) * HALF_COLS + block_cols,
                    value.to(out_ptr.dtype.element_ty),
                    mask=mask,
                )


def situ(
    hidden_states: torch.Tensor,
    group_list: torch.Tensor | None = None,
    group_list_type: int | None = None,
    *,
    need_quant: bool,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply Kimi-K3 SiTU with optional grouping and INT8 requantization.

    ``group_list=None`` processes every row and is intended for dense/shared
    experts. Routed MoE callers pass their count or cumulative group list.
    """
    has_group_list = group_list is not None
    if has_group_list and group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
    if hidden_states.ndim < 1 or hidden_states.shape[-1] % 2:
        raise ValueError("SiTU input must have shape [..., 2 * intermediate]")
    if hidden_states.shape[-1] == 0:
        raise ValueError("SiTU input must have a non-empty last dimension")

    hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
    if has_group_list:
        if group_list.dtype == torch.int64:
            num_experts_aligned = (group_list.numel() + 7) // 8 * 8
        elif group_list.dtype == torch.int32:
            num_experts_aligned = (group_list.numel() + 15) // 16 * 16
        else:
            raise ValueError("group_list must use int32 or int64")
        group_list_arg = group_list
        num_experts = group_list.numel()
        group_list_type_arg = group_list_type
    else:
        # The dense kernel branch never reads this pointer.
        group_list_arg = hidden_states_2d
        num_experts = 1
        num_experts_aligned = 1
        group_list_type_arg = 0

    rows, total_cols = hidden_states_2d.shape
    half_cols = total_cols // 2
    out = torch.empty(
        (rows, half_cols),
        dtype=torch.int8 if need_quant else hidden_states.dtype,
        device=hidden_states.device,
    )
    scale = (
        torch.empty(rows, dtype=torch.float32, device=hidden_states.device)
        if need_quant
        else None
    )
    if rows == 0:
        return out.reshape(*hidden_states.shape[:-1], half_cols), scale

    _, num_vector_cores = get_device_properties()
    linear_beta_value = linear_beta if linear_beta is not None else 1.0
    col_block_size = (
        half_cols if need_quant else min(4096, triton.next_power_of_2(half_cols))
    )
    _situ_kernel[(num_vector_cores,)](
        hidden_states_2d,
        group_list_arg,
        out,
        scale if scale is not None else out,
        TOTAL_COLS=total_cols,
        HALF_COLS=half_cols,
        COL_BLOCK_SIZE=col_block_size,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_ALIGNED=num_experts_aligned,
        GROUP_LIST_TYPE=group_list_type_arg,
        N_ROWS=rows,
        NUM_CORES=num_vector_cores,
        HAS_GROUP_LIST=has_group_list,
        BETA=float(beta),
        INV_BETA=1.0 / float(beta),
        DO_LINEAR_BETA=linear_beta is not None,
        LINEAR_BETA=float(linear_beta_value),
        INV_LINEAR_BETA=1.0 / float(linear_beta_value),
        NEED_QUANT=need_quant,
        multibuffer=True,
    )
    return out.reshape(*hidden_states.shape[:-1], half_cols), scale
