"""
Complete the following functions:
    Fully fused gather-scatter with built-in masking for mamba state updates.

    This function fuses the following operations into a single kernel:
    1. valid_mask = step_indices_raw >= 0
    2. valid_indices = valid_mask.nonzero()
    3. dst_indices = dst_indices_raw[valid_indices]  (index_select)
    4. step_indices = step_indices_raw[valid_indices]  (index_select)
    5. for each valid i: dst[:, dst_indices[i], :] = src[:, i, step_indices[i], :]

follow gpu kernel: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def move_cache_dynamic_last_kernel_h_block(
    dst_cache_ptr,
    src_cache_ptr,
    valid_indices_ptr,
    last_steps_ptr,
    layer_stride,
    size_stride,
    draft_stride,
    dst_layer_stride,
    dst_size_stride,
    num_src_slots,
    num_dst_slots,
    num_drafts,
    h_dim,
    dim_v,
    dim_k,
    num_layers,
    H_BLOCK_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,  # Block size for dim_v
    BLOCK_K: tl.constexpr,  # Block size for dim_k
):
    valid_id = tl.program_id(0)

    # Load actual indices
    valid_idx_val = tl.load(valid_indices_ptr + valid_id)
    last_step_val = tl.load(last_steps_ptr + valid_id)
    if valid_idx_val < 0:
        return
    if valid_idx_val >= num_src_slots:
        return
    if valid_idx_val >= num_dst_slots:
        return
    if last_step_val < 0:
        return
    if last_step_val >= num_drafts:
        return
    h_offsets = tl.arange(0, H_BLOCK_SIZE)
    k_offsets = tl.arange(0, BLOCK_K)

    # Process each layer
    for l in range(num_layers):
        src_base_addr = (
            src_cache_ptr
            + tl.cast(l, tl.int64) * layer_stride
            + tl.cast(valid_idx_val, tl.int64) * size_stride
        )
        dst_base_addr = (
            dst_cache_ptr
            + tl.cast(l, tl.int64) * dst_layer_stride
            + tl.cast(valid_idx_val, tl.int64) * dst_size_stride
        )
        src_addr = src_base_addr + tl.cast(last_step_val, tl.int64) * draft_stride

        # Process h dimension in blocks
        for h_start in range(0, h_dim, H_BLOCK_SIZE):
            h_real = h_start + h_offsets
            h_mask = h_real < h_dim

            for v_start in range(0, dim_v, BLOCK_V):
                v_real = v_start + tl.arange(0, BLOCK_V)
                v_mask = v_real < dim_v
                k_mask = k_offsets < dim_k

                mask = (
                    h_mask[:, None, None]
                    & v_mask[None, :, None]
                    & k_mask[None, None, :]
                )

                linear_offset = (
                    h_real[:, None, None] * dim_v * dim_k
                    + v_real[None, :, None] * dim_k
                    + k_offsets[None, None, :]
                )

                src_block = tl.load(src_addr + linear_offset, mask=mask, other=0)
                # recurrent_gated_delta_rule consumes recurrent_state through a
                # raw data_ptr without tensor strides. Keep the physical dense
                # slot layout instead of applying the destination view's
                # logical H/V/K permutation a second time.
                tl.store(dst_base_addr + linear_offset, src_block, mask=mask)


def move_intermediate_cache(
    ssm_states,
    intermediate_state_cache,
    valid_tensor,
    last_steps_tensor,
    h_block_size=1,
):
    """
    Move intermediate cache to SSM states using Triton kernel.

    Args:
        ssm_states: Destination SSM states tensor. Its per-slot backing storage
            must be dense, but its logical H/V/K view may be a permutation.
        intermediate_state_cache: Source intermediate state cache. The final
            H/V/K dimensions must be contiguous in that order.
        valid_tensor: Valid indices tensor
        last_steps_tensor: Last steps tensor
        h_block_size: Block size for h dimension processing
    """
    if intermediate_state_cache.ndim != 6:
        raise ValueError(
            "intermediate_state_cache must be 6D [L, S, D, H, V, K], "
            f"got {intermediate_state_cache.ndim}D"
        )
    if ssm_states.ndim != 5:
        raise ValueError(
            f"ssm_states must be 5D [L, S, H, V, K], got {ssm_states.ndim}D"
        )
    if valid_tensor.ndim != 1 or last_steps_tensor.ndim != 1:
        raise ValueError("valid_tensor and last_steps_tensor must be 1D")
    if (
        valid_tensor.device != ssm_states.device
        or last_steps_tensor.device != ssm_states.device
    ):
        raise ValueError("all cache and index tensors must be on the same device")
    if intermediate_state_cache.device != ssm_states.device:
        raise ValueError("all cache and index tensors must be on the same device")
    if ssm_states.dtype != intermediate_state_cache.dtype:
        raise ValueError(
            "source and destination cache dtypes must match, "
            f"got {intermediate_state_cache.dtype} and {ssm_states.dtype}"
        )
    for name, tensor in (
        ("valid_tensor", valid_tensor),
        ("last_steps_tensor", last_steps_tensor),
    ):
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must use int32 or int64, got {tensor.dtype}")

    L, S, D, H, V, K = intermediate_state_cache.shape
    if min(L, S, D, H, V, K) <= 0:
        raise ValueError("cache dimensions must all be positive")
    if ssm_states.shape[0] != L:
        raise ValueError(
            "source and destination layer counts must match, "
            f"got {L} and {ssm_states.shape[0]}"
        )

    inner_numel = H * V * K
    dst_inner_numel = ssm_states.shape[2] * ssm_states.shape[3] * ssm_states.shape[4]
    if dst_inner_numel != inner_numel:
        raise ValueError(
            "source and destination state sizes must match, "
            f"got {inner_numel} and {dst_inner_numel} elements per slot"
        )

    strides = intermediate_state_cache.stride()
    layer_stride, size_stride, draft_stride = (
        int(strides[0]),
        int(strides[1]),
        int(strides[2]),
    )
    expected_src_inner_strides = (V * K, K, 1)
    src_inner_strides = tuple(map(int, strides[3:6]))
    if src_inner_strides != expected_src_inner_strides:
        raise ValueError(
            "intermediate_state_cache H/V/K dimensions must be contiguous; "
            f"expected strides {expected_src_inner_strides}, got {src_inner_strides}"
        )

    dst_strides = ssm_states.stride()
    dst_layer_stride, dst_size_stride = int(dst_strides[0]), int(dst_strides[1])

    # The NPU recurrent GDN op ignores tensor strides and consumes each state
    # slot through a raw dense pointer. Dense inner permutations are valid ABI
    # shims, while views with holes or overlapping elements are unsafe.
    dense_stride = 1
    for stride, size in sorted(
        zip(map(int, dst_strides[2:5]), map(int, ssm_states.shape[2:5]))
    ):
        if size > 1:
            if stride != dense_stride:
                raise ValueError(
                    "ssm_states must have dense per-slot backing storage; "
                    f"got inner shape {tuple(ssm_states.shape[2:5])} and "
                    f"strides {tuple(dst_strides[2:5])}"
                )
            dense_stride *= size
    if dst_size_stride < inner_numel:
        raise ValueError(
            "ssm_states slot stride is too small for one dense state block"
        )
    if dst_layer_stride < ssm_states.shape[1] * dst_size_stride:
        raise ValueError(
            "ssm_states layer stride is too small for all destination slots"
        )
    if len(valid_tensor) != len(last_steps_tensor):
        raise ValueError("valid indices and last steps lengths must match")
    if h_block_size <= 0 or h_block_size & (h_block_size - 1):
        raise ValueError("h_block_size must be a positive power of two")

    if len(valid_tensor) == 0:
        return ssm_states

    valid_tensor = valid_tensor.contiguous()
    last_steps_tensor = last_steps_tensor.contiguous()

    # Grid: one thread per valid index
    grid = (len(valid_tensor),)

    move_cache_dynamic_last_kernel_h_block[grid](
        dst_cache_ptr=ssm_states,
        src_cache_ptr=intermediate_state_cache,
        valid_indices_ptr=valid_tensor,
        last_steps_ptr=last_steps_tensor,
        layer_stride=layer_stride,
        size_stride=size_stride,
        draft_stride=draft_stride,
        dst_layer_stride=dst_layer_stride,
        dst_size_stride=dst_size_stride,
        num_src_slots=S,
        num_dst_slots=ssm_states.shape[1],
        num_drafts=D,
        h_dim=H,
        dim_v=V,
        dim_k=K,
        num_layers=L,
        H_BLOCK_SIZE=h_block_size,
        BLOCK_V=64,
        BLOCK_K=triton.next_power_of_2(K),  # Block size for dim_k
    )

    return ssm_states


@triton.jit
def _conv_state_rollback_kernel(
    conv_states_ptr,
    state_indices_ptr,
    step_indices_ptr,
    draft_token_num,
    num_slots,
    num_layers,
    num_dims,
    conv_window_size: tl.constexpr,
    layer_stride: tl.constexpr,
    req_stride: tl.constexpr,
    window_stride: tl.constexpr,
    dim_stride: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    """
    Triton kernel for rolling back conv states after MTP verification.

    Args:
        conv_states_ptr: Pointer to conv states tensor [num_layers, pool_size, conv_window_size, num_dims]
        state_indices_ptr: Pointer to state indices [num_requests]
        step_indices_ptr: Pointer to step indices (accepted steps) [num_requests]
        draft_token_num: Number of draft tokens
        num_slots: Number of request slots in conv_states
        num_layers: Number of layers
        num_dims: Number of dimensions
        conv_window_size: Convolution window size
        layer_stride: Stride for layer dimension
        req_stride: Stride for request dimension
        window_stride: Stride for window dimension
        dim_stride: Stride for dimension dimension
    """
    pid_req = tl.program_id(0)

    # Load state and step indices
    state_idx = tl.load(state_indices_ptr + pid_req).to(tl.int64)
    step_idx = tl.load(step_indices_ptr + pid_req).to(tl.int64)

    if state_idx < 0:
        return
    if state_idx >= num_slots:
        return
    if step_idx < 0:
        return
    if step_idx >= draft_token_num:
        return

    # Calculate rollback shift
    shift = (draft_token_num - 1) - step_idx

    # Early exit if no rollback needed
    if shift <= 0:
        return

    # Generate dimension offsets once
    dim_offsets = tl.arange(0, BLOCK_DIM)
    dim_mask = dim_offsets < num_dims

    # Process each layer
    for layer in range(num_layers):
        # Calculate base offset for this request and layer
        base_offset = state_idx * req_stride + layer * layer_stride

        # Process each window position that needs to be moved
        # Move data from [0, conv_window_size-shift) to [shift, conv_window_size)
        for window_idx1 in range(0, conv_window_size - shift):
            window_idx = conv_window_size - shift - 1 - window_idx1

            # Calculate source and destination pointers
            src_offset = (
                base_offset + window_idx * window_stride + dim_offsets * dim_stride
            )
            src_ptr = conv_states_ptr + src_offset

            dst_offset = (
                base_offset
                + (window_idx + shift) * window_stride
                + dim_offsets * dim_stride
            )
            dst_ptr = conv_states_ptr + dst_offset

            # Load and store all dimensions at once
            data = tl.load(src_ptr, mask=dim_mask)
            tl.store(dst_ptr, data, mask=dim_mask)


def conv_state_rollback(
    conv_states: torch.Tensor,  # [num_layers, pool_size, conv_window_size, num_dims]
    state_indices: torch.Tensor,  # [num_requests]
    step_indices: torch.Tensor,  # [num_requests]
    draft_token_num: int,
):
    """
    Roll back conv states after MTP verification using Triton kernel.

    Args:
        conv_states: Conv states tensor [num_layers, pool_size, conv_window_size, num_dims]
        state_indices: State indices for each request [num_requests]
        step_indices: Accepted steps for each request [num_requests]
        draft_token_num: Number of draft tokens
    """
    if conv_states.ndim != 4:
        raise ValueError(f"conv_states must be 4D, got {conv_states.ndim}D")
    if state_indices.ndim != 1 or step_indices.ndim != 1:
        raise ValueError("state_indices and step_indices must be 1D")
    if state_indices.shape[0] != step_indices.shape[0]:
        raise ValueError("state_indices and step_indices must have the same length")
    if (
        state_indices.device != conv_states.device
        or step_indices.device != conv_states.device
    ):
        raise ValueError("conv_states and index tensors must be on the same device")
    for name, tensor in (
        ("state_indices", state_indices),
        ("step_indices", step_indices),
    ):
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must use int32 or int64, got {tensor.dtype}")
    if not isinstance(draft_token_num, int) or draft_token_num <= 0:
        raise ValueError("draft_token_num must be a positive integer")

    num_requests = state_indices.shape[0]
    num_layers = conv_states.shape[0]
    num_slots = conv_states.shape[1]
    conv_window_size = conv_states.shape[2]
    num_dims = conv_states.shape[3]
    if min(num_layers, num_slots, conv_window_size, num_dims) <= 0:
        raise ValueError("conv_states dimensions must all be positive")
    if draft_token_num > conv_window_size + 1:
        raise ValueError(
            "draft_token_num cannot exceed conv_window_size + 1, "
            f"got {draft_token_num} and {conv_window_size}"
        )
    if num_requests == 0:
        return conv_states

    # Get strides (in elements, not bytes)
    layer_stride = conv_states.stride(0)
    req_stride = conv_states.stride(1)
    window_stride = conv_states.stride(2)
    dim_stride = conv_states.stride(3)
    if min(layer_stride, req_stride, window_stride, dim_stride) <= 0:
        raise ValueError("conv_states must have positive strides")

    # Keep the caller's original view and update its backing state pool in
    # place. The runtime ignores this helper's return value.
    state_indices = state_indices.contiguous()
    step_indices = step_indices.contiguous()

    # Grid over all requests
    grid = (num_requests,)

    _conv_state_rollback_kernel[grid](
        conv_states_ptr=conv_states,
        state_indices_ptr=state_indices,
        step_indices_ptr=step_indices,
        draft_token_num=draft_token_num,
        num_slots=num_slots,
        num_layers=num_layers,
        num_dims=num_dims,
        conv_window_size=conv_window_size,
        layer_stride=layer_stride,
        req_stride=req_stride,
        window_stride=window_stride,
        dim_stride=dim_stride,
        BLOCK_DIM=triton.next_power_of_2(num_dims),
    )

    return conv_states
