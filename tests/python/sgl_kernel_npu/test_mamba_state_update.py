# -*- coding: utf-8 -*-
import random

import pytest
import torch
from sgl_kernel_npu.mamba.mamba_state_update_triton import (
    conv_state_rollback,
    move_intermediate_cache,
)

device = "npu"


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    else:
        assert error_rate < ratio, msg


def conv_state_rollback_ref(
    conv_states, valid_state_indices, last_steps, draft_token_num
):
    cpu_valid_indices = valid_state_indices.cpu().numpy()
    cpu_last_steps = last_steps.cpu().numpy()

    for idx, step in zip(cpu_valid_indices, cpu_last_steps):
        if idx < 0 or idx >= conv_states.shape[1]:
            continue
        if step < 0 or step >= draft_token_num:
            continue
        # Calculate rollback steps
        shift = (draft_token_num - 1) - step
        if shift > 0:
            # Select states for this request across all layers and dims
            req_conv_state = conv_states[:, idx, :, :]

            # Perform right shift (Rollback)
            req_conv_state[:, shift:, :].copy_(req_conv_state[:, :-shift, :].clone())

    return conv_states


@pytest.mark.parametrize(
    ("num_layers", "pool_size", "num_dims", "draft_token_num", "num_requests", "dtype"),
    [
        pytest.param(
            *test,
            id="layers{0}_pool{1}_dims{2}_draft{3}_req{4}_{5}".format(*test),
        )
        for test in [
            (32, 32, 2048, 3, 3, torch.bfloat16),
            (16, 16, 1024, 3, 2, torch.bfloat16),
            (32, 32, 2048, 4, 4, torch.bfloat16),
            (16, 16, 1024, 4, 1, torch.bfloat16),
        ]
    ],
)
@torch.no_grad
def test_conv_state_rollback(
    num_layers: int,
    pool_size: int,
    num_dims: int,
    draft_token_num: int,
    num_requests: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    conv_window_size = 3 + draft_token_num - 1

    conv_states = torch.randn(
        num_layers, pool_size, conv_window_size, num_dims, device=device, dtype=dtype
    )

    valid_state_indices = torch.randint(
        0, pool_size, (num_requests,), device=device, dtype=torch.int32
    )
    last_steps = torch.randint(
        -1, draft_token_num, (num_requests,), device=device, dtype=torch.int32
    )
    original_states = conv_states.clone()

    gt_states = conv_state_rollback_ref(
        original_states, valid_state_indices, last_steps, draft_token_num
    )

    result_states = conv_state_rollback(
        conv_states,
        valid_state_indices,
        last_steps,
        draft_token_num,
    )
    assert_close("conv_state", gt_states, result_states, 1e-3)


@torch.no_grad
def test_conv_state_rollback_updates_noncontiguous_view_in_place():
    """Rollback honors real strides and masks a non-power-of-two dim."""
    L, S, D, W, N = 2, 5, 4, 6, 65
    storage = torch.arange(
        L * S * N * W,
        device=device,
        dtype=torch.int64,
    ).reshape(L, S, N, W)
    storage = (storage % 2048).to(torch.bfloat16)
    conv_states = storage.transpose(-1, -2)
    assert conv_states.shape == (L, S, W, N)
    assert not conv_states.is_contiguous()

    expected_storage = storage.clone()
    expected = expected_storage.transpose(-1, -2)
    state_indices = torch.tensor([0, 2, 4], device=device, dtype=torch.int64)
    step_indices = torch.tensor([0, 2, 3], device=device, dtype=torch.int64)
    conv_state_rollback_ref(expected, state_indices, step_indices, D)

    result = conv_state_rollback(conv_states, state_indices, step_indices, D)

    assert result.data_ptr() == conv_states.data_ptr()
    assert result.stride() == conv_states.stride()
    assert torch.equal(storage, expected_storage)


@torch.no_grad
def test_conv_state_rollback_ignores_out_of_range_metadata():
    """Invalid request and step metadata cannot access outside the state pool."""
    L, S, D, W, N = 1, 3, 3, 5, 17
    conv_states = torch.arange(
        L * S * W * N,
        device=device,
        dtype=torch.int64,
    ).reshape(L, S, W, N)
    conv_states = (conv_states % 2048).to(torch.bfloat16)
    expected = conv_states.clone()
    state_indices = torch.tensor([1, -1, S, 0, 2], device=device, dtype=torch.int32)
    step_indices = torch.tensor([0, 0, 0, -1, D], device=device, dtype=torch.int32)
    conv_state_rollback_ref(expected, state_indices, step_indices, D)

    conv_state_rollback(conv_states, state_indices, step_indices, D)

    assert torch.equal(conv_states, expected)


@pytest.mark.parametrize(
    ("L", "S", "D", "H", "V", "K", "num_valid", "dtype"),
    [
        pytest.param(
            *test,
            id="L{0}_S{1}_D{2}_H{3}_V{4}_K{5}_valid{6}_{7}".format(*test),
        )
        for test in [
            (36, 229, 4, 8, 128, 128, 180, torch.bfloat16),
            (18, 100, 4, 4, 64, 64, 50, torch.bfloat16),
            (36, 229, 4, 8, 128, 128, 229, torch.bfloat16),
            (18, 100, 4, 4, 64, 64, 100, torch.bfloat16),
        ]
    ],
)
@torch.no_grad
def test_move_intermediate_cache(
    L: int,
    S: int,
    D: int,
    H: int,
    V: int,
    K: int,
    num_valid: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    dst_cache = torch.randn(L, S, H, V, K, device=device, dtype=dtype)
    dst_cache_clone = dst_cache.clone()
    src_cache = torch.randn(L, S, D, H, V, K, device=device, dtype=dtype)

    population = range(S)
    valid_indices = random.sample(population, num_valid)
    last_step_pos = [random.randint(0, D - 1) for _ in range(num_valid)]

    valid_tensor = torch.tensor(valid_indices, device=device, dtype=torch.int32)
    last_steps_tensor = torch.tensor(last_step_pos, device=device, dtype=torch.int32)
    valid_mask = last_steps_tensor >= 0
    valid_state_indices = valid_tensor[valid_mask].to(torch.int64)
    valid_last_steps = last_steps_tensor[valid_mask].to(torch.int64)
    dst_cache[:, valid_state_indices, :] = src_cache[
        :, valid_state_indices, valid_last_steps
    ]

    move_intermediate_cache(dst_cache_clone, src_cache, valid_tensor, last_steps_tensor)

    assert_close("move_cache", dst_cache, dst_cache_clone, 1e-3)


@pytest.mark.parametrize("V", [65, 128, 129])
@torch.no_grad
def test_move_intermediate_cache_copies_every_v_tile(V: int):
    """Every V tile must be committed, including values beyond BLOCK_V=64."""
    L, S, D, H, K = 2, 5, 4, 2, 16
    src_cache = torch.arange(
        L * S * D * H * V * K,
        device=device,
        dtype=torch.int64,
    ).reshape(L, S, D, H, V, K)
    src_cache = (src_cache % 2048).to(torch.bfloat16)
    dst_cache = torch.full(
        (L, S, H, V, K),
        -1,
        device=device,
        dtype=torch.bfloat16,
    )
    expected = dst_cache.clone()

    valid_tensor = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)
    last_steps = torch.tensor([0, D - 1, 1], device=device, dtype=torch.int32)
    expected[:, valid_tensor.long()] = src_cache[
        :, valid_tensor.long(), last_steps.long()
    ]

    move_intermediate_cache(dst_cache, src_cache, valid_tensor, last_steps)

    assert torch.equal(dst_cache, expected)
    assert torch.equal(
        dst_cache[:, valid_tensor.long(), :, 64:],
        expected[:, valid_tensor.long(), :, 64:],
    )


@torch.no_grad
def test_move_intermediate_cache_preserves_destination_physical_layout():
    """The GDN mover keeps the dense physical slot ABI of a transposed view."""
    L, S, D, H, V, K = 2, 4, 3, 2, 128, 16
    src_cache = torch.randn(L, S, D, H, V, K, device=device, dtype=torch.bfloat16)
    dst_storage = torch.full(
        (L, S, H, K, V),
        -7,
        device=device,
        dtype=torch.bfloat16,
    )
    dst_cache = dst_storage.transpose(-1, -2)
    assert not dst_cache.is_contiguous()
    expected_storage = dst_storage.clone()

    valid_tensor = torch.tensor([0, 3, 2], device=device, dtype=torch.int32)
    last_steps = torch.tensor([2, -1, 0], device=device, dtype=torch.int32)
    selected = src_cache[:, valid_tensor[[0, 2]].long(), last_steps[[0, 2]].long()]
    expected_storage[:, valid_tensor[[0, 2]].long()] = selected.reshape(L, 2, H, K, V)

    move_intermediate_cache(dst_cache, src_cache, valid_tensor, last_steps)

    assert torch.equal(dst_storage, expected_storage)
    assert torch.all(dst_storage[:, valid_tensor[1].long()] == -7)
    assert not torch.equal(dst_cache[:, valid_tensor[[0, 2]].long()], selected)


@torch.no_grad
def test_move_intermediate_cache_ignores_out_of_range_metadata():
    """Invalid metadata must not read or write outside either cache."""
    L, S, D, H, V, K = 1, 3, 2, 1, 65, 8
    src_cache = torch.arange(
        L * S * D * H * V * K,
        device=device,
        dtype=torch.int64,
    ).reshape(L, S, D, H, V, K)
    src_cache = (src_cache % 2048).to(torch.bfloat16)
    dst_cache = torch.full((L, S + 1, H, V, K), -3, device=device, dtype=torch.bfloat16)
    expected = dst_cache.clone()

    valid_tensor = torch.tensor([1, -1, S, 0, 2], device=device, dtype=torch.int32)
    last_steps = torch.tensor([1, 0, 0, -1, D], device=device, dtype=torch.int32)
    expected[:, 1] = src_cache[:, 1, 1]

    move_intermediate_cache(dst_cache, src_cache, valid_tensor, last_steps)

    assert torch.equal(dst_cache, expected)


@torch.no_grad
def test_move_intermediate_cache_rejects_non_dense_source_inner_layout():
    """The custom recurrent op contract requires dense source H/V/K storage."""
    src_cache = torch.zeros(
        (1, 2, 2, 1, 8, 16), device=device, dtype=torch.bfloat16
    ).transpose(-1, -2)
    dst_cache = torch.zeros((1, 2, 1, 16, 8), device=device, dtype=torch.bfloat16)
    valid_tensor = torch.tensor([0], device=device, dtype=torch.int32)
    last_steps = torch.tensor([0], device=device, dtype=torch.int32)

    with pytest.raises(ValueError, match="H/V/K dimensions must be contiguous"):
        move_intermediate_cache(dst_cache, src_cache, valid_tensor, last_steps)
