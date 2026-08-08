import torch
from sgl_kernel_npu.mamba.speculative_state_scatter import (
    _speculative_state_scatter_kernel,
    speculative_state_scatter_npu,
)


@torch.no_grad()
def test_speculative_state_scatter_dynamic_request_counts():
    layers, slots, scratch, steps, channels, window = 2, 12, 8, 4, 33, 5

    for num_requests in (1, 2, 4, 7):
        dst = torch.randn(
            layers, slots, channels, window, device="npu", dtype=torch.bfloat16
        )
        src = torch.randn(
            layers,
            scratch,
            steps,
            channels,
            window,
            device="npu",
            dtype=torch.bfloat16,
        )
        expected = dst.clone()
        dst_indices = torch.arange(num_requests, device="npu", dtype=torch.int32) + 2
        src_indices = torch.arange(num_requests, device="npu", dtype=torch.int32)
        step_indices = torch.arange(num_requests, device="npu", dtype=torch.int32)
        step_indices %= steps
        if num_requests > 1:
            step_indices[-1] = -1

        valid = step_indices >= 0
        expected[:, dst_indices[valid].long()] = src[
            :, src_indices[valid].long(), step_indices[valid].long()
        ]

        actual = speculative_state_scatter_npu(
            dst, src, dst_indices, src_indices, step_indices
        )
        assert torch.equal(actual, expected)


def test_speculative_state_scatter_disables_grid_specialization():
    assert "logical_grid" in _speculative_state_scatter_kernel.do_not_specialize
