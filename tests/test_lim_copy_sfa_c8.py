"""A5 quant LIM -> C8 Copy-SFA, registered DRAM, dense/tail/padding and replay."""

import math

import pytest
import torch
import torch_npu  # noqa: F401
from fused_quant_scatter_copy_sparse_flash_attention_golden import (
    make_packed_bytes,
    unpack_rows,
)


@pytest.mark.parametrize("graph_replay", [False, True])
@pytest.mark.parametrize("capacity", [2048, 4096, 8192])
def test_c8_lim_copy_sfa_lifecycle(graph_replay, capacity):
    import acl
    import sgl_kernel_npu  # noqa: F401
    from sgl_kernel_npu.sparsity_driven_kv_offload import create_shm_tensor, free_shm

    if acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("C8 numerical integration requires Ascend950 (A5)")
    torch.npu.set_device(0)
    device = "npu:0"
    batch, rows, top_k = 4, 4, 2048
    context, physical = 2 * capacity + 512, capacity + 256
    index_heads, query_heads = 32, 8
    generator = torch.Generator().manual_seed(20261023)
    host, _, pointer = create_shm_tensor(
        (rows * context // 128, 128, 1, 656), torch.int8
    )
    try:
        host.copy_(make_packed_bytes(rows * context // 128, generator))
        logical = host.view(rows, context, 656)
        hbm = torch.full(
            (rows * physical // 128, 128, 1, 656), 19, dtype=torch.int8, device=device
        )
        index_key = torch.zeros(
            rows, context, 128, dtype=torch.float8_e4m3fn, device=device
        )
        query = torch.zeros(
            batch, index_heads, 128, dtype=torch.float8_e4m3fn, device=device
        )
        # Write via BF16 before casting: FP8 indexed assignment is not generally supported.
        q_cpu = torch.zeros(batch, index_heads, 128)
        q_cpu[..., 0] = 1
        query.copy_(q_cpu.to(torch.float8_e4m3fn))
        weights = torch.full(
            (batch, index_heads), 1 / index_heads, dtype=torch.bfloat16, device=device
        )
        q_scale = torch.ones(batch, index_heads, device=device)
        k_scale = torch.ones(rows * context // 128, 128, 1, device=device)
        requests = torch.tensor([1, rows, 2, 3], dtype=torch.int32, device=device)
        key_table = torch.stack(
            [
                torch.arange(context // 128, dtype=torch.int32, device=device)
                + request * (context // 128)
                if request < rows
                else torch.zeros(context // 128, dtype=torch.int32, device=device)
                for request in (1, rows, 2, 3)
            ]
        )
        hbm_table = torch.stack(
            [
                torch.arange(physical // 128, dtype=torch.int32, device=device)
                + request * (physical // 128)
                if request < rows
                else torch.zeros(physical // 128, dtype=torch.int32, device=device)
                for request in (1, rows, 2, 3)
            ]
        )
        ends = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
        lengths = torch.empty(batch, dtype=torch.int32, device=device)
        prefix = torch.empty_like(lengths)
        caches = torch.empty_like(lengths)
        states = torch.empty_like(lengths)
        hbm_lengths = torch.empty_like(lengths)
        pool = torch.full((rows, context), -1, dtype=torch.int32, device=device)
        ids = torch.full((batch, 1, top_k), -1, dtype=torch.int32, device=device)
        slots = torch.full_like(ids, -1)
        topk_counts = torch.zeros_like(lengths)
        miss_src = torch.full((batch, 32768), -1, dtype=torch.int32, device=device)
        miss_dst = torch.full_like(miss_src, -1)
        miss_counts = torch.zeros_like(lengths)
        attention_query = torch.randn(batch, query_heads, 576, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        output = torch.empty(
            batch, query_heads, 512, dtype=torch.bfloat16, device=device
        )
        scale = 1 / math.sqrt(576)

        def enqueue():
            torch.ops.npu.fused_quant_lightning_indexer_manage(
                weights,
                q_scale,
                query,
                k_scale,
                index_key.view(-1, 128, 1, 128),
                key_table,
                ends,
                lengths,
                prefix,
                caches,
                states,
                requests,
                pool,
                ids,
                slots,
                topk_counts,
                miss_src,
                miss_dst,
                miss_counts,
            )
            valid = lengths > 0
            slots.masked_fill_(~valid[:, None, None], -1)
            ids.masked_fill_(~valid[:, None, None], -1)
            topk_counts.masked_fill_(~valid, 0)
            miss_counts.masked_fill_(~valid, 0)
            torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention(
                attention_query,
                hbm,
                host,
                pointer,
                hbm_table,
                key_table,
                slots,
                ids,
                topk_counts,
                ends,
                lengths,
                hbm_lengths,
                miss_src.unsqueeze(1),
                miss_dst.unsqueeze(1),
                miss_counts,
                scale,
                output,
            )
            output.masked_fill_(~valid[:, None, None], 0)

        def prepare(a, b, c, reset=False):
            values = [a, 0, b, c]
            p = [((value - 1) // 128) * 128 if value else 0 for value in values]
            sparse = [value >= capacity for value in p]
            lengths.copy_(torch.tensor(values, dtype=torch.int32, device=device))
            prefix.copy_(torch.tensor(p, dtype=torch.int32, device=device))
            caches.copy_(
                torch.tensor(
                    [capacity if active else 0 for active in sparse],
                    dtype=torch.int32,
                    device=device,
                )
            )
            hbm_lengths.copy_(
                torch.tensor(
                    [physical if active else 0 for active in sparse],
                    dtype=torch.int32,
                    device=device,
                )
            )
            states.copy_(
                torch.tensor(
                    [
                        -2 if reset and active else -1 if active else -3
                        for active in sparse
                    ],
                    dtype=torch.int32,
                    device=device,
                )
            )
            key = torch.zeros(rows, context, 128)
            key[..., 0] = -8
            for request, length in ((1, a), (2, b), (3, c)):
                key[request, max(0, length - top_k) : length, 0] = 8
                # Framework writes dense identity KV and prefix-relative tail rows.
                if length <= capacity:
                    hbm.view(rows, physical, 656)[request, :length].copy_(
                        logical[request, :length].to(device)
                    )
                tail_prefix = ((length - 1) // 128) * 128
                source = torch.arange(tail_prefix, length, dtype=torch.int64)
                destination = capacity + source - tail_prefix
                hbm.view(rows, physical, 656)[request].index_copy_(
                    0, destination.to(device), logical[request, source].to(device)
                )
            index_key.copy_(key.to(torch.float8_e4m3fn).to(device))

        prepare(capacity + 2, 2 * capacity + 2, 130, reset=True)
        enqueue()
        torch.npu.synchronize()
        graph = None
        if graph_replay:
            states.copy_(
                torch.tensor([-1, -3, -1, -3], dtype=torch.int32, device=device)
            )
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                enqueue()
        for a, b, c, reset in (
            (capacity - 1, capacity, 130, True),
            (capacity + 2, 2 * capacity + 2, 130, True),
            (capacity + 3, 2 * capacity + 3, 130, False),
            (capacity + 129, 2 * capacity + 129, capacity + 1, True),
            (capacity + 266, capacity + 953, 130, True),
        ):
            prepare(a, b, c, reset)
            graph.replay() if graph is not None else enqueue()
            torch.npu.synchronize()
            ids_cpu, slots_cpu = ids.cpu(), slots.cpu()
            for row, request, length in ((0, 1, a), (2, 2, b), (3, 3, c)):
                count = min(length, top_k)
                actual_ids = ids_cpu[row, 0, :count].long()
                assert set(actual_ids.tolist()) == set(
                    range(max(0, length - top_k), length)
                )
                actual_slots = slots_cpu[row, 0, :count].long()
                torch.testing.assert_close(
                    hbm.cpu().view(rows, physical, 656)[request, actual_slots],
                    logical[request, actual_ids],
                    rtol=0,
                    atol=0,
                )
                if reset and ((length - 1) // 128) * 128 >= capacity:
                    # A C>K cold fill must publish the entire resident cache,
                    # while attention still consumes exactly K selected rows.
                    assert miss_counts[row].item() == capacity
                    sources = miss_src[row, :capacity].cpu().long()
                    destinations = miss_dst[row, :capacity].cpu().long()
                    torch.testing.assert_close(
                        hbm.cpu().view(rows, physical, 656)[request, destinations],
                        logical[request, sources],
                        rtol=0,
                        atol=0,
                    )
                key, value = unpack_rows(logical[request, actual_ids])
                scores = attention_query[row].cpu().float() @ key.T * scale
                expected = scores.softmax(-1) @ value
                torch.testing.assert_close(
                    output[row].cpu().float(), expected, rtol=0.05, atol=0.008
                )
                if length > capacity:
                    selected_tail = actual_ids >= ((length - 1) // 128) * 128
                    assert torch.equal(
                        actual_slots[selected_tail],
                        capacity
                        + actual_ids[selected_tail]
                        - ((length - 1) // 128) * 128,
                    )
            assert torch.count_nonzero(output[1]) == 0
            assert miss_counts[1] == 0
    finally:
        torch.npu.synchronize()
        free_shm()
