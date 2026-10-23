# A5 C8 scatter copy + sparse flash attention

`torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention` is an A5-only
normal-wheel operator ported from [vllm-ascend #18490](https://github.com/vllm-project/vllm-ascend/pull/18490),
commit `1b180deb840484c2fe33cbdd7abb68a615085172`. Donor source hashes and retained
licenses are recorded alongside the imported payloads. Build the Ascend950 wheel
with `bash build.sh -a kernels Ascend950`.

The API takes:

```text
query, hbm_kv_bytes, dram_kv_bytes, dram_kv_dev_ptr,
hbm_block_table, dram_block_table, topk_dst_slots, topk_src_ids,
topk_miss_counts, actual_seq_lengths_query, actual_seq_lengths_kv,
num_cache_tokens, miss_src_ids, miss_dst_slots, miss_counts,
scale_value, attention_out
```

Query is BF16/FP16 `[T,H,576]`; output is `[T,H,512]` with the same dtype.
Supported head counts are 8, 16, 32, 64, and 128. Both KV tensors are opaque INT8
byte views `[pages,128,1,656]`: 512 FP8 NoPE bytes, 128 BF16 RoPE bytes, then
16 FP32 scale bytes. Copy preserves all 656 bytes.

The direct-launch adaptation accepts the owning CPU Tensor returned by
`create_shm_tensor` plus its registered device-visible address. HBM, query,
metadata and output remain on one NPU. This preserves the existing SGLang DRAM
allocator and avoids an HBM copy masquerading as offload. The caller must retain
registered storage throughout capture/replay and synchronize before `free_shm`.
The host also retains Tensor/workspace owners while a launch is queued.

`num_cache_tokens=0` means dense, using the real sequence length. Otherwise it is
the exposed HBM logical address-space length. When consuming quant LIM's tail
slots, pass `C+256` here and `C` to LIM; the HBM block table must cover `C+256`.
`actual_seq_lengths_kv` remains the real `S`. The operator attends the TopK slot
list; it does not append tail tokens.

TopK arrays are `[T,1,2048]`; miss counts are `[T]`. Request miss arrays are
`[B,1,32768]` with counts `[B]`. LIM's contiguous `[B,32768]` outputs can be
unsqueezed without copying. Metadata drives the fused miss path for at most 512
misses, or a complete union copy followed by attention inside the same kernel.
The interface permits 1–16 queries per request; the paired LIM limit is 14 and
SGLang initially enables Q1.

HBM/DRAM tables have at most 2048 pages per request, so the effective source
capacity of the paired interface is at most 262144 tokens at page size 128.
Invalid rows must have zero miss counts and `-1` slots; callers zero their outputs
because an empty row may leave output unwritten. Query/output storage stays
BF16/FP16 even though KV storage is C8.

```bash
python3 -m pytest -q \
  tests/test_fused_quant_scatter_copy_sparse_flash_attention.py \
  tests/test_lim_copy_sfa_c8.py
```

The standalone tests preserve the donor's independent CPU attention golden and
bitwise payload checks. The paired tests add full-visible tail selection,
dense/sparse/padding rows, request reset, and graph replay. Numerical execution
requires A5; an A3 cross-build does not establish A5 accuracy or performance.
