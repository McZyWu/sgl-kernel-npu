# A5 C8 Lightning Indexer + cache management

The normal `sgl-kernel-npu` Ascend950 wheel registers
`torch.ops.npu.fused_quant_lightning_indexer_manage`. The implementation is
ported from [vllm-ascend #18383](https://github.com/vllm-project/vllm-ascend/pull/18383),
commit `7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e`.
`UPSTREAM_PROVENANCE.json` records the original payload hashes. Source licenses
and attribution are retained; the CANN and Apache license texts are also present
under `../fused_lightning_indexer_manage/licenses/`.

This is an A5-only operator. The existing non-quantized A2/A3 LIM entry is separate.
Build with `bash build.sh -a kernels Ascend950` from the repository root.

The 19 Tensor arguments follow the donor interface: BF16 index weights,
FP32 query scales, FP8 E4M3FN query `[T,32|64,128]`, FP32 index-key scales,
FP8 paged index keys, index block table, cumulative query ends, final KV lengths,
offload-prefix lengths, persistent capacities, request states, request pool rows,
the mutable source-to-slot pool, TopK source IDs, TopK slots, per-query miss
counts, request miss sources, request miss slots, and request miss counts.
Outputs and pool are caller-owned and updated in place. Meta performs the same
static validation without submitting a kernel.

For each request, let `S` be its final KV length, `P` its offloaded prefix,
and `C` its persistent sparse capacity. TopK searches the entire causal-visible
range. Selected tail sources use `C + (source_id - P)`, are always hits, and do
not enter the persistent pool or miss-copy plan. The caller maintains at least
`ceil((S-P)/128)*128` additional packed KV slots. This does not append all tail
tokens to TopK. These slots are relative to P, including on odd page boundaries.

The native contract supports `K=2048`, page size 128, 1–14 query routes,
`C<=floor((32766-ceil((S-P)/128)*128)/128)*128`, and source capacity at most
`2**21`. Prefix and capacity are page
aligned. Offloaded requests require `P>=2048`, `C<=P`,
`P<=S-(Q-1)`. If `P<=Q*2048`, `C=P`; otherwise `C>=Q*2048`.
Request states are `-3` dense, `-2` first offload, and `-1` steady offload.

The direct-launch host serializes the donor's tiling and actual weights/key/scale
strides. It retains Tensor/workspace owners across ordered torch_npu submission.
The device entry loads GM tiling fields explicitly, provides the missing event
constant, and excludes A5 payloads from the host-only parsing pass.

The normal direct-launch build uses the donor's vector management path
(`LIM_SIMT_POSTPROCESS=0`, `LIM_SIMT_VICTIM_UNPACK=0`). The donor's SIMT sources
remain available, but enabling them also requires its 216 KiB user-UB launch
reservation. No SIMT performance parity is claimed by this port.

Run the donor's A5 contract/oracle tests with:

```bash
python3 -m pytest -q tests/test_fused_quant_lightning_indexer_manage_a5.py
```

Run `tests/test_lim_copy_sfa_c8.py` for the paired registered-DRAM Copy-SFA path.
The SGLang integration initially enables Q1 only; native MTP contracts do not
enable framework speculation or rollback handling.

The direct-launch entry explicitly flushes each AIV's DCache after management
publishes its compact miss counts and invalid rows. GlobalTensor::SetValue
updates DCache; a core barrier alone does not make those stores visible in GM.
Homogeneous dense counts already use DMA, which is why that path is unaffected.
