// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Apache-2.0. A5-only normal-wheel registration; no vLLM extension dependency.
#include "torch_helper.h"
#include "a5_c8_ops.h"
#ifdef SGL_KERNEL_A5_C8_ORIGINAL
namespace sglang::npu_kernel {
c10::Dict<std::string, std::string> resolve_a5_c8_original_libraries();
}
#endif
namespace {
std::string a5_c8_backend()
{
#ifdef SGL_KERNEL_A5_C8_ORIGINAL
    return "original";
#else
    return "direct";
#endif
}
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("a5_c8_backend() -> str", &a5_c8_backend);
#ifdef SGL_KERNEL_A5_C8_ORIGINAL
    m.def("resolve_a5_c8_original_libraries() -> Dict(str, str)",
          &sglang::npu_kernel::resolve_a5_c8_original_libraries);
#endif
    m.def(
        "fused_quant_lightning_indexer_manage(Tensor index_weights, Tensor query_dequant_scale, Tensor query, "
        "Tensor index_key_dequant_scale, Tensor index_key_cache, Tensor index_block_table, "
        "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_key, Tensor offload_seq_lengths_key, "
        "Tensor num_cache_tokens, Tensor request_state, Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
        "Tensor(b!) topk_src_ids, Tensor(c!) topk_dst_slots, Tensor(d!) topk_miss_counts, "
        "Tensor(e!) miss_src_ids, Tensor(f!) miss_dst_slots, Tensor(g!) miss_counts) -> ()");

    m.def(
        "fused_quant_scatter_copy_sparse_flash_attention(Tensor query, Tensor(a!) hbm_kv_bytes, "
        "Tensor dram_kv_bytes, int dram_kv_dev_ptr, Tensor hbm_block_table, Tensor dram_block_table, "
        "Tensor topk_dst_slots, Tensor topk_src_ids, Tensor topk_miss_counts, Tensor actual_seq_lengths_query, "
        "Tensor actual_seq_lengths_kv, Tensor num_cache_tokens, Tensor miss_src_ids, Tensor miss_dst_slots, "
        "Tensor miss_counts, float scale_value, Tensor(b!) attention_out) -> ()");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("fused_quant_lightning_indexer_manage", TORCH_FN(sglang::npu_kernel::fused_quant_lightning_indexer_manage));
    m.impl("fused_quant_scatter_copy_sparse_flash_attention",
           TORCH_FN(sglang::npu_kernel::fused_quant_scatter_copy_sparse_flash_attention));
}
TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("fused_quant_lightning_indexer_manage", TORCH_FN(sglang::npu_kernel::fused_quant_lightning_indexer_manage));
    m.impl("fused_quant_scatter_copy_sparse_flash_attention",
           TORCH_FN(sglang::npu_kernel::fused_quant_scatter_copy_sparse_flash_attention));
}
}  // namespace
