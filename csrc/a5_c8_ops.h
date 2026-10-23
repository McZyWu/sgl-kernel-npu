#pragma once
#include <ATen/ATen.h>
namespace sglang::npu_kernel {
void fused_quant_lightning_indexer_manage(
    const at::Tensor &index_weights, const at::Tensor &query_dequant_scale, const at::Tensor &query,
    const at::Tensor &index_key_dequant_scale, const at::Tensor &index_key_cache, const at::Tensor &index_block_table,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
    const at::Tensor &offload_seq_lengths_key, const at::Tensor &num_cache_tokens, const at::Tensor &request_state,
    const at::Tensor &req_pool_entries, const at::Tensor &cache_slots_pool, const at::Tensor &topk_src_ids,
    const at::Tensor &topk_dst_slots, const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids,
    const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts);
void fused_quant_scatter_copy_sparse_flash_attention(
    const at::Tensor &query, at::Tensor hbm_kv_bytes, const at::Tensor &dram_kv_bytes, int64_t dram_kv_dev_ptr,
    const at::Tensor &hbm_block_table, const at::Tensor &dram_block_table, const at::Tensor &topk_dst_slots,
    const at::Tensor &topk_src_ids, const at::Tensor &topk_miss_counts, const at::Tensor &actual_seq_lengths_query,
    const at::Tensor &actual_seq_lengths_kv, const at::Tensor &num_cache_tokens, const at::Tensor &miss_src_ids,
    const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts, double scale_value, at::Tensor attention_out);
}  // namespace sglang::npu_kernel
