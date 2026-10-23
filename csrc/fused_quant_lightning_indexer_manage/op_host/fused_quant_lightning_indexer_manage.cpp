// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Apache-2.0. Validation/tiling adapted from vllm-ascend PR #18383.
#include <algorithm>
#include <limits>
#include "defines.h"
#include "ge_helper.h"
#include "owned_kernel_launch.h"
#include "aclrtlaunch_fused_quant_lightning_indexer_manage.h"
#include "../op_kernel/arch35/fused_quant_lightning_indexer_manage_tiling.h"
#include "../op_kernel/arch35/fused_quant_lightning_indexer_manage_workspace.h"
#include "../op_kernel/arch35/lim_payload_layout.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
namespace sglang::npu_kernel {
inline void check_fused_quant_lightning_indexer_manage_arch35(
    const at::Tensor &index_weights, const at::Tensor &query_dequant_scale, const at::Tensor &query,
    const at::Tensor &index_key_dequant_scale, const at::Tensor &index_key_cache, const at::Tensor &index_block_table,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
    const at::Tensor &offload_seq_lengths_key, const at::Tensor &num_cache_tokens, const at::Tensor &request_state,
    const at::Tensor &req_pool_entries, const at::Tensor &cache_slots_pool, const at::Tensor &topk_src_ids,
    const at::Tensor &topk_dst_slots, const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids,
    const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts)
{
    constexpr int64_t kTopK = lim_request_limits::TOPK;
    constexpr int64_t kMissCapacity = lim_request_limits::MISS_CAPACITY;
    constexpr int64_t kBlock = lim_request_limits::BLOCK_SIZE;
    constexpr int64_t kMaxSourceCapacity = lim_payload_layout::MAX_SOURCE_CAPACITY;
    TORCH_CHECK(query.dim() == 3 && query.size(0) > 0 && (query.size(1) == 32 || query.size(1) == 64) &&
                    query.size(2) == 128 && query.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "query must be float8_e4m3fn [T,32|64,128].");
    const int64_t total_queries = query.size(0);
    const int64_t heads = query.size(1);
    TORCH_CHECK(index_weights.dim() == 2 && index_weights.size(0) == total_queries && index_weights.size(1) == heads &&
                    index_weights.scalar_type() == at::kBFloat16 && index_weights.stride(1) == 1 &&
                    index_weights.stride(0) >= heads,
                "index_weights must be BF16 [T,H] with only T-axis stride.");
    TORCH_CHECK(query_dequant_scale.dim() == 2 && query_dequant_scale.size(0) == total_queries &&
                    query_dequant_scale.size(1) == heads && query_dequant_scale.scalar_type() == at::kFloat,
                "query_dequant_scale must be FP32 [T,H].");
    TORCH_CHECK(index_key_cache.dim() == 4 && index_key_cache.size(0) > 0 && index_key_cache.size(1) == kBlock &&
                    index_key_cache.size(2) == 1 && index_key_cache.size(3) == 128 &&
                    index_key_cache.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "index_key_cache must be float8_e4m3fn [blocks,128,1,128].");
    TORCH_CHECK(index_key_dequant_scale.dim() == 3 && index_key_dequant_scale.size(0) == index_key_cache.size(0) &&
                    index_key_dequant_scale.size(1) == kBlock && index_key_dequant_scale.size(2) == 1 &&
                    index_key_dequant_scale.scalar_type() == at::kFloat,
                "index_key_dequant_scale must be FP32 [blocks,128,1].");
    TORCH_CHECK(index_key_cache.stride(1) == 128 && index_key_cache.stride(2) == 128 &&
                    index_key_cache.stride(3) == 1 && index_key_dequant_scale.stride(1) == 1 &&
                    index_key_dequant_scale.stride(2) == 1,
                "key tensors may only be strided on the block axis.");
    TORCH_CHECK(index_key_cache.stride(0) >= kBlock * 128 && index_key_dequant_scale.stride(0) >= kBlock,
                "key and key-scale blocks must not overlap.");
    for (auto stride : {index_weights.stride(0), index_key_cache.stride(0), index_key_dequant_scale.stride(0)})
        TORCH_CHECK(stride <= UINT32_MAX, "quant LIM stride exceeds uint32");
    TORCH_CHECK(index_block_table.dim() == 2 && index_block_table.size(0) > 0 && index_block_table.size(1) > 0 &&
                    index_block_table.size(1) * kBlock <= kMaxSourceCapacity,
                "index_block_table must be non-empty and cover at most 2^21 tokens.");
    const int64_t batch = index_block_table.size(0);
    const int64_t source_capacity = index_block_table.size(1) * kBlock;
    auto batch_vector = [batch](const at::Tensor &tensor, const char *name) {
        TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == batch, name, " must be [B].");
    };
    batch_vector(actual_seq_lengths_query, "actual_seq_lengths_query");
    batch_vector(actual_seq_lengths_key, "actual_seq_lengths_key");
    batch_vector(offload_seq_lengths_key, "offload_seq_lengths_key");
    batch_vector(num_cache_tokens, "num_cache_tokens");
    batch_vector(request_state, "request_state");
    batch_vector(req_pool_entries, "req_pool_entries");
    TORCH_CHECK(total_queries >= batch && total_queries <= lim_request_limits::MAX_ROUTES * batch,
                "the operator supports 1..14 query rows per request.");
    TORCH_CHECK(
        cache_slots_pool.dim() == 2 && cache_slots_pool.size(0) > 0 && cache_slots_pool.size(1) == source_capacity,
        "cache_slots_pool must be [POOL_SIZE,index_block_table.size(1)*128].");
    TORCH_CHECK(topk_src_ids.dim() == 3 && topk_src_ids.size(0) == total_queries && topk_src_ids.size(1) == 1 &&
                    topk_src_ids.size(2) == kTopK && topk_dst_slots.sizes() == topk_src_ids.sizes(),
                "topk outputs must be [T,1,2048].");
    TORCH_CHECK(topk_miss_counts.dim() == 1 && topk_miss_counts.size(0) == total_queries,
                "topk_miss_counts must be [T].");
    TORCH_CHECK(miss_src_ids.dim() == 2 && miss_src_ids.size(0) == batch && miss_src_ids.size(1) == kMissCapacity &&
                    miss_dst_slots.sizes() == miss_src_ids.sizes(),
                "miss outputs must be [B,32768].");
    batch_vector(miss_counts, "miss_counts");
    const at::Tensor int_tensors[] = {index_block_table,      actual_seq_lengths_query,
                                      actual_seq_lengths_key, offload_seq_lengths_key,
                                      num_cache_tokens,       request_state,
                                      req_pool_entries,       cache_slots_pool,
                                      topk_src_ids,           topk_dst_slots,
                                      topk_miss_counts,       miss_src_ids,
                                      miss_dst_slots,         miss_counts};
    for (const auto &tensor : int_tensors)
        TORCH_CHECK(tensor.scalar_type() == at::kInt, "metadata and outputs must be int32.");
    const auto device = query.device();
    TORCH_CHECK(device.is_privateuseone() || query.is_meta(), "all tensors must be on NPU or Meta.");
    const at::Tensor tensors[] = {index_weights,
                                  query_dequant_scale,
                                  query,
                                  index_key_dequant_scale,
                                  index_key_cache,
                                  index_block_table,
                                  actual_seq_lengths_query,
                                  actual_seq_lengths_key,
                                  offload_seq_lengths_key,
                                  num_cache_tokens,
                                  request_state,
                                  req_pool_entries,
                                  cache_slots_pool,
                                  topk_src_ids,
                                  topk_dst_slots,
                                  topk_miss_counts,
                                  miss_src_ids,
                                  miss_dst_slots,
                                  miss_counts};
    for (const auto &tensor : tensors)
        TORCH_CHECK(tensor.device() == device, "all tensors must be on one device.");
    for (const at::Tensor *tensor :
         {&query_dequant_scale, &query, &index_block_table, &actual_seq_lengths_query, &actual_seq_lengths_key,
          &offload_seq_lengths_key, &num_cache_tokens, &request_state, &req_pool_entries, &cache_slots_pool,
          &topk_src_ids, &topk_dst_slots, &topk_miss_counts, &miss_src_ids, &miss_dst_slots, &miss_counts})
        TORCH_CHECK(tensor->is_contiguous(), "all tensors except weights/key/key-scale must be contiguous.");
}

HOST_API void fused_quant_lightning_indexer_manage(
    const at::Tensor &index_weights, const at::Tensor &query_dequant_scale, const at::Tensor &query,
    const at::Tensor &index_key_dequant_scale, const at::Tensor &index_key_cache, const at::Tensor &index_block_table,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
    const at::Tensor &offload_seq_lengths_key, const at::Tensor &num_cache_tokens, const at::Tensor &request_state,
    const at::Tensor &req_pool_entries, const at::Tensor &cache_slots_pool, const at::Tensor &topk_src_ids,
    const at::Tensor &topk_dst_slots, const at::Tensor &topk_miss_counts, const at::Tensor &miss_src_ids,
    const at::Tensor &miss_dst_slots, const at::Tensor &miss_counts)
{
    check_fused_quant_lightning_indexer_manage_arch35(
        index_weights, query_dequant_scale, query, index_key_dequant_scale, index_key_cache, index_block_table,
        actual_seq_lengths_query, actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens, request_state,
        req_pool_entries, cache_slots_pool, topk_src_ids, topk_dst_slots, topk_miss_counts, miss_src_ids,
        miss_dst_slots, miss_counts);
    if (query.is_meta()) return;
    c10_npu::OptionalNPUGuard guard(query.device());
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform && platform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND950,
                "quant LIM requires Ascend950 (A5)");
    const uint32_t groups = std::min(platform->GetCoreNumAic(), platform->GetCoreNumAiv() / 2U);
    TORCH_CHECK(groups > 0, "quant LIM: no usable mix core groups");
    const uint64_t batch = index_block_table.size(0), total = query.size(0);
    const uint64_t capacity = cache_slots_pool.size(1);
    const uint64_t routes = std::max(4U * batch, total);
    const uint64_t segments = fused_quant_li_manage_workspace::CandidateSegmentCapacity(capacity, groups);
    const uint64_t qli = fused_quant_li_manage_workspace::TotalBytes(
        capacity, routes, segments, capacity > lim_payload_layout::PACKED_SOURCE_CAPACITY);
    TORCH_CHECK(qli <= UINT32_MAX && routes <= UINT32_MAX && cache_slots_pool.size(0) <= UINT32_MAX,
                "quant LIM workspace/shape exceeds uint32");
    FusedQuantLightningIndexerManageTilingData t{};
    t.usedCoreNum = groups;
    t.batchSize = batch;
    t.totalQueryRows = total;
    t.poolSize = cache_slots_pool.size(0);
    t.sourceCapacity = capacity;
    t.indexHeads = query.size(1);
    t.maxBlockNumPerBatch = index_block_table.size(1);
    t.weightStride = index_weights.stride(0);
    t.keyStride = index_key_cache.stride(0);
    t.scaleStride = index_key_dequant_scale.stride(0);
    t.managementWorkspaceOffset = qli;
    t.scoreWorkspaceStride = 4U * capacity * sizeof(uint16_t);
    t.scoreRowStride = capacity;
    t.queryTileSize = query.size(1) == 32 && (total > 64 || batch * 2U > groups) ? 4U : 2U;
    t.routeStorageRows = routes;
    auto tiling = ge_helper::TilingTensorCache<FusedQuantLightningIndexerManageTilingData>::Get(t, "quant_lim");
    const uint64_t bytes = platform->GetLibApiWorkSpaceSize() + qli + (batch + total) * 32U;
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, query.options().dtype(at::kByte));
    const uint32_t blocks = platform->CalcTschBlockDim(groups * 2U, groups, groups * 2U);
    const std::vector<at::Tensor> owners = {index_weights,
                                            query_dequant_scale,
                                            query,
                                            index_key_dequant_scale,
                                            index_key_cache,
                                            index_block_table,
                                            actual_seq_lengths_query,
                                            actual_seq_lengths_key,
                                            offload_seq_lengths_key,
                                            num_cache_tokens,
                                            request_state,
                                            req_pool_entries,
                                            cache_slots_pool,
                                            topk_src_ids,
                                            topk_dst_slots,
                                            topk_miss_counts,
                                            miss_src_ids,
                                            miss_dst_slots,
                                            miss_counts,
                                            workspace,
                                            tiling};
    EXEC_OWNED_KERNEL_CMD(fused_quant_lightning_indexer_manage, blocks, owners, index_weights, query_dequant_scale,
                          query, index_key_dequant_scale, index_key_cache, index_block_table, actual_seq_lengths_query,
                          actual_seq_lengths_key, offload_seq_lengths_key, num_cache_tokens, request_state,
                          req_pool_entries, cache_slots_pool, topk_src_ids, topk_dst_slots, topk_miss_counts,
                          miss_src_ids, miss_dst_slots, miss_counts, cache_slots_pool, workspace, tiling);
}
}  // namespace sglang::npu_kernel
