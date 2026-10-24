// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Apache-2.0. A5 entry adapted from vllm-ascend PR #18383.
#include "kernel_operator.h"
#define EVENT_ID5 5U
#if (__CCE_AICORE__ == 310)
#include "arch35/fused_quant_lightning_indexer_manage_impl.h"
#endif
__global__ __aicore__ void fused_quant_lightning_indexer_manage(
    GM_ADDR weights, GM_ADDR queryDequantScale,
    GM_ADDR query, GM_ADDR keyDequantScale,
    GM_ADDR key, GM_ADDR blockTable,
    GM_ADDR actualSeqLengthsQuery,
    GM_ADDR actualSeqLengthsKey,
    GM_ADDR offloadSeqLengthsKey,
    GM_ADDR cacheTokens, GM_ADDR requestState,
    GM_ADDR reqPoolEntries, GM_ADDR cacheSlots,
    GM_ADDR topkSourceIds, GM_ADDR topkSlots,
    GM_ADDR topkMissCounts, GM_ADDR missSrcIds,
    GM_ADDR missDstSlots, GM_ADDR missCount,
    GM_ADDR cacheSlotsOut, GM_ADDR workspace,
    GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    RunFusedQuantLightningIndexerManageArch35(
        weights, queryDequantScale, query, keyDequantScale, key, blockTable,
        actualSeqLengthsQuery, actualSeqLengthsKey, offloadSeqLengthsKey,
        cacheTokens, requestState, reqPoolEntries, cacheSlots, topkSourceIds,
        topkSlots, topkMissCounts, missSrcIds, missDstSlots, missCount,
        cacheSlotsOut, workspace, tiling);
    if ASCEND_IS_AIV {
        // The management publishers use scalar GM stores for compact counts
        // and invalid rows. SetValue dirties the AIV DCache; SyncAll and the
        // direct-launch return do not publish those stores to GM. Flush on
        // every AIV, including owners of invalid non-offload requests.
        AscendC::GlobalTensor<int32_t> scalarOutputs;
        scalarOutputs.SetGlobalBuffer((__gm__ int32_t *)topkMissCounts);
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                         AscendC::DcciDst::CACHELINE_OUT>(scalarOutputs);
    }
#endif
}
