# HiSparse MF SplitKV 配套说明

本说明对应个人 kernel 分支 `feat/hisparse-mf-split-kv-adapt`，framework 配套分支为 [`feat/hisparse-mf-split-kv-adapt`](https://github.com/McZyWu/sgl-sglang/tree/feat/hisparse-mf-split-kv-adapt/benchmark/hisparse/split_kv)。kernel 基线提交为 `b47332ce9736a7920139ad8f29364b27ef416dea`；framework 调用与 kernel split ABI 必须配套。

## Split KV ABI

`fused_scatter_copy_sparse_flash_attention` 接收独立的 HBM K、V tensor，以及独立的 MF DRAM K、V pointer。CKV K 与 KPE/RoPE V 的物理布局和宽度不同，调用方必须提供各自的 tensor 和 pointer，以及 HBM、DRAM 对应的 block table；K/V 共享 token 到 slot 的映射，不能把两个 DRAM base 当作单个合并 KV 地址。

当前kernel接口在 `csrc/pytorch_extensions.cpp` 声明 `hbm_k_cache`、`hbm_v_cache`、`dram_k_dev_ptr` 与 `dram_v_dev_ptr`；C++ host入口在 `csrc/fused_scatter_copy_sparse_flash_attention/op_host/fused_copy_sfa.cpp` 接收两个DRAM指针。该接口本身不证明真实MF/GVA映射、分布式可见性、PD serving、数值精度或吞吐已经通过。

## 容器内构建与独立安装

在已准备好CANN 9.1和准确submodule checkout的容器内运行：

```bash
bash benchmark/hisparse/split_kv/setup_kernel.sh --check-only /path/to/sgl-kernel-npu
BUILD_JOBS=2 bash benchmark/hisparse/split_kv/setup_kernel.sh /path/to/sgl-kernel-npu /path/to/private/kernel-prefix
```

这些命令从 framework 仓库根目录执行。`CANN_ENV` 可覆盖环境脚本路径，默认 `/usr/local/Ascend/ascend-toolkit/set_env.sh`。`KERNEL_SOC` 默认为 `910`（A3）；A5 构建应明确设置为 `950` 或实际编译目标。容器应预装构建依赖和 wheel；脚本在 wheel 缺失时退出，避免底层 `build.sh` 隐式安装到共享 Python 环境。

若省略prefix，脚本会在唯一的任务临时目录下创建独立安装prefix。构建前先source CANN环境，保留原有 `PYTHONPATH`；要求kernel HEAD为 `b47332ce9736a7920139ad8f29364b27ef416dea` 或其后代，并逐项核对PTO `af66b6ada43c40ad9583731883fb9bef65ac840a` 与 catlass `70eda02b1271ff0e8c47db94469fae576feba97b`。检查submodule HEAD之前会先确认该子目录自身有 `.git`，避免Git回落到父仓库。

构建从 `git archive` 与真实submodule源码归档出的独立副本进行。只在任务副本内规范化shell换行并将CMake构建并行度由仓库脚本的 `-j 16` 调整为 `--parallel ${BUILD_JOBS:-2}`；原始kernel工作区保持不变。wheel用 `pip install --no-deps --target <prefix>` 安装，不写共享site-packages。成功时脚本打印 `export KERNEL_PYTHON_PATH=<prefix>`，容器服务可在启动时使用该变量追加本次wheel路径。

脚本不运行NPU测试，也不停止服务。真实NPU算子、MF共享内存、HCCL、PD功能、模型精度与性能尚需在资源授权并现场确认后验证。编译成功只表示wheel构建完成，不代表这些验证通过。

## 2026-11-04 A3 实机结果

217 的 B103/CANN 9.1.0 容器已从上述 kernel/PTO/catlass 精确版本构建 wheel，退出码 0；构建副本仅限制并行度和规范化 shell LF。两机安装到独立 prefix，检查实际算子 `torch.ops.npu.fused_scatter_copy_sparse_flash_attention.default` schema 的四个 split 参数，未覆盖共享 site-packages。

| 证据 | 实际结果 |
| --- | --- |
| wheel SHA256 | `5de65d7eb4ee9acda2f86007eb02ada47ed71e4c0c390645c8dd70a6cac1286c` |
| 两端主 SO SHA256 | `0b27a873ddd35b789cbe649c1dd17509b6ffac4e62b88829ce856bbb502096f7` |
| Q4 跨页尾部 eager/graph | 通过；CPU oracle max abs error `0.000669136643409729` |
| 128 head eager/graph | 通过；max abs error `0.0006541609764099121` |
| C12288 Q4 首填 eager/graph | 通过；max abs error `0.0005991607904434204` |

三个用例均复用本仓 `tests/test_copy_sfa_dynamic_tnd.py` 原有断言，由 framework 的 `probe_copy_sfa.py` 打印误差；cache 和 eager/graph 为零容差数值比较，不扩大为原始字节比较。探针退出码为 0。

framework 另使用 MF 1.3 私有 A3 AccOffload 扩展完成 TP16 的跨页快照/恢复小探针：16 rank 各 39 项检查均通过，包含 K/V、index、scale 原始字节与 owner 恢复，reset 仅为 spy。这一结果不等于完整 resident manager 或 PD 回退已经通过。两机原生自然语言取值的 512/4096/64000 输入诊断已正确且 exact，强制回退和正式性能仍待完整 controller 验证；没有本轮实测 TPS。

原随机性能输入的切片在原生 P 单 token 路径上也发生过跨请求不一致，失败数据保留。配套 framework 将自然语言精度输入与历史性能输入分开，保持正式性能的 64000/1024、prefix100、C12288 和 cached63872 门禁不变。完整范围见 framework [验证记录](https://github.com/McZyWu/sgl-sglang/blob/feat/hisparse-mf-split-kv-adapt/benchmark/hisparse/split_kv/VALIDATION_zh.md)。本轮 kernel 算子源码沿用 b47332ce，未为这些结果修改算法或阈值。
