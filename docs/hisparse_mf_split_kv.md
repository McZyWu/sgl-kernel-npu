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
