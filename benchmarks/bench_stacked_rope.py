"""Compare per-layer Q/K RoPE with one non-power-of-two stacked-head call.

Run on an idle NPU from an environment with this kernel checkout installed:
python benchmarks/bench_stacked_rope.py --output stacked-rope.json
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.norm.fused_rope_qk_mqa import fused_rope_qk_mqa


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(47)
    cases = []
    for tokens in (32, 256):
        layers, heads, dim = 5, 2, 64
        q = torch.randn(tokens, layers, heads, dim, dtype=torch.bfloat16, device="npu")
        k = torch.randn_like(q)
        cache = torch.randn(tokens, dim, device="npu")

        def per_layer():
            return torch.stack(
                [
                    fused_rope_qk_mqa(q[:, i], k[:, i], cache, dim, True)[1]
                    for i in range(layers)
                ],
                dim=1,
            )

        def stacked():
            return fused_rope_qk_mqa(
                q.flatten(1, 2), k.flatten(1, 2), cache, dim, True
            )[1].view_as(k)

        graphs, outputs = {}, {}
        samples = {"per_layer": [], "stacked": []}
        for name, operation in (("per_layer", per_layer), ("stacked", stacked)):
            operation()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(20):
                    outputs[name] = operation()
            graphs[name] = graph
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            outputs["per_layer"], outputs["stacked"], rtol=0, atol=0
        )
        for trial in range(7):
            for name in (
                ("per_layer", "stacked") if trial % 2 == 0 else ("stacked", "per_layer")
            ):
                graphs[name].replay()
                torch.npu.synchronize()
                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                    enable_timing=True
                )
                begin.record()
                for _ in range(3):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end) * 1000 / 60)
        case = dict(
            tokens=tokens,
            layers=layers,
            heads_per_layer=heads,
            samples_us=samples,
            median_us={
                name: statistics.median(values) for name, values in samples.items()
            },
            bitwise_equal=True,
        )
        print(json.dumps(case), flush=True)
        cases.append(case)
        del graphs, graph, outputs, operation, per_layer, stacked
    args.output.write_text(
        json.dumps({"torch": torch.__version__, "cases": cases}, indent=2)
    )


if __name__ == "__main__":
    main()
