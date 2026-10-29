"""Backend registration and calls; imports of accelerator libraries are deferred."""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def loaded_libraries():
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split()
        p = fields[-1] if fields else ""
        if p.startswith("/") and ".so" in p and Path(p).is_file():
            paths.add(os.path.realpath(p))
    return [{"path": p, "sha256": file_sha(p)} for p in sorted(paths)]


def load_backends(args):
    import torch
    import torch_npu

    acl = importlib.import_module("acl")

    root = Path(args.kernel_python_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("kernel-python-root must be a directory")
    sys.path.insert(0, str(root))
    if "sgl_kernel_npu" in sys.modules:
        raise RuntimeError("sgl_kernel_npu was already imported; cannot enforce source")
    package = importlib.import_module("sgl_kernel_npu")
    package_path = Path(package.__file__).resolve(strict=True)
    if root not in package_path.parents:
        raise RuntimeError(f"sgl_kernel_npu came from {package_path}, outside {root}")
    sgl_library = (package_path.parent / "lib" / "libsgl_kernel_npu.so").resolve(
        strict=True
    )
    if root not in sgl_library.parents:
        raise RuntimeError(f"SGLang library came from {sgl_library}, outside {root}")
    registered_libraries = {
        str(Path(path).resolve()) for path in torch.ops.loaded_libraries
    }
    if str(sgl_library) not in registered_libraries:
        raise RuntimeError(f"SGLang library was not registered by torch: {sgl_library}")
    if args.original_library:
        torch.ops.load_library(str(Path(args.original_library).resolve(strict=True)))
    if args.original_module:
        importlib.import_module(args.original_module)
    namespace, sep, name = args.original_op.partition(".")
    if not sep or not namespace or not name:
        raise ValueError("--original-op must be namespace.operator")
    try:
        original = getattr(getattr(torch.ops, namespace), name)
        ported = torch.ops.npu.fused_quant_lightning_indexer_manage
        li = torch_npu.npu_quant_lightning_indexer
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"{namespace}::{name}", "PrivateUse1"
        ):
            raise RuntimeError(f"{args.original_op} lacks PrivateUse1 kernel")
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            "npu::fused_quant_lightning_indexer_manage", "PrivateUse1"
        ):
            raise RuntimeError("SGLang LIM lacks PrivateUse1 kernel")
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            f"failed to register original or SGLang kernel: {exc}"
        ) from exc
    if namespace == "npu" and name == "fused_quant_lightning_indexer_manage":
        raise RuntimeError(
            "original and SGLang must use distinct operator registrations"
        )
    torch.npu.set_device(args.device)
    device_index = torch.device(args.device).index or 0
    soc = acl.rt.get_device_info(device_index, 601)
    if tuple(soc) != (3510, 0):
        raise RuntimeError(f"A5/Ascend 950 required; acl device info 601={soc!r}")
    return (
        torch,
        torch_npu,
        li,
        original,
        ported,
        {
            "soc_acl_601": list(soc),
            "sglang_package": str(package_path),
            "sglang_package_sha256": file_sha(package_path),
            "sglang_library": str(sgl_library),
            "sglang_library_sha256": file_sha(sgl_library),
            "original_library": (
                str(Path(args.original_library).resolve())
                if args.original_library
                else None
            ),
            "original_library_sha256": (
                file_sha(Path(args.original_library).resolve())
                if args.original_library
                else None
            ),
            "original_module": args.original_module,
            "original_op": args.original_op,
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", None),
            "device": args.device,
        },
    )


def call_li(li, d):
    return li(
        query=d["query"],
        key=d["key"],
        weights=d["weights"],
        query_dequant_scale=d["query_scale"],
        key_dequant_scale=d["key_scale"],
        actual_seq_lengths_query=d["query_ends"],
        actual_seq_lengths_key=d["actual"],
        block_table=d["table"],
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=3,
    )


def call_lim(op, d, pool, outputs):
    result = op(
        d["weights"],
        d["query_scale"],
        d["query"],
        d["key_scale"],
        d["key"],
        d["table"],
        d["query_ends"],
        d["actual"],
        d["prefix"],
        d["capacity"],
        d["state"],
        d["entries"],
        pool,
        *outputs,
    )
    if result is not None:
        raise RuntimeError("LIM ABI requires caller-owned output and None return")


def opp_libraries(root):
    """Record installed OPP .so candidates; presence does not prove execution."""
    base = Path(root).resolve(strict=True)
    if not base.is_dir():
        raise ValueError(f"OPP root is not a directory: {base}")
    paths = sorted(path.resolve() for path in base.rglob("*.so*") if path.is_file())
    return [{"path": str(path), "sha256": file_sha(path)} for path in paths]
