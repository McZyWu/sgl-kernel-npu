"""Build an isolated Torch wrapper for an installed donor A5 ACLNN kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    build = args.build_dir.resolve()
    if build.exists():
        parser.error("build directory exists; preserve it and choose a new directory")
    csrc = source / "csrc"
    adapter = csrc / "attention/fused_quant_lightning_indexer_manage"
    bridge_sources = [
        csrc / "aclnn_torch_adapter/NPUBridge.cpp",
        csrc / "aclnn_torch_adapter/NPUStorageImpl.cpp",
    ]
    required = [
        adapter / "fused_quant_lightning_indexer_manage_torch_adpt.h",
        adapter / "fused_quant_lightning_indexer_manage_ordered.h",
        csrc / "aclnn_torch_adapter/op_api_common.h",
    ] + bridge_sources
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("donor A5 source files missing: " + ", ".join(missing))
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        parser.error("source the Ascend environment before building the wrapper")

    import torch
    import torch_npu
    from torch.utils.cpp_extension import load

    build.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
    result = load(
        name="a5_original_lim_binding",
        sources=[str(here / "original_binding.cpp")]
        + [str(path) for path in bridge_sources],
        extra_include_paths=[
            str(csrc),
            str(npu_root.parent),
            str(npu_root / "include"),
            str(Path(ascend_home) / "include"),
        ],
        extra_cflags=["-O3", "-std=c++17", "-fvisibility=hidden"],
        extra_ldflags=[
            "-L" + str(npu_root / "lib"),
            "-ltorch_npu",
            "-L" + str(Path(ascend_home) / "lib64"),
            "-lascendcl",
        ],
        build_directory=str(build),
        is_python_module=False,
        verbose=True,
    )
    library = Path(result) if isinstance(result, str) else None
    if library is None or not library.is_file():
        matches = list(build.glob("a5_original_lim_binding*.so"))
        if len(matches) != 1:
            raise RuntimeError("cannot identify the built original A5 wrapper")
        library = matches[0]
    donor_files = sorted(path for path in adapter.rglob("*") if path.is_file())
    manifest = {
        "kind": "donor_adapter_wrapper_not_a_rebuilt_device_kernel",
        "library": str(library.resolve()),
        "library_sha256": checksum(library),
        "torch_op": "a5_original_lim.run",
        "source_root": str(source),
        "source_revision": git_value(source, "rev-parse", "HEAD"),
        "source_status": git_value(source, "status", "--short"),
        "donor_files": {
            str(path.relative_to(source)): checksum(path) for path in donor_files
        },
        "wrapper_sha256": checksum(here / "original_binding.cpp"),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "ascend_home": ascend_home,
        "ascend_custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "note": (
            "The donor ACLNN/OPP device kernel must already be installed. "
            "This wrapper preserves the donor adapter, including its ordered "
            "submission, and does not substitute the SGLang direct launcher."
        ),
    }
    (build / "binding_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
