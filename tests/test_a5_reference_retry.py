"""CPU-only validation of the local reference retry and wheel reuse path."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmark/lim_a5_mtp_ab/retry_reference_local.sh"
ENV_HELPER = ROOT / "python/sgl_kernel_npu/sgl_kernel_npu/_original_opp.py"
BASH = shutil.which("bash")
if not BASH and os.name == "nt":
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    if candidate.is_file():
        BASH = str(candidate)
PIN = "7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e"
VENDOR = "vendors/sglc8original_transformer"
try:
    CURRENT_TORCH = importlib.metadata.version("torch")
    CURRENT_TORCH_NPU = importlib.metadata.version("torch-npu")
except importlib.metadata.PackageNotFoundError:
    CURRENT_TORCH = CURRENT_TORCH_NPU = None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(BASH and shutil.which("findmnt"), "bash and findmnt required")
class LocalReferenceRetry(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="a5-reference-retry-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.old = self.base / "old"
        self.new = self.base / "new-local"
        self.ascend = self.base / "set_env.sh"
        self.ascend.write_text(
            "export ASCEND_HOME_PATH=/fake/ascend\n", encoding="utf-8"
        )
        self._make_wheel_and_install()
        self._make_reference()

    def _make_wheel_and_install(self):
        self.old.mkdir()
        (self.old / "sglang-build.exitcode").write_text("0\n", encoding="utf-8")
        package = self.old / "kernel_runtime/sgl_kernel_npu"
        package.mkdir(parents=True)
        package_files = {
            "__init__.py": b"# fixture package\n",
            "_original_opp.py": ENV_HELPER.read_bytes(),
            "lib/libsgl_kernel_npu.so": b"fake native library bytes",
        }
        bundle_vendor = package / "lib/a5_original_opp" / VENDOR
        opapi = bundle_vendor / "op_api/lib/libcust_opapi.so"
        opapi.parent.mkdir(parents=True)
        opapi.write_bytes(b"fake production OPP library")
        runtime = bundle_vendor / "op_impl/runtime.json"
        runtime.parent.mkdir(parents=True)
        runtime.write_text("{}\n", encoding="utf-8")
        bundle_files = {
            p.relative_to(package / "lib/a5_original_opp").as_posix(): digest(p)
            for p in (opapi, runtime)
        }
        manifest = {
            "backend": "original",
            "vendor": VENDOR,
            "files": bundle_files,
            "operators": {"lim": {"revision": PIN}},
        }
        bundle = package / "lib/a5_original_opp"
        (bundle / "runtime_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        package_files["lib/a5_original_opp/runtime_manifest.json"] = (
            bundle / "runtime_manifest.json"
        ).read_bytes()
        package_files[
            "lib/a5_original_opp" + "/" + VENDOR + "/op_api/lib/libcust_opapi.so"
        ] = opapi.read_bytes()
        package_files[
            "lib/a5_original_opp" + "/" + VENDOR + "/op_impl/runtime.json"
        ] = runtime.read_bytes()
        for name, content in package_files.items():
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.wheel = self.old / "sgl_kernel_npu-fixture.whl"
        with zipfile.ZipFile(self.wheel, "w") as archive:
            for name, content in package_files.items():
                archive.writestr("sgl_kernel_npu/" + name, content)
        (self.old / "wheel.json").write_text(
            json.dumps({"path": str(self.wheel), "sha256": digest(self.wheel)}),
            encoding="utf-8",
        )
        self.package = package

    def _make_reference(self):
        self.reference = self.base / "previous-container-reference"
        self.reference.mkdir()
        (self.reference / "exit_code").write_text("0\n", encoding="utf-8")
        source = self.reference / "prepared/source"
        (source / "csrc/include").mkdir(parents=True)
        (source / "csrc/include/kernel.h").write_text(
            "pinned source\n", encoding="utf-8"
        )
        (self.reference / "prepared/SOURCE_MANIFEST.json").write_text(
            json.dumps({"operator": "lim", "upstream": {"revision": PIN}}),
            encoding="utf-8",
        )
        (self.reference / "source_files.sha256").write_text(
            f"{digest(source / 'csrc/include/kernel.h')}  csrc/include/kernel.h\n",
            encoding="utf-8",
        )
        installed = (
            self.reference / "opp/vendors/a5limbaseline_transformer/bin/set_env.bash"
        )
        installed.parent.mkdir(parents=True)
        installed.write_text("export MOCK_VENDOR=1\n", encoding="utf-8")
        opapi = (
            self.reference
            / "opp/vendors/a5limbaseline_transformer/op_api/lib/libcust_opapi.so"
        )
        opapi.parent.mkdir(parents=True)
        opapi.write_bytes(b"fake reference OPP library")
        (self.reference / "installed_files.sha256").write_text(
            f"{digest(installed)}  vendors/a5limbaseline_transformer/bin/set_env.bash\n"
            f"{digest(opapi)}  vendors/a5limbaseline_transformer/op_api/lib/libcust_opapi.so\n",
            encoding="utf-8",
        )
        binding = self.reference / "binding/a5_original_lim_binding.so"
        binding.parent.mkdir(parents=True)
        binding.write_bytes(b"fake binding")
        binding_manifest = {
            "kind": "donor_adapter_wrapper_not_a_rebuilt_device_kernel",
            "library": str(binding.resolve()),
            "library_sha256": digest(binding),
            "source_manifest_sha256": digest(
                self.reference / "prepared/SOURCE_MANIFEST.json"
            ),
            "torch": CURRENT_TORCH or "unavailable",
            "torch_npu": CURRENT_TORCH_NPU or "unavailable",
        }
        (binding.parent / "binding_manifest.json").write_text(
            json.dumps(binding_manifest), encoding="utf-8"
        )
        (self.reference / "paths.env").write_text(
            f"export ORIGINAL_LIM_BINDING={binding}\n"
            f"export ORIGINAL_LIBRARY={binding}\n"
            f"export ORIGINAL_OPP_ROOT={self.reference / 'opp/vendors/a5limbaseline_transformer'}\n"
            f"export ORIGINAL_BINDING_MANIFEST={self.reference / 'binding/binding_manifest.json'}\n"
            f"export ORIGINAL_SOURCE_ROOT={source}\n"
            f"export ORIGINAL_SOURCE_REVISION={PIN}\n"
            f"export ORIGINAL_SOURCE_MANIFEST={self.reference / 'prepared/SOURCE_MANIFEST.json'}\n"
            f"export ORIGINAL_SOURCE_HASHES={self.reference / 'source_files.sha256'}\n"
            f"export ORIGINAL_INSTALLED_HASHES={self.reference / 'installed_files.sha256'}\n",
            encoding="utf-8",
        )

    def _snapshot_old(self):
        return {
            path.relative_to(self.old).as_posix(): digest(path)
            for path in self.old.rglob("*")
            if path.is_file()
        }

    def _run(self, *, output=None, mutate=None):
        args = [
            BASH,
            str(SCRIPT),
            str(self.old),
            str(output or self.new),
            "--ascend-env",
            str(self.ascend),
            "--python",
            sys.executable,
        ]
        if mutate:
            args.extend(["--reference-root", str(self.reference)])
        return subprocess.run(args, capture_output=True, text=True, check=False)

    def test_valid_wheel_install_and_reference_are_verified_without_rebuild(self):
        if not CURRENT_TORCH or not CURRENT_TORCH_NPU:
            self.skipTest(
                "torch and torch-npu distribution metadata required for reference reuse"
            )
        cache = self.package / "__pycache__"
        cache.mkdir()
        (cache / "__init__.cpython-312.pyc").write_bytes(b"pip-generated cache file")
        before = self._snapshot_old()
        result = self._run(mutate=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, self._snapshot_old())
        self.assertEqual(
            (self.new / "retry-reference.exitcode").read_text().strip(), "0"
        )
        wheel_proof = json.loads((self.new / "wheel-install-proof.json").read_text())
        self.assertEqual(wheel_proof["status"], "verified")
        self.assertEqual(wheel_proof["package_file_count"], len(self.package_files()))
        self.assertEqual(
            json.loads((self.new / "reference-proof.json").read_text())["status"],
            "verified-standard-build_original-output",
        )
        session = (self.new / "session.env").read_text()
        for key in ("A5_BUILD", "A5_REFERENCE", "A5_KERNEL", "A5_ROOT"):
            self.assertIn("export " + key + "=", session)
        self.assertFalse(
            (self.new / "tools").exists(), "reuse path must not bootstrap tools"
        )
        sourced = subprocess.run(
            [
                BASH,
                "-c",
                'set -euo pipefail; source "$1"; printf "%s\\n%s\\n%s\\n%s\\n" "$ORIGINAL_LIM_BINDING" "$ORIGINAL_OPP_ROOT" "$KERNEL_PYTHON_ROOT" "$ASCEND_CUSTOM_OPP_PATH"',
                "bash",
                str(self.new / "session.env"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(sourced.returncode, 0, sourced.stdout + sourced.stderr)
        self.assertEqual(
            sourced.stdout.splitlines(),
            [
                str(self.reference / "binding/a5_original_lim_binding.so"),
                str(self.reference / "opp/vendors/a5limbaseline_transformer"),
                str(self.old / "kernel_runtime"),
                str(self.package / "lib/a5_original_opp" / VENDOR),
            ],
        )

    def package_files(self):
        return [
            path
            for path in self.package.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ]

    def test_bad_wheel_sha_is_rejected_without_modifying_old_tree(self):
        receipt = json.loads((self.old / "wheel.json").read_text())
        receipt["sha256"] = "0" * 64
        (self.old / "wheel.json").write_text(json.dumps(receipt), encoding="utf-8")
        before = self._snapshot_old()
        result = self._run(mutate=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wheel.json SHA256", result.stdout + result.stderr)
        self.assertEqual(before, self._snapshot_old())
        self.assertFalse((self.new / "session.env").exists())

    def test_bad_installed_package_file_is_rejected_without_modifying_old_tree(self):
        (self.package / "lib/libsgl_kernel_npu.so").write_bytes(
            b"tampered installed file"
        )
        before = self._snapshot_old()
        result = self._run(mutate=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("installed package SHA256 differs", result.stdout + result.stderr)
        self.assertEqual(before, self._snapshot_old())
        self.assertFalse((self.new / "session.env").exists())

    def test_reference_binding_hash_mismatch_is_rejected(self):
        (self.reference / "binding/a5_original_lim_binding.so").write_bytes(
            b"changed reference binding"
        )
        before = self._snapshot_old()
        result = self._run(mutate=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("binding library SHA256 mismatch", result.stdout + result.stderr)
        self.assertEqual(before, self._snapshot_old())
        self.assertFalse((self.new / "session.env").exists())

    def test_reference_from_different_torch_is_rejected(self):
        path = self.reference / "binding/binding_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["torch"] = "0.0.incompatible"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        result = self._run(mutate=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("torch version does not match", result.stdout + result.stderr)
        self.assertFalse((self.new / "session.env").exists())

    def test_extra_installed_source_is_rejected(self):
        (self.package / "stale_extension.py").write_text("stale = True\n")
        result = self._run(mutate=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("file set differs from wheel", result.stdout + result.stderr)
        self.assertFalse((self.new / "session.env").exists())

    def test_existing_or_symlink_output_root_is_rejected(self):
        existing = self.base / "already-exists"
        existing.mkdir()
        marker = existing / "keep"
        marker.write_text("keep", encoding="utf-8")
        result = self._run(output=existing, mutate=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertEqual(
            self._snapshot_old()["wheel.json"], digest(self.old / "wheel.json")
        )
        link = self.base / "dangling-output"
        link.symlink_to(self.base / "absent-target", target_is_directory=True)
        result = self._run(output=link, mutate=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(link.is_symlink())
        self.assertFalse((self.base / "absent-target").exists())


if __name__ == "__main__":
    unittest.main()
