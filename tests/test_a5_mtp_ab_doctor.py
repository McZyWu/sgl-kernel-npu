"""CPU-only tests for the environment doctor and private dependency bootstrap."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DOCTOR_PATH = ROOT / "benchmark" / "lim_a5_mtp_ab" / "doctor.py"
BOOTSTRAP_PATH = ROOT / "benchmark" / "lim_a5_mtp_ab" / "bootstrap_deps.sh"
SPEC = importlib.util.spec_from_file_location("a5_mtp_ab_doctor", DOCTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
DOCTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOCTOR)


class DoctorTests(unittest.TestCase):
    def test_default_collection_is_environment_only(self) -> None:
        report = DOCTOR.collect_environment(env={})
        self.assertEqual(report["mode"], "environment_only")
        self.assertFalse(report["device_context_created"])
        self.assertNotIn("device_result", report)
        self.assertIsInstance(report["required_missing"], list)

    def test_cann_path_checks_are_cpu_file_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "bin" / "setenv.bash").write_text("", encoding="utf-8")
            (root / "include" / "acl").mkdir(parents=True)
            (root / "include" / "acl" / "acl.h").write_text("", encoding="utf-8")
            ascendc = root / "tools" / "tikcpp" / "ascendc_kernel_cmake"
            ascendc.mkdir(parents=True)
            (ascendc / "ascendc.cmake").write_text("", encoding="utf-8")

            checked = DOCTOR._root_checks({"ASCEND_HOME_PATH": str(root)})

        self.assertTrue(
            checked["set_env_script"].replace("\\", "/").endswith("bin/setenv.bash")
        )
        self.assertTrue(checked["acl_header"])
        self.assertTrue(checked["ascendc_cmake"].endswith("ascendc.cmake"))

    def test_arch35_ascconfig_and_env_script_file_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setup = root / "bin" / "setenv.bash"
            setup.parent.mkdir()
            setup.write_text("", encoding="utf-8")
            config = root / "ascendc_devkit" / "arch35" / "ASCConfig.cmake"
            config.parent.mkdir(parents=True)
            config.write_text("", encoding="utf-8")

            checked = DOCTOR._root_checks({"ASCEND_HOME_PATH": str(setup)})

        self.assertTrue(checked["found"])
        self.assertEqual(Path(checked["set_env_script"]), setup.resolve())
        self.assertEqual(Path(checked["ascendc_cmake"]), config.resolve())

    def test_main_without_device_flag_never_runs_device_check(self) -> None:
        report = {
            "status": "ok",
            "device_context_created": False,
            "distributions": {},
            "cann": {},
            "opp": {},
            "required_missing": [],
            "warnings": [],
        }
        with (
            patch.object(DOCTOR, "collect_environment", return_value=report),
            patch.object(
                DOCTOR,
                "_device_check",
                side_effect=AssertionError("device check called"),
            ),
        ):
            self.assertEqual(DOCTOR.main([]), 0)
        self.assertFalse(report["device_context_created"])

    def test_device_check_requires_explicit_device(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            DOCTOR.main(["--device-check"])
        self.assertEqual(raised.exception.code, 2)

    def test_json_output_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "doctor.json"
            output.write_text("keep", encoding="utf-8")
            with patch.object(
                DOCTOR,
                "_device_check",
                side_effect=AssertionError("device check called"),
            ):
                with self.assertRaises(SystemExit) as raised:
                    DOCTOR.main(
                        ["--output", str(output), "--device-check", "--device", "npu:0"]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")


class BootstrapTests(unittest.TestCase):
    def _bash(self) -> str:
        bash = shutil.which("bash")
        if bash:
            return bash
        windows_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if windows_bash.is_file():
            return str(windows_bash)
        self.skipTest("Bash is unavailable")
        return "bash"

    def test_dry_run_is_private_and_does_not_create_venv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "build-tools"
            requirements = Path(temporary) / "build-requirements.txt"
            requirements.write_text("ninja==1.11.1\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    self._bash(),
                    str(BOOTSTRAP_PATH),
                    str(root),
                    "--dry-run",
                    "--requirements",
                    str(requirements),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--system-site-packages", completed.stdout)
            self.assertIn("cmake==3.30.5", completed.stdout)
            self.assertIn("ninja==1.11.1", completed.stdout)
            self.assertIn("wheel==0.45.1", completed.stdout)
            self.assertFalse(root.exists())

    def test_bootstrap_rejects_runtime_package_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "build-tools"
            requirements = Path(temporary) / "unsafe-requirements.txt"
            requirements.write_text("torch==2.1.0\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    self._bash(),
                    str(BOOTSTRAP_PATH),
                    str(root),
                    "--dry-run",
                    "--requirements",
                    str(requirements),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not on the build-only allowlist", completed.stderr)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
