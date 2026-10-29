"""CPU-only diagnostics tests for the A5 MTP preflight shell script."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "benchmark" / "lim_a5_mtp_ab" / "preflight.sh"


def bash_executable() -> str | None:
    configured = os.environ.get("BASH_EXE")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("bash")
    if found:
        return found
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    return str(git_bash) if git_bash.is_file() else None


def bash_path(path: Path) -> str:
    """Convert a Windows drive path to the Git Bash /<drive>/ form."""
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 2 and value[1] == ":":
        return f"/{value[0].lower()}{value[2:]}"
    return value


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bash = bash_executable()
        if cls.bash is None:
            raise unittest.SkipTest("bash is required for preflight shell tests")

    def run_preflight(
        self,
        npu_failure: str = "",
        python_failure: bool = False,
        proc_exit_code: int = 0,
        legacy_syntax_help: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        kernel_root = root / "kernel"
        kernel_root.mkdir()

        failing_args = npu_failure.removeprefix("npu-smi ")
        proc_case = (
            f"  'info proc') echo 'mock proc diagnostic'; exit {proc_exit_code} ;;\n"
            if proc_exit_code
            else "  'info proc') echo 'mock proc ok' ;;\n"
        )
        legacy_syntax_cases = (
            "  '-m') echo 'Command: -h --help -v info set clear upgrade'; exit 23 ;;\n"
            "  'info -t proc') echo 'Usage: npu-smi info <watch|proc|-h|-m|-l|-t type>'; exit 23 ;;\n"
            if legacy_syntax_help
            else ""
        )
        npu_mock = (
            "#!/usr/bin/env bash\n"
            'printf "%s\\n" "$*" >> "$NPU_CALLS"\n'
            'case "$*" in\n'
            f"  '{failing_args}') echo 'mock npu-smi diagnostic'; exit 23 ;;\n"
            + "  'info') echo 'mock info ok' ;;\n"
            + "  'info -m') echo 'mock mapping ok' ;;\n"
            + legacy_syntax_cases
            + proc_case
            + "  *) echo 'mock npu-smi ok' ;;\n"
            + "esac\n"
        )
        (bin_dir / "npu-smi").write_text(npu_mock, encoding="utf-8")
        os.chmod(bin_dir / "npu-smi", 0o755)
        python_code = "#!/usr/bin/env bash\n" "cat >/dev/null\n" + (
            "echo 'Traceback (most recent call last):'; echo 'RuntimeError: mock probe failure' >&2; exit 29\n"
            if python_failure
            else "exit 0\n"
        )
        (bin_dir / "python-mock").write_text(python_code, encoding="utf-8")
        os.chmod(bin_dir / "python-mock", 0o755)

        output = root / "preflight"
        calls_file = root / "npu-smi-calls.log"
        env = os.environ.copy()
        env.update(
            {
                "PATH": bash_path(bin_dir) + ":/usr/bin:/bin",
                "PYTHON_BIN": bash_path(bin_dir / "python-mock"),
                "NPU_CALLS": bash_path(calls_file),
                "ASCEND_HOME_PATH": bash_path(root),
                "ASCEND_RT_VISIBLE_DEVICES": "0",
                "KERNEL_PYTHON_ROOT": bash_path(kernel_root),
                "ORIGINAL_MODULE": "mock_backend",
                "ORIGINAL_OP": "mock_op",
                "ASCEND_CUSTOM_OPP_PATH": "",
                "ORIGINAL_LIM_BINDING": "",
            }
        )
        result = subprocess.run(
            [
                self.bash,
                bash_path(PREFLIGHT),
                "--output",
                bash_path(output),
                "--device",
                "npu:0",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        return result, output

    def test_fatal_npu_smi_failures_are_printed_and_probe_still_runs(self) -> None:
        for command in ("npu-smi info", "npu-smi info -m"):
            with self.subTest(command=command):
                result, output = self.run_preflight(npu_failure=command)
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"FAILED check={command} exit_code=23", result.stderr)
                self.assertIn("mock npu-smi diagnostic", result.stderr)
                log_name = command.replace(" ", "_").replace("/", "_") + ".log"
                self.assertIn(bash_path(output / log_name), result.stderr)
                self.assertTrue((output / "python-probe.log").is_file())

    def test_legacy_command_syntax_is_never_invoked(self) -> None:
        result, output = self.run_preflight(legacy_syntax_help=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = (output.parent / "npu-smi-calls.log").read_text().splitlines()
        self.assertEqual(calls, ["info", "info -m", "info proc"])
        self.assertTrue((output / "npu-smi_info.log").is_file())
        self.assertTrue((output / "npu-smi_info_-m.log").is_file())
        self.assertTrue((output / "npu-smi_info_proc.log").is_file())

    def test_python_failure_prints_traceback_and_actual_rc(self) -> None:
        result, output = self.run_preflight(python_failure=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAILED check=python runtime probe exit_code=29", result.stderr)
        self.assertIn("RuntimeError: mock probe failure", result.stderr)
        self.assertIn(bash_path(output / "python-probe.log"), result.stderr)

    def test_proc_timeout_is_logged_but_not_fatal(self) -> None:
        result, output = self.run_preflight(proc_exit_code=124)
        self.assertEqual(result.returncode, 0, result.stderr)
        process_log = (output / "npu-smi_info_proc.log").read_text()
        self.assertIn("$ timeout 5s npu-smi info proc", process_log)
        self.assertIn("exit_code=124", process_log)
        self.assertNotIn("FAILED check=", result.stderr)

    def test_success_keeps_status_and_success_marker(self) -> None:
        result, output = self.run_preflight()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((output / "status").read_text().strip(), "0")
        self.assertIn("PREFLIGHT_OK", result.stdout)
        self.assertEqual(
            (output / "npu-smi_info_proc.log").read_text().splitlines()[-1],
            "exit_code=0",
        )


if __name__ == "__main__":
    unittest.main()
