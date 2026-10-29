#!/usr/bin/env bash
# Build into a task-private package root; never replace the installed wheel.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT=$(cd -- "$HERE/../.." && pwd)
BUILD_ARTIFACT_ROOT=${1:?Usage: bash build_private.sh /new/absolute/build-artifacts}
PYTHON=${PYTHON:-python3}
SOC_VERSION=${SOC_VERSION:-Ascend950}
if [[ "$BUILD_ARTIFACT_ROOT" != /* || -e "$BUILD_ARTIFACT_ROOT" ]]; then
  echo 'Choose a new absolute artifact directory; existing results are preserved.' >&2
  exit 2
fi
mkdir -p -- "$BUILD_ARTIFACT_ROOT/logs"
trap 'rc=$?; printf "%s\n" "$rc" > "$BUILD_ARTIFACT_ROOT/build.exitcode"' EXIT
ASCEND_ENV=${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
source "$ASCEND_ENV"
export BUILD_ARTIFACT_ROOT SOURCE_ROOT SOC_VERSION
"$PYTHON" - <<'PY' > "$BUILD_ARTIFACT_ROOT/source_provenance.json"
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ["SOURCE_ROOT"])
def git(*args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    return result.stdout.decode(errors="replace") if result.returncode == 0 else None
diff = git("diff", "HEAD", "--binary")
print(json.dumps({"source_root": str(root), "revision": git("rev-parse", "HEAD"),
                  "status": git("status", "--short"),
                  "diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
                  "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
                  "target": os.environ.get("SOC_VERSION", "Ascend950")}, indent=2))
PY
cd -- "$SOURCE_ROOT"
set +e
bash build.sh -a kernels "$SOC_VERSION" 2>&1 | tee "$BUILD_ARTIFACT_ROOT/logs/sglang-build.log"
build_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$build_rc" > "$BUILD_ARTIFACT_ROOT/sglang-build.exitcode"
if (( build_rc != 0 )); then exit "$build_rc"; fi
"$PYTHON" - "$SOURCE_ROOT" "$BUILD_ARTIFACT_ROOT" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
wheels = list((source / "output").glob("sgl_kernel_npu*.whl"))
if len(wheels) != 1:
    raise SystemExit("Expected one fresh kernel wheel; inspect output/ before retrying")
destination = target / wheels[0].name
shutil.copy2(wheels[0], destination)
with destination.open("rb") as stream:
    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
(target / "wheel.json").write_text(json.dumps({"path": str(destination),
                                             "sha256": checksum}, indent=2) + "\n")
PY
wheel=("$BUILD_ARTIFACT_ROOT"/sgl_kernel_npu*.whl)
"$PYTHON" -m pip install --no-deps --target "$BUILD_ARTIFACT_ROOT/kernel_runtime" "${wheel[0]}" \
  2>&1 | tee "$BUILD_ARTIFACT_ROOT/logs/private-install.log"
if [[ -n "${VLLM_ASCEND_SOURCE:-}" ]]; then
  "$PYTHON" "$HERE/build_original_binding.py" \
    --source-root "$VLLM_ASCEND_SOURCE" \
    --build-dir "$BUILD_ARTIFACT_ROOT/original_binding" \
    2>&1 | tee "$BUILD_ARTIFACT_ROOT/logs/original-binding-build.log"
fi
printf 'export KERNEL_PYTHON_ROOT=%q\n' "$BUILD_ARTIFACT_ROOT/kernel_runtime" \
  > "$BUILD_ARTIFACT_ROOT/paths.env"
if [[ -f "$BUILD_ARTIFACT_ROOT/original_binding/binding_manifest.json" ]]; then
  "$PYTHON" - "$BUILD_ARTIFACT_ROOT/original_binding/binding_manifest.json" \
    >> "$BUILD_ARTIFACT_ROOT/paths.env" <<'PY'
import json
import shlex
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
print("export ORIGINAL_LIM_BINDING=" + shlex.quote(manifest["library"]))
if manifest.get("source_revision"):
    print("export ORIGINAL_SOURCE_REVISION=" + shlex.quote(manifest["source_revision"]))
print("export ORIGINAL_BINDING_MANIFEST=" + shlex.quote(sys.argv[1]))
PY
fi
printf 'Private build complete. Read %s/paths.env before running.\n' "$BUILD_ARTIFACT_ROOT"
