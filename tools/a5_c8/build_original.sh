#!/usr/bin/env bash
# Build only the pinned upstream LIM OPP; never build DeepEP.
set -eo pipefail

if [[ ${1:-} == --help ]]; then
    echo "Usage: build_original.sh FRESH_OUTPUT_ROOT"
    exit 0
fi
[[ $# -eq 1 ]] || { echo "Usage: build_original.sh FRESH_OUTPUT_ROOT" >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output=$1
[[ ! -e "$output" ]] || { echo "Preserve existing build directory: $output" >&2; exit 2; }
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
output=$(cd -- "$output" && pwd -P)
[[ "$output" != *[[:space:]]* ]] || { echo "CANN OPP build path must not contain whitespace" >&2; exit 2; }
exec > >(tee "$output/build.log") 2>&1
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > "$output/exit_code"
    exit "$result"
}
trap finish EXIT
ascend_env=${ASCEND_ENV_PATH:-}
if [[ -z "$ascend_env" ]]; then
    if [[ -f "${ASCEND_HOME_PATH:-}/set_env.sh" ]]; then
        ascend_env="$ASCEND_HOME_PATH/set_env.sh"
    elif [[ -f "${ASCEND_HOME_PATH:-}/bin/setenv.bash" ]]; then
        ascend_env="$ASCEND_HOME_PATH/bin/setenv.bash"
    else
        ascend_env=/usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
fi
source "$ascend_env"
set -u
python3 "$script_dir/prepare.py" --output-root "$output/prepared"
csrc="$output/prepared/source/csrc"
(cd "$csrc" && bash build.sh --pkg --soc=ascend950 \
    --ops=fused_quant_lightning_indexer_manage \
    --vendor_name=sglc8original "-j${SGL_KERNEL_A5_C8_BUILD_JOBS:-4}")
shopt -s nullglob
installers=("$csrc"/build/*.run)
shopt -u nullglob
[[ ${#installers[@]} -eq 1 ]] || { echo "Expected one original OPP installer" >&2; exit 1; }
mkdir -- "$output/opp"
bash "${installers[0]}" -- "--install-path=$output/opp"
python3 - "$output" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
opp = root / "opp"
files = {}
for path in sorted(opp.rglob("*")):
    if path.is_file():
        if path.is_symlink():
            raise RuntimeError(f"Unexpected symlink in original OPP: {path}")
        files[path.relative_to(opp).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "installed_files.sha256.json").write_text(json.dumps(files, indent=2) + "\n")
PY
echo "Original C8 OPP built and privately installed: $output"
