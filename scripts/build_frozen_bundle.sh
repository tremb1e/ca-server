#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="${1:-${BUNDLE_TOOL:-pyinstaller}}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  else
    echo "python3/python not found in PATH" >&2
    exit 1
  fi
fi

cd "${ROOT_DIR}"
rm -rf build dist nuitka-dist

common_hidden_imports=(
  accelerator
  hmog_consecutive_rejects
  hmog_data
  hmog_metrics
  hmog_token_auth_inference
  hmog_token_transformer
  hmog_tokenizer
  hmog_vqgan_experiment
  hmog_vqgan_token_transformer_experiment
  platformdirs
  runtime_paths
  vqgan
)

if [[ "${TOOL}" == "nuitka" ]]; then
  "${PYTHON_BIN}" -m nuitka \
    --standalone \
    --assume-yes-for-downloads \
    --output-dir="${ROOT_DIR}/nuitka-dist" \
    --output-filename=ca-server \
    --include-package=src \
    --include-module=accelerator \
    --include-module=hmog_consecutive_rejects \
    --include-module=hmog_data \
    --include-module=hmog_metrics \
    --include-module=hmog_token_auth_inference \
    --include-module=hmog_token_transformer \
    --include-module=hmog_tokenizer \
    --include-module=hmog_vqgan_experiment \
    --include-module=hmog_vqgan_token_transformer_experiment \
    --include-module=runtime_paths \
    --include-module=vqgan \
    src/cli.py

  mkdir -p "${ROOT_DIR}/dist"
  nuitka_output="$(find "${ROOT_DIR}/nuitka-dist" -maxdepth 1 -type d -name '*.dist' | head -n 1)"
  if [[ -z "${nuitka_output}" ]]; then
    echo "Nuitka build did not produce a standalone dist directory" >&2
    exit 1
  fi
  mv "${nuitka_output}" "${ROOT_DIR}/dist/ca-server"
  if [[ -f "${ROOT_DIR}/dist/ca-server/ca-server.bin" ]]; then
    mv "${ROOT_DIR}/dist/ca-server/ca-server.bin" "${ROOT_DIR}/dist/ca-server/ca-server"
  fi
  exit 0
fi

pyinstaller_args=(
  --clean
  --noconfirm
  --onedir
  --name ca-server
  --paths "${ROOT_DIR}"
  --paths "${ROOT_DIR}/ca_train"
  --collect-submodules src
  --collect-submodules grpc_health
  --copy-metadata torch
  --copy-metadata pandas
  --copy-metadata scikit-learn
)

if "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("torch_npu") is not None else 1)
PY
then
  pyinstaller_args+=(
    --collect-submodules torch_npu
    --collect-data torch_npu
    --collect-binaries torch_npu
  )
fi

for module_name in "${common_hidden_imports[@]}"; do
  pyinstaller_args+=(--hidden-import "${module_name}")
done

TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}" \
  "${PYTHON_BIN}" -m PyInstaller "${pyinstaller_args[@]}" src/cli.py
