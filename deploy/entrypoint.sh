#!/usr/bin/env bash
set -euo pipefail

export APP_ROOT="${APP_ROOT:-/app}"
export CA_APP_ROOT="${CA_APP_ROOT:-${APP_ROOT}}"
export CA_CONFIG_PATH="${CA_CONFIG_PATH:-${APP_ROOT}/ca_config.toml}"
export ASCEND_DRIVER_HOME="${ASCEND_DRIVER_HOME:-/usr/local/Ascend/driver}"
export ASCEND_INSTALL_INFO="${ASCEND_INSTALL_INFO:-/etc/ascend_install.info}"
export ASCEND_TOOLKIT_ROOT="${ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/ascend-toolkit}"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-${ASCEND_TOOLKIT_ROOT}/latest}"
export ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-${ASCEND_TOOLKIT_HOME}}"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-${ASCEND_TOOLKIT_HOME}/opp}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME}}"
export TOOLCHAIN_HOME="${TOOLCHAIN_HOME:-${ASCEND_TOOLKIT_HOME}/toolkit}"
export CA_SERVER_RUNTIME="${CA_SERVER_RUNTIME:-auto}"

if [ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]; then
  set +u
  # shellcheck disable=SC1091
  source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
  set -u
elif [ -f "${ASCEND_TOOLKIT_ROOT}/set_env.sh" ]; then
  set +u
  # shellcheck disable=SC1091
  source "${ASCEND_TOOLKIT_ROOT}/set_env.sh"
  set -u
fi

mkdir -p \
  "${APP_ROOT}/ascend_logs" \
  "${APP_ROOT}/kernel_meta" \
  "${APP_ROOT}/data_storage/raw_data" \
  "${APP_ROOT}/data_storage/processed_data" \
  "${APP_ROOT}/data_storage/inference" \
  "${APP_ROOT}/data_storage/models" \
  "${APP_ROOT}/data_storage/hmog_preprocessed" \
  "${APP_ROOT}/runtime/ca_train/cached_windows" \
  "${APP_ROOT}/runtime/ca_train/token_caches" \
  "${APP_ROOT}/results" \
  "${APP_ROOT}/logs" \
  "${APP_ROOT}/certs"

app_pythonpath="${APP_ROOT}:${APP_ROOT}/ca_train"
ascend_pythonpath="${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe"
ascend_binpath="${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:/usr/local/sbin"
bundle_ld_library_path="/app/bin/_internal:/app/bin/_internal/torch:/app/bin/_internal/torch/lib:/app/bin/_internal/torch_npu:/app/bin/_internal/torch_npu/lib"
ascend_ld_library_path="${bundle_ld_library_path}:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/$(arch):${ASCEND_DRIVER_HOME}/lib64:${ASCEND_DRIVER_HOME}/lib64/common:${ASCEND_DRIVER_HOME}/lib64/driver"
export PYTHONPATH="${app_pythonpath}:${ascend_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}"

case ":${PATH:-}:" in
  *:"${ASCEND_TOOLKIT_HOME}/bin":*)
    ;;
  *)
    export PATH="${ascend_binpath}${PATH:+:${PATH}}"
    ;;
esac

case ":${LD_LIBRARY_PATH:-}:" in
  *:"${ASCEND_TOOLKIT_HOME}/lib64":*)
    ;;
  *)
    export LD_LIBRARY_PATH="${ascend_ld_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    ;;
esac

if [ $# -eq 0 ]; then
  set -- server
fi

runtime_mode="${CA_SERVER_RUNTIME}"
if [ "${runtime_mode}" = "auto" ]; then
  if [ -x "/app/bin/ca-server" ]; then
    runtime_mode="frozen"
  elif [ -f "/app/src/cli.py" ]; then
    runtime_mode="python"
  else
    echo "Unable to detect runtime layout under /app" >&2
    exit 1
  fi
fi

if [ "${runtime_mode}" = "python" ]; then
  cli_cmd=(python -m src.cli)
else
  cli_cmd=(/app/bin/ca-server)
fi

case "$1" in
  server|serve)
    shift
    set -- "${cli_cmd[@]}" serve "$@"
    ;;
  processing)
    shift
    set -- "${cli_cmd[@]}" processing "$@"
    ;;
  training)
    shift
    set -- "${cli_cmd[@]}" training "$@"
    ;;
  policy-search)
    shift
    set -- "${cli_cmd[@]}" policy-search "$@"
    ;;
  auth)
    shift
    set -- "${cli_cmd[@]}" auth "$@"
    ;;
  ca-train-vqgan)
    shift
    if [ "${runtime_mode}" = "python" ] && [ -f "/app/ca_train/hmog_vqgan_experiment.py" ]; then
      set -- python /app/ca_train/hmog_vqgan_experiment.py "$@"
    else
      set -- "${cli_cmd[@]}" ca-train-vqgan "$@"
    fi
    ;;
  help|-h|--help)
    set -- "${cli_cmd[@]}" --help
    ;;
esac

exec "$@"
