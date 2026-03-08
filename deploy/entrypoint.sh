#!/usr/bin/env bash
set -euo pipefail

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
fi

mkdir -p /app/ascend_logs /app/kernel_meta

case ":${PYTHONPATH:-}:" in
  *:/app:*)
    ;;
  *)
    export PYTHONPATH="/app${PYTHONPATH:+:${PYTHONPATH}}"
    ;;
esac

exec "$@"
