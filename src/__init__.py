from __future__ import annotations

import os

# Prevent PyTorch from auto-importing broken optional backends (e.g. torch_npu
# without full Ascend runtime). Backends can still be imported explicitly.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
