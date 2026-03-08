import numpy as np

from ca_train.hmog_data import _resample_time_axis


def test_resample_time_axis_same_width_returns_detached_copy() -> None:
    src = np.arange(12 * 20, dtype=np.float32).reshape(12, 20)
    out = _resample_time_axis(src, target_width=20)

    assert out.shape == src.shape
    assert np.array_equal(out, src)

    # Guard against accidental aliasing: callers reuse a mutable row buffer.
    out[0, 0] = -12345.0
    assert float(src[0, 0]) != -12345.0
