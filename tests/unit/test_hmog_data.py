import csv

import numpy as np

from ca_train.hmog_data import _resample_time_axis, iter_windows_from_csv_unlabeled_with_session


def test_resample_time_axis_same_width_returns_detached_copy() -> None:
    src = np.arange(6 * 20, dtype=np.float32).reshape(6, 20)
    out = _resample_time_axis(src, target_width=20)

    assert out.shape == src.shape
    assert np.array_equal(out, src)

    # Guard against accidental aliasing: callers reuse a mutable row buffer.
    out[0, 0] = -12345.0
    assert float(src[0, 0]) != -12345.0


def test_window_reader_selects_six_or_nine_axes(tmp_path) -> None:
    path = tmp_path / "windows.csv"
    header = [
        "subject", "session", "timestamp",
        "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
        "mag_x", "mag_y", "mag_z", "window_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow(["u", "s", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        writer.writerow(["u", "s", 10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 0])

    six = list(
        iter_windows_from_csv_unlabeled_with_session(
            path, window_size_sec=0.02, target_width=2, input_height=6
        )
    )
    nine = list(
        iter_windows_from_csv_unlabeled_with_session(
            path, window_size_sec=0.02, target_width=2, input_height=9
        )
    )
    assert six[0][3].shape == (1, 6, 2)
    assert nine[0][3].shape == (1, 9, 2)
    assert nine[0][3][0, 8].tolist() == [9.0, 90.0]
