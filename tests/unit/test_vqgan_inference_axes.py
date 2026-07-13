import numpy as np
import pandas as pd
import pytest

from src.authentication.vqgan_inference import windowize_dataframe
from src.processing.magnetometer import NINE_AXIS_COLUMNS


def test_online_windowizer_uses_model_input_height() -> None:
    rows = 20
    df = pd.DataFrame({name: np.arange(rows, dtype=np.float32) for name in NINE_AXIS_COLUMNS})
    _, six = windowize_dataframe(
        df,
        window_size_sec=0.2,
        overlap=0.5,
        sampling_rate_hz=100,
        target_width=20,
        input_height=6,
    )
    _, nine = windowize_dataframe(
        df,
        window_size_sec=0.2,
        overlap=0.5,
        sampling_rate_hz=100,
        target_width=20,
        input_height=9,
    )
    assert six.shape == (1, 1, 6, 20)
    assert nine.shape == (1, 1, 9, 20)


def test_nine_axis_windowizer_rejects_non_finite_magnetometer() -> None:
    rows = 20
    df = pd.DataFrame({name: np.zeros(rows, dtype=np.float32) for name in NINE_AXIS_COLUMNS})
    df.loc[0, "mag_x"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        windowize_dataframe(
            df,
            window_size_sec=0.2,
            overlap=0.5,
            sampling_rate_hz=100,
            target_width=20,
            input_height=9,
        )
