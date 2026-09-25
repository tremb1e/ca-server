import pytest

from src.ca_config import TrainingConfig, load_ca_config


def test_sensor_weights_default_and_toml_override(tmp_path):
    assert load_ca_config(tmp_path / "missing.toml").training.sensor_weights == (0.5, 0.5, 0.0)
    path = tmp_path / "ca_config.toml"
    path.write_text("[training]\nsensor_weights = [0.6, 0.3, 0.1]\n", encoding="utf-8")
    assert load_ca_config(path).training.sensor_weights == (0.6, 0.3, 0.1)


@pytest.mark.parametrize("value", ["[0.5, 0.5]", "[0.4, 0.4, 0.1]", "[-0.1, 0.6, 0.5]", "[nan, 0.5, 0.5]"])
def test_invalid_sensor_weight_config_is_rejected(tmp_path, value):
    path = tmp_path / "ca_config.toml"
    path.write_text(f"[training]\nsensor_weights = {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sensor_weights"):
        load_ca_config(path)


def test_training_config_requires_three_weights():
    with pytest.raises(ValueError, match="sensor_weights"):
        TrainingConfig(sensor_weights=None)
