import sys

from src.utils import ca_train as ca_train_utils


def test_ca_train_command_uses_python_for_script_override(tmp_path) -> None:
    script = tmp_path / "hmog_vqgan_experiment.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    assert ca_train_utils.ca_train_command("hmog_vqgan_experiment.py", override=script) == [sys.executable, str(script)]


def test_ca_train_command_uses_frozen_subcommand(monkeypatch) -> None:
    monkeypatch.setattr(ca_train_utils, "is_frozen", lambda: True)

    assert ca_train_utils.ca_train_command("hmog_vqgan_experiment.py") == [sys.executable, "ca-train-vqgan"]
