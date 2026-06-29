from src.training.cli import parse_args


def test_training_cli_accepts_dash_prefixed_user_id() -> None:
    user = "-OzOz6DF-4eSwCnStI2kQBpbQTkWN6ODZSxyQv5MAAI="
    args = parse_args(["--user", user, "--device", "cpu"])
    assert args.user == user
    assert args.device == "cpu"


def test_training_cli_parses_window_sizes_and_epochs() -> None:
    args = parse_args(["--user", "plain", "--window-sizes", "0.2", "--vqgan-epochs", "3"])
    assert args.user == "plain"
    assert args.window_sizes == [0.2]
    assert args.vqgan_epochs == 3
    assert args.reuse is True
