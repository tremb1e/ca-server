from src.training.cli import parse_args


def test_training_cli_accepts_dash_prefixed_user_id() -> None:
    user_id = "-OzOz6DF-4eSwCnStI2kQBpbQTkWN6ODZSxyQv5MAAI="

    args = parse_args(["--user", user_id, "--device", "cpu"])

    assert args.user == user_id
    assert args.device == "cpu"
