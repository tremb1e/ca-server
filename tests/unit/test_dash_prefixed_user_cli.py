from src.authentication.cli import parse_args as parse_auth_args
from src.policy_search.cli import parse_args as parse_policy_search_args
from src.processing.cli import parse_args as parse_processing_args
from src.training.cli import parse_args as parse_training_args


def test_user_cli_options_accept_dash_prefixed_device_id() -> None:
    user_id = "-OzOz6DF-4eSwCnStI2kQBpbQTkWN6ODZSxyQv5MAAI="

    assert parse_processing_args(["--user", user_id]).user == user_id
    assert parse_training_args(["--user", user_id, "--device", "cpu"]).user == user_id
    assert parse_policy_search_args(["--user", user_id, "--device", "cpu"]).user == user_id
    assert parse_auth_args(["--user", user_id, "--csv-path", "/tmp/in.csv"]).user == user_id


def test_user_cli_options_keep_plain_device_id_supported() -> None:
    user_id = "plain-device"

    assert parse_processing_args(["--user", user_id]).user == user_id
    assert parse_training_args(["--user", user_id, "--device", "cpu"]).user == user_id
    assert parse_policy_search_args(["--user", user_id, "--device", "cpu"]).user == user_id
    assert parse_auth_args(["--user", user_id, "--csv-path", "/tmp/in.csv"]).user == user_id
