"""Regression tests: device/user IDs starting with '-' must not break CLI parsing.

A base64-like device hash can legitimately start with a dash (e.g. ``-NMxmSk...``);
argparse would otherwise treat it as an unknown option. ``normalize_option_value_args``
rewrites ``--user -X`` into ``--user=-X`` so the value parses correctly.
"""

from src.authentication.cli import parse_args as parse_auth_args
from src.policy_search.cli import parse_args as parse_policy_args
from src.processing.cli import parse_args as parse_processing_args
from src.training.cli import parse_args as parse_training_args

DASH_USER = "-OzOz6DF-4eSwCnStI2kQBpbQTkWN6ODZSxyQv5MAAI="
PLAIN_USER = "plain-device"


def test_user_cli_options_accept_dash_prefixed_device_id() -> None:
    assert parse_training_args(["--user", DASH_USER, "--device", "cpu"]).user == DASH_USER
    assert parse_auth_args(["--user", DASH_USER, "--csv-path", "/tmp/x.csv"]).user == DASH_USER
    assert parse_policy_args(["--user", DASH_USER, "--device", "cpu"]).user == DASH_USER
    assert parse_processing_args(["--user", DASH_USER]).user == DASH_USER


def test_user_cli_options_keep_plain_device_id_supported() -> None:
    assert parse_training_args(["--user", PLAIN_USER]).user == PLAIN_USER
    assert parse_auth_args(["--user", PLAIN_USER, "--csv-path", "/tmp/x.csv"]).user == PLAIN_USER
    assert parse_policy_args(["--user", PLAIN_USER]).user == PLAIN_USER
    assert parse_processing_args(["--user", PLAIN_USER]).user == PLAIN_USER


def test_user_cli_options_accept_already_normalized_equals_form() -> None:
    assert parse_training_args([f"--user={DASH_USER}", "--device", "cpu"]).user == DASH_USER
    assert parse_policy_args([f"--user={DASH_USER}"]).user == DASH_USER
