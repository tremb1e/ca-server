from src import cli


def test_main_bootstraps_multiprocessing_before_dispatch(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(cli.mp, "freeze_support", lambda: calls.append("freeze"))
    monkeypatch.setattr(cli, "_run_server", lambda: calls.append("serve") or 0)

    rc = cli.main(["serve"])

    assert rc == 0
    assert calls == ["freeze", "serve"]
