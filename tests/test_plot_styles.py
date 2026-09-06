from simulator.plot_styles import find_method_run_dir, method_style


def test_method7_output_uses_starlit_style() -> None:
    assert method_style("method7") == method_style("starlit")


def test_starlit_finds_historical_method7_output(tmp_path) -> None:
    historical = tmp_path / "method7"
    historical.mkdir()

    assert find_method_run_dir(tmp_path, "starlit") == historical


def test_starlit_prefers_current_output_name(tmp_path) -> None:
    historical = tmp_path / "method7"
    current = tmp_path / "starlit"
    historical.mkdir()
    current.mkdir()

    assert find_method_run_dir(tmp_path, "starlit") == current


def test_phoenix_finds_historical_phoenix2_output(tmp_path) -> None:
    historical = tmp_path / "phoenix2"
    historical.mkdir()

    assert find_method_run_dir(tmp_path, "phoenix") == historical


def test_phoenix_prefers_current_output_name(tmp_path) -> None:
    historical = tmp_path / "phoenix2"
    current = tmp_path / "phoenix"
    historical.mkdir()
    current.mkdir()

    assert find_method_run_dir(tmp_path, "phoenix") == current
