import pandas as pd
import pytest
from typer.testing import CliRunner

from edsmith.cli import app
from edsmith.session.state import SessionState, save_state

runner = CliRunner()

_STUB_RESPONSE = (
    "<score>6.5</score>\n"
    "<confidence>high</confidence>\n"
    "The essay addresses the task with clear arguments and sufficient detail."
)

_SMALL_TRAIN = pd.DataFrame([
    {"question": "Should cities ban cars?", "essay": f"Essay number {i}.", "band": "6.0"}
    for i in range(3)
])


@pytest.fixture
def cli_env(tmp_path, stub_provider, monkeypatch):
    stub_provider.set(_STUB_RESPONSE)
    monkeypatch.setattr(
        "edsmith.providers.openrouter.OpenRouterProvider",
        lambda: stub_provider,
    )
    drive_path = tmp_path / "drive"
    state = SessionState(session_id="test-session")
    save_state(state, drive_path)
    data_dir = drive_path / "sessions" / "test-session" / "data"
    data_dir.mkdir(parents=True)
    _SMALL_TRAIN.to_parquet(data_dir / "train.parquet", index=False)
    return drive_path


class TestExaminerPassCLI:
    def _invoke(self, drive_path):
        return runner.invoke(
            app,
            ["examiner-pass", "test-session", "0", "--drive", str(drive_path)],
        )

    def test_exit_code_zero(self, cli_env):
        result = self._invoke(cli_env)
        assert result.exit_code == 0, result.output

    def test_output_contains_done(self, cli_env):
        result = self._invoke(cli_env)
        assert "Done" in result.output

    def test_output_contains_session_id(self, cli_env):
        result = self._invoke(cli_env)
        assert "test-session" in result.output

    def test_output_contains_feedback_path(self, cli_env):
        result = self._invoke(cli_env)
        assert "feedback_iter0.parquet" in result.output

    def test_output_contains_score_distributions(self, cli_env):
        result = self._invoke(cli_env)
        assert "mean=" in result.output

    def test_parquet_written(self, cli_env):
        self._invoke(cli_env)
        assert (cli_env / "sessions" / "test-session" / "feedback_iter0.parquet").exists()
