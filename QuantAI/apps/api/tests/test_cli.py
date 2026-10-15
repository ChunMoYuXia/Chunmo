import subprocess
import sys


def test_cli_cannot_bind_public_network() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "portfolio_client", "--host", "0.0.0.0"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
