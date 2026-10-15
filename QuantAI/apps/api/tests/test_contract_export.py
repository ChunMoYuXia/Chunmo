import subprocess
import sys
from pathlib import Path


def test_openapi_export_is_repeatable_and_check_detects_drift(tmp_path):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "openapi.json"
    command = [sys.executable, "-W", "error", "scripts/export_openapi.py", "--output", str(output)]
    first = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    content = output.read_bytes()
    assert b'"/v1/jobs/{job_id}"' in content and b"openapi-export-only" not in content
    second = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert second.returncode == 0 and output.read_bytes() == content
    assert subprocess.run([*command, "--check"], cwd=root, capture_output=True).returncode == 0
    output.write_text("{}", encoding="utf-8")
    assert subprocess.run([*command, "--check"], cwd=root, capture_output=True).returncode == 1
    assert output.read_text(encoding="utf-8") == "{}"
