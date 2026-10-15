from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "relative_destination",
    ["dist/stock-portfolio-client/portfolio-api", "apps/client/src-tauri/resources/backend"],
)
def test_runtime_publication_removes_stale_files_and_preserves_other_outputs(
    tmp_path: Path, relative_destination: str
):
    from scripts.build_sidecar import publish_runtime

    source = tmp_path / ".cache/sidecar/portfolio-api"
    source.mkdir(parents=True)
    (source / "portfolio-api.exe").write_bytes(b"current-runtime")
    (source / "current.js").write_text("current-page", encoding="utf-8")
    destination = tmp_path / relative_destination
    destination.mkdir(parents=True)
    (destination / "old.js").write_text("stale-page", encoding="utf-8")
    unrelated = tmp_path / "dist/keep.txt"
    unrelated.parent.mkdir(exist_ok=True)
    unrelated.write_text("keep", encoding="utf-8")

    publish_runtime(tmp_path, relative_destination)

    assert {item.name for item in destination.iterdir()} == {"portfolio-api.exe", "current.js"}
    assert (destination / "portfolio-api.exe").read_bytes() == b"current-runtime"
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_runtime_publication_rejects_non_owned_target(tmp_path: Path):
    from scripts.build_sidecar import publish_runtime

    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        publish_runtime(tmp_path, "../outside")
    assert unrelated.read_text(encoding="utf-8") == "keep"
