"""Generated Local acceptance fixtures; no GPU or scientific evidence."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest


def controller():
    path = Path(__file__).resolve().parents[1] / "scripts/research.py"
    spec = importlib.util.spec_from_file_location("research_controller_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_queued_software_check_snapshots_research_fixtures_without_runtime_caches(tmp_path, monkeypatch):
    """Use the real git inventory/hash capture, replacing only harness dispatch."""
    from recursive_ssd import suite_queue
    cli = controller()
    root = tmp_path / "delivery"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    fixture = root / "research/design-v2/method-specs.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text('{"fixture": "retained method comparison specification"}\n')
    source = root / "recursive_ssd/source.py"
    source.parent.mkdir()
    source.write_text("fixture = True\n")
    excluded = ["runs/attempt/large-cache.json", "artifacts/model/shard.json",
                "returns/prior-archive.json", ".research-autopilot/records/cache.json"]
    for relative in excluded:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("runtime cache must not become software-test source\n")
    monkeypatch.setattr(cli, "ROOT", root)
    original_budget = root / "runs/authorizations/original.json"
    monkeypatch.setattr(suite_queue, "authorization_path", lambda r, q: original_budget)
    captured = {}

    def engineering(r, folder, command, **kwargs):
        assert r == root
        assert kwargs["budget_path"] == original_budget
        assert kwargs["seconds"] == 123
        refs = {value["path"]: value["sha256"] for value in kwargs["code"]}
        assert refs["research/design-v2/method-specs.json"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
        assert refs["recursive_ssd/source.py"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert not set(excluded) & refs.keys()
        assert command[1:] == ["-m", "pytest", "-q", "tests/test_suite_design.py", "-k", "catalog"]
        captured["called"] = True
        return {"status": "completed", "fixture": "dispatch replaced; no workload run"}

    monkeypatch.setattr(suite_queue, "_engineering", engineering)
    assert cli.main(["check", "--queue", "runs/q", "--seconds", "123", "--",
                     "tests/test_suite_design.py", "-k", "catalog"]) == 0
    assert captured["called"] is True


def test_explicit_engineering_snapshot_cannot_drop_its_queue_accounting():
    cli = controller()
    with pytest.raises(ValueError, match="original queue budget"):
        cli.engineering(["unused-fixture-command"], label="fixture", seconds=1,
                        code=[{"path": "source.py", "sha256": "a" * 64}])


def test_native_source_snapshot_closes_training_and_catalogue_file_dependencies(tmp_path, monkeypatch):
    """Replay real file readers against only the research records in source_refs."""
    from recursive_ssd import suite_queue, suite_design, suite_train
    root = Path(__file__).resolve().parents[1]
    captured = suite_queue.source_refs(root)
    research = [value for value in captured if value["path"].startswith("research/")]
    expected = {
        "research/design-v2/method-specs.json",
        "research/design-v2/data/train_prompts-clean-v2.jsonl",
        "research/design-v2/data/clean-training-manifest.json",
        "research/design-v2/data/humaneval-manifest.json",
        "research/design-v2/data/mbpp-plus-audit.json",
        "research/design-v2/data/lcb-native-row-audit.json",
    }
    assert {value["path"] for value in research} == expected
    staged = tmp_path / "isolated-attempt"
    for value in research:
        source = suite_queue.verify_ref(root, value)
        target = staged / value["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        suite_queue.verify_ref(staged, value)
    monkeypatch.setattr(suite_train, "DESIGN_DATA", staged / "research/design-v2/data")
    monkeypatch.setattr(suite_design, "ROOT", staged)
    ids = suite_train._clean_ids()
    assert len(ids) == len(set(ids)) == 32
    suite = suite_design.make_suite("development")
    assert len(suite["candidate_ids"]) == 15
    assert len(suite["evaluation_units"]) == 190
