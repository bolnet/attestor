"""Tests for CLI."""

import json
import os
import tempfile

import pytest

from attestor.cli import main

from .conftest import TEST_CONFIG


@pytest.fixture
def store_dir():
    with tempfile.TemporaryDirectory() as d:
        store_path = os.path.join(d, "test-agent")
        main(["init", store_path])
        # Overwrite config.json with test database settings
        config_path = os.path.join(store_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        config.update(TEST_CONFIG)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        yield store_path


class TestCLI:
    def test_init(self, test_config):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "my-agent")
            main(["init", path])
            assert os.path.isfile(os.path.join(path, "memory.db"))

    def test_add_and_list(self, store_dir, capsys):
        main(["add", store_dir, "User likes Python", "--tags", "preference,coding", "--category", "preference"])
        captured = capsys.readouterr()
        assert "Added memory" in captured.out

        main(["list", store_dir])
        captured = capsys.readouterr()
        assert "User likes Python" in captured.out

    def test_recall(self, store_dir, capsys):
        main(["add", store_dir, "User prefers dark mode"])
        main(["recall", store_dir, "dark mode"])
        captured = capsys.readouterr()
        assert "dark mode" in captured.out

    def test_search(self, store_dir, capsys):
        main(["add", store_dir, "Python fact", "--category", "tech"])
        main(["search", store_dir, "Python", "--category", "tech"])
        captured = capsys.readouterr()
        assert "Python" in captured.out

    def test_stats(self, store_dir, capsys):
        main(["add", store_dir, "fact1"])
        main(["add", store_dir, "fact2"])
        main(["stats", store_dir])
        captured = capsys.readouterr()
        assert "Total memories: 2" in captured.out

    def test_forget(self, store_dir, capsys):
        main(["add", store_dir, "temp fact"])
        # Get the memory id from list
        main(["list", store_dir])
        captured = capsys.readouterr()
        # Extract id from output like "(abc123def456) general: temp fact"
        memory_id = captured.out.split("(")[1].split(")")[0]

        main(["forget", store_dir, memory_id])
        captured = capsys.readouterr()
        assert "Archived" in captured.out

    def test_export_import(self, store_dir, capsys):
        main(["add", store_dir, "export test", "--tags", "test"])

        export_file = os.path.join(store_dir, "backup.json")
        main(["export", store_dir, "-o", export_file])

        # Import into new store
        with tempfile.TemporaryDirectory() as d:
            new_store = os.path.join(d, "imported")
            main(["init", new_store])
            # Overwrite with test config
            config_path = os.path.join(new_store, "config.json")
            with open(config_path) as f:
                config = json.load(f)
            config.update(TEST_CONFIG)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            main(["import", new_store, export_file])
            captured = capsys.readouterr()
            assert "Imported 1" in captured.out

    def test_inspect(self, store_dir, capsys):
        main(["add", store_dir, "inspect test"])
        main(["inspect", store_dir])
        captured = capsys.readouterr()
        assert "inspect test" in captured.out

    def test_compact(self, store_dir, capsys):
        main(["compact", store_dir])
        captured = capsys.readouterr()
        assert "Removed" in captured.out

    def test_timeline(self, store_dir, capsys):
        main(["add", store_dir, "Joined SoFi", "--entity", "SoFi", "--category", "career"])
        main(["timeline", store_dir, "--entity", "SoFi"])
        captured = capsys.readouterr()
        assert "SoFi" in captured.out

    def test_no_command(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "agent-memory" in captured.out.lower()
