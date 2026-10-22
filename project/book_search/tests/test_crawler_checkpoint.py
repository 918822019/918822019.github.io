"""爬虫断点续爬单元测试。"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.crawler.engine import (
    CHECKPOINT_TABLE_SQL,
    _INTERRUPTED,
    _INTERRUPT_LOCK,
    clear_checkpoint,
    ensure_checkpoint_table,
    get_resume_start,
    is_interrupted,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute(CHECKPOINT_TABLE_SQL)
    c.commit()
    return c


class TestCheckpointTable:
    def test_ensure_checkpoint_table_creates_table(self, conn: sqlite3.Connection) -> None:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='crawl_checkpoints'")
        assert cur.fetchone() is not None

    def test_save_and_load_checkpoint(self, conn: sqlite3.Connection) -> None:
        save_checkpoint(conn, "crawl-books", 1, 1000, 500)
        result = load_checkpoint(conn, "crawl-books", 1, 1000)
        assert result == 500

    def test_load_nonexistent_checkpoint(self, conn: sqlite3.Connection) -> None:
        result = load_checkpoint(conn, "crawl-books", 1, 1000)
        assert result is None

    def test_save_overwrites_existing(self, conn: sqlite3.Connection) -> None:
        save_checkpoint(conn, "crawl-books", 1, 1000, 500)
        save_checkpoint(conn, "crawl-books", 1, 1000, 800)
        assert load_checkpoint(conn, "crawl-books", 1, 1000) == 800

    def test_different_commands_have_separate_checkpoints(self, conn: sqlite3.Connection) -> None:
        save_checkpoint(conn, "crawl-books", 1, 1000, 500)
        save_checkpoint(conn, "crawl-content", 1, 1000, 300)
        assert load_checkpoint(conn, "crawl-books", 1, 1000) == 500
        assert load_checkpoint(conn, "crawl-content", 1, 1000) == 300

    def test_clear_checkpoint(self, conn: sqlite3.Connection) -> None:
        save_checkpoint(conn, "crawl-books", 1, 1000, 500)
        clear_checkpoint(conn, "crawl-books", 1, 1000)
        assert load_checkpoint(conn, "crawl-books", 1, 1000) is None

    def test_clear_only_removes_matching_range(self, conn: sqlite3.Connection) -> None:
        save_checkpoint(conn, "crawl-books", 1, 1000, 500)
        save_checkpoint(conn, "crawl-books", 1001, 2000, 1500)
        clear_checkpoint(conn, "crawl-books", 1, 1000)
        assert load_checkpoint(conn, "crawl-books", 1, 1000) is None
        assert load_checkpoint(conn, "crawl-books", 1001, 2000) == 1500

    def test_save_within_transaction_succeeds(self, conn: sqlite3.Connection) -> None:
        save_checkpoint(conn, "sync-all", 1, 5000, 2500)
        assert load_checkpoint(conn, "sync-all", 1, 5000) == 2500


class TestGetResumeStart:
    def test_returns_start_when_auto_continue_disabled(self, conn: sqlite3.Connection) -> None:
        assert get_resume_start(conn, "crawl-books", 1, 1000, False) == 1

    def test_returns_start_when_no_checkpoint(self, conn: sqlite3.Connection) -> None:
        assert get_resume_start(conn, "crawl-books", 1, 1000, True) == 1

    def test_returns_checkpoint_plus_one(self, conn: sqlite3.Connection) -> None:
        save_checkpoint(conn, "crawl-books", 1, 1000, 500)
        assert get_resume_start(conn, "crawl-books", 1, 1000, True) == 501

    def test_clamps_to_start_if_checkpoint_before_start(self, conn: sqlite3.Connection) -> None:
        save_checkpoint(conn, "crawl-books", 1, 1000, 50)
        assert get_resume_start(conn, "crawl-books", 100, 1000, True) == 100

    def test_clamps_to_end_plus_one_if_checkpoint_past_end(self, conn: sqlite3.Connection) -> None:
        save_checkpoint(conn, "crawl-books", 1, 1000, 999)
        result = get_resume_start(conn, "crawl-books", 1, 1000, True)
        assert result == 1000
        assert result <= 1000

    def test_different_command_checkpoint_not_confused(self, conn: sqlite3.Connection) -> None:
        save_checkpoint(conn, "crawl-content", 1, 1000, 700)
        result = get_resume_start(conn, "crawl-books", 1, 1000, True)
        assert result == 1


class TestSignalHandler:
    def test_is_interrupted_returns_false_initially(self) -> None:
        with _INTERRUPT_LOCK:
            saved = _INTERRUPTED
        assert saved is False

    def test_is_interrupted_returns_true_after_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with _INTERRUPT_LOCK:
            _INTERRUPTED = False

        from src.crawler.engine import _signal_handler
        _signal_handler(2, None)
        assert is_interrupted() is True

        with _INTERRUPT_LOCK:
            _INTERRUPTED = False

    def test_double_signal_calls_sys_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with _INTERRUPT_LOCK:
            _INTERRUPTED = False

        exit_called = []
        monkeypatch.setattr(sys, "exit", lambda code: exit_called.append(code))

        from src.crawler.engine import _signal_handler
        _signal_handler(2, None)
        _signal_handler(2, None)

        assert len(exit_called) >= 1

        with _INTERRUPT_LOCK:
            _INTERRUPTED = False


class TestEnsureSchemaWithCheckpoint:
    def test_checkpoint_table_created_with_normal_schema(self) -> None:
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        from src.crawler.engine import ensure_schema
        ensure_schema(c)
        ensure_checkpoint_table(c)
        tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "crawl_checkpoints" in tables
        assert "books" in tables
        assert "chapters" in tables
        c.close()

    def test_checkpoint_table_idempotent(self) -> None:
        c = sqlite3.connect(":memory:")
        ensure_checkpoint_table(c)
        ensure_checkpoint_table(c)
        count = c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='crawl_checkpoints'").fetchone()[0]
        assert count == 1
        c.close()


if __name__ == "__main__":
    pytest.main([__file__])
