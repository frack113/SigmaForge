"""Tests for TaskDispatcher."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.workers.processor import TaskDispatcher
from src.workers.enums import WorkerName, WorkerStatus


class TestTaskDispatcherInit:
    def test_default_poll_interval(self, mock_db: MagicMock) -> None:
        with patch("src.workers.processor.DatabaseService.get_instance", return_value=mock_db):
            dispatcher = TaskDispatcher()
            assert dispatcher.poll_interval == 0.2

    def test_custom_poll_interval(self, mock_db: MagicMock) -> None:
        with patch("src.workers.processor.DatabaseService.get_instance", return_value=mock_db):
            dispatcher = TaskDispatcher(poll_interval=10)
            assert dispatcher.poll_interval == 10

    def test_initial_state(self, mock_db: MagicMock) -> None:
        with patch("src.workers.processor.DatabaseService.get_instance", return_value=mock_db):
            dispatcher = TaskDispatcher()
            assert dispatcher._running is False
            assert dispatcher._pending_tasks == {}
            assert dispatcher._worker_states == {}


class TestAskForWorker:
    def test_accepts_when_idle(self, mock_db: MagicMock) -> None:
        with patch("src.workers.processor.DatabaseService.get_instance", return_value=mock_db):
            dispatcher = TaskDispatcher()
            result = dispatcher.ask_for_worker(WorkerName.SIGMAREF_DISCOVERY, foo="bar")

        assert result is not None
        state = dispatcher._worker_states[WorkerName.SIGMAREF_DISCOVERY]
        assert state["status"] == WorkerStatus.WAITING
        assert state["current_task_id"]  # task_id generated internally
        assert dispatcher._pending_tasks[WorkerName.SIGMAREF_DISCOVERY]["foo"] == "bar"

    def test_rejects_when_waiting(self, mock_db: MagicMock) -> None:
        with patch("src.workers.processor.DatabaseService.get_instance", return_value=mock_db):
            dispatcher = TaskDispatcher()
            dispatcher.ask_for_worker(WorkerName.SIGMAREF_DISCOVERY)
            result = dispatcher.ask_for_worker(WorkerName.SIGMAREF_DISCOVERY)

        assert result is None

    def test_generates_unique_task_ids(self, mock_db: MagicMock) -> None:
        with patch("src.workers.processor.DatabaseService.get_instance", return_value=mock_db):
            dispatcher = TaskDispatcher()
            dispatcher.ask_for_worker(WorkerName.SIGMAREF_DISCOVERY)
            id1 = dispatcher._worker_states[WorkerName.SIGMAREF_DISCOVERY]["current_task_id"]

            # Reset to idle to accept another
            dispatcher._worker_states[WorkerName.SIGMAREF_DISCOVERY]["status"] = WorkerStatus.IDLE
            dispatcher.ask_for_worker(WorkerName.SIGMAREF_DISCOVERY)
            id2 = dispatcher._worker_states[WorkerName.SIGMAREF_DISCOVERY]["current_task_id"]

        assert id1 != id2


class TestTaskDispatcherThread:
    def test_start_creates_thread(self, mock_db: MagicMock) -> None:
        with patch("src.workers.processor.DatabaseService.get_instance", return_value=mock_db):
            dispatcher = TaskDispatcher()
            dispatcher.start()
            time.sleep(0.5)
            assert dispatcher._thread is not None
            assert dispatcher._thread.is_alive()
            dispatcher.stop()

    def test_stop_joins_thread(self, mock_db: MagicMock) -> None:
        with patch("src.workers.processor.DatabaseService.get_instance", return_value=mock_db):
            dispatcher = TaskDispatcher()
            dispatcher.start()
            time.sleep(0.5)
            dispatcher.stop()
            assert not dispatcher._thread.is_alive()

    def test_stop_when_not_started(self, mock_db: MagicMock) -> None:
        with patch("src.workers.processor.DatabaseService.get_instance", return_value=mock_db):
            dispatcher = TaskDispatcher()
            dispatcher.stop()  # Should not raise
