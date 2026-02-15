"""
The Librarian — Ingestion Queue (Phase 8)

Decouples raw message capture (fast, synchronous) from enrichment
(async background workers). Enables 100% ingestion — every message
is stored immediately; extraction, embedding, and categorization
happen in the background.

Architecture:
    ingest() → persist raw stub → enqueue enrichment → return immediately
    [Background workers] → chunk → extract → embed → update DB

Workers pause when a query arrives (retrieval takes priority) and
resume once the search completes.
"""
import asyncio
import uuid
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Awaitable, Any
from datetime import datetime
from enum import Enum

from ..core.types import (
    RolodexEntry, Message, ContentModality, EntryCategory, Tier
)


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionTask:
    """
    A unit of background enrichment work.
    Created when a message is ingested; processed by workers.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message: Optional[Message] = None
    stub_entry_ids: List[str] = field(default_factory=list)
    conversation_id: str = ""
    turn_number: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class IngestionQueue:
    """
    Async queue with background workers for enrichment tasks.

    Usage:
        queue = IngestionQueue(enrichment_fn=agent.process_enrichment_task)
        await queue.start()

        # Fast path: create stub + enqueue
        stub = queue.create_stub_entry(msg, conversation_id)
        await queue.enqueue(task)

        # On query: pause workers, run search, resume
        await queue.pause()
        results = await searcher.search(...)
        await queue.resume()

        # Shutdown
        await queue.shutdown()
    """

    def __init__(
        self,
        enrichment_fn: Optional[Callable[[IngestionTask], Awaitable[None]]] = None,
        num_workers: int = 2,
        max_queue_size: int = 1000,
        pause_on_query: bool = True,
    ):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._enrichment_fn = enrichment_fn
        self._num_workers = num_workers
        self._pause_on_query = pause_on_query

        # Worker management
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._paused = asyncio.Event()
        self._paused.set()  # Start unpaused (set = not paused)

        # Stats
        self._pending_count = 0
        self._processing_count = 0
        self._completed_count = 0
        self._failed_count = 0

    # ─── Lifecycle ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start background workers."""
        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self._num_workers)
        ]

    async def shutdown(self) -> None:
        """Graceful shutdown: finish current tasks, cancel workers."""
        self._running = False
        # Resume if paused so workers can exit
        self._paused.set()
        # Send poison pills
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        # Wait for workers to finish (with timeout)
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    # ─── Pause / Resume ──────────────────────────────────────────────────

    async def pause(self, reason: str = "query") -> None:
        """
        Pause background workers. Workers finish their current task
        then wait until resumed. Non-blocking for the caller.
        """
        if self._pause_on_query:
            self._paused.clear()  # Clear = paused

    async def resume(self) -> None:
        """Resume paused workers."""
        self._paused.set()  # Set = unpaused

    def is_paused(self) -> bool:
        return not self._paused.is_set()

    # ─── Stub Creation ───────────────────────────────────────────────────

    def create_stub_entry(
        self,
        message: Message,
        conversation_id: str,
    ) -> RolodexEntry:
        """
        Create a minimal RolodexEntry stub for immediate storage.
        The stub has content and FTS-searchable text but no embedding
        or enriched categorization yet. That comes from background workers.
        """
        return RolodexEntry(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            content=message.content,
            content_type=ContentModality.CONVERSATIONAL,
            category=EntryCategory.NOTE,
            tags=["pending-enrichment"],
            source_range={"turn_number": message.turn_number},
            embedding=None,  # Will be filled by enrichment
            tier=Tier.COLD,
            metadata={"enrichment_status": "pending"},
        )

    # ─── Enqueue ─────────────────────────────────────────────────────────

    async def enqueue(self, task: IngestionTask) -> None:
        """Add an enrichment task to the queue."""
        self._pending_count += 1
        await self._queue.put(task)

    def enqueue_nowait(self, task: IngestionTask) -> bool:
        """Non-blocking enqueue. Returns False if queue is full."""
        try:
            self._queue.put_nowait(task)
            self._pending_count += 1
            return True
        except asyncio.QueueFull:
            return False

    # ─── Workers ─────────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        """
        Background worker loop.
        Pulls tasks from queue, processes them, respects pause signals.
        """
        while self._running:
            try:
                # Wait if paused
                await self._paused.wait()

                # Get next task (with timeout to check _running flag)
                try:
                    task = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Poison pill for shutdown
                if task is None:
                    break

                # Process the task
                task.status = TaskStatus.PROCESSING
                self._pending_count = max(0, self._pending_count - 1)
                self._processing_count += 1

                try:
                    if self._enrichment_fn:
                        await self._enrichment_fn(task)
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.utcnow()
                    self._completed_count += 1
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    self._failed_count += 1
                finally:
                    self._processing_count = max(0, self._processing_count - 1)
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception:
                # Worker must not die
                continue

    # ─── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return queue statistics."""
        return {
            "enabled": True,
            "running": self._running,
            "paused": self.is_paused(),
            "num_workers": self._num_workers,
            "queue_size": self._queue.qsize(),
            "pending": self._pending_count,
            "processing": self._processing_count,
            "completed": self._completed_count,
            "failed": self._failed_count,
        }

    async def wait_for_drain(self, timeout: float = 30.0) -> bool:
        """
        Wait until all queued tasks are processed.
        Returns True if drained, False if timed out.
        Useful for testing and graceful shutdown.
        """
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
