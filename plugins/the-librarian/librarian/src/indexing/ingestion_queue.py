


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


        self._workers: List[asyncio.Task] = []
        self._running = False
        self._paused = asyncio.Event()
        self._paused.set()


        self._pending_count = 0
        self._processing_count = 0
        self._completed_count = 0
        self._failed_count = 0


    async def start(self) -> None:

        if self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self._num_workers)
        ]

    async def shutdown(self) -> None:

        self._running = False

        self._paused.set()

        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()


    async def pause(self, reason: str = "query") -> None:


        if self._pause_on_query:
            self._paused.clear()

    async def resume(self) -> None:

        self._paused.set()

    def is_paused(self) -> bool:
        return not self._paused.is_set()


    def create_stub_entry(
        self,
        message: Message,
        conversation_id: str,
    ) -> RolodexEntry:


        return RolodexEntry(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            content=message.content,
            content_type=ContentModality.CONVERSATIONAL,
            category=EntryCategory.NOTE,
            tags=["pending-enrichment"],
            source_range={"turn_number": message.turn_number},
            embedding=None,
            tier=Tier.COLD,
            metadata={"enrichment_status": "pending"},
        )


    async def enqueue(self, task: IngestionTask) -> None:

        self._pending_count += 1
        await self._queue.put(task)

    def enqueue_nowait(self, task: IngestionTask) -> bool:

        try:
            self._queue.put_nowait(task)
            self._pending_count += 1
            return True
        except asyncio.QueueFull:
            return False


    async def _worker(self, worker_id: int) -> None:


        while self._running:
            try:

                await self._paused.wait()


                try:
                    task = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue


                if task is None:
                    break


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

                continue


    def get_stats(self) -> Dict[str, Any]:

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


        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
