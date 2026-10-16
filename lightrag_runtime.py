"""Synchronous facade that keeps LightRAG on one owned asyncio event loop."""

from __future__ import annotations

import asyncio
import json
import re
from concurrent.futures import Future
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any, Callable

from lightrag import QueryParam


_DISEASE_NAME_PATTERN = re.compile(r"^疾病名称：([^\r\n]+)", re.MULTILINE)


def find_exact_disease_context(query: str, chunks: list[str]) -> str | None:
    """Return the imported chunk whose disease name appears verbatim in a query."""
    matches = []
    for chunk in chunks:
        disease_match = _DISEASE_NAME_PATTERN.search(chunk)
        if disease_match and disease_match.group(1) in query:
            matches.append((disease_match.group(1), chunk))
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]


def compact_context(context: str | None, max_characters: int = 3000) -> str:
    """Keep fallback retrieval within the answer model's effective context window."""
    return (context or "")[:max_characters]


def _load_disease_chunks(working_dir: str) -> list[str]:
    path = Path(working_dir) / "kv_store_text_chunks.json"
    if not path.exists():
        return []
    raw_chunks = json.loads(path.read_text(encoding="utf-8"))
    return [
        record["content"]
        for record in raw_chunks.values()
        if isinstance(record, dict)
        and isinstance(record.get("content"), str)
        and record["content"].startswith("疾病名称：")
    ]


class AsyncLightRAGClient:
    """Run LightRAG initialization and async queries on one background loop."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._loop = asyncio.new_event_loop()
        self._ready = Event()
        self._closed = False
        self._lock = RLock()
        self._thread = Thread(target=self._run_loop, name="lightrag-runtime", daemon=True)
        self._thread.start()
        self._ready.wait()
        self._submit(self._initialize()).result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _submit(self, coroutine) -> Future:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    async def _initialize(self) -> None:
        self._rag = self._factory()
        await self._rag.initialize_storages()
        working_dir = getattr(self._rag, "working_dir", None)
        self._disease_chunks = _load_disease_chunks(working_dir) if working_dir else []

    async def _query_local_context(self, query: str) -> str:
        exact_context = find_exact_disease_context(query, self._disease_chunks)
        if exact_context:
            return exact_context

        context = await self._rag.aquery(
            query,
            param=QueryParam(
                mode="local",
                only_need_context=True,
                top_k=8,
                chunk_top_k=2,
                max_entity_tokens=1000,
                max_relation_tokens=1000,
                max_total_tokens=3000,
            ),
        )
        return compact_context(context)

    def query_local_context(self, query: str) -> str:
        """Synchronously obtain local graph context through LightRAG.aquery()."""
        with self._lock:
            if self._closed:
                raise RuntimeError("LightRAG runtime is closed.")
            return self._submit(self._query_local_context(query)).result()

    async def _finalize(self) -> None:
        await self._rag.finalize_storages()

    async def _shutdown(self) -> None:
        await self._finalize()
        current_task = asyncio.current_task()
        pending = [
            task
            for task in asyncio.all_tasks()
            if task is not current_task and not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._submit(self._shutdown()).result()
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()
