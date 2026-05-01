"""Per-speaker persistent memory using mem0."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from mem0 import Memory as _Mem0Memory
    _MEM0_AVAILABLE = True
except ImportError:
    _Mem0Memory = None  # type: ignore[assignment,misc]
    _MEM0_AVAILABLE = False


class SpeakerMemory:
    """Persist and retrieve per-speaker facts across sessions using mem0."""

    def __init__(self, storage_dir: str, extraction_model: str, api_key: str) -> None:
        self._mem: Optional[object] = None
        if not _MEM0_AVAILABLE:
            logger.warning("mem0ai not installed; speaker memory disabled. pip install mem0ai")
            return
        try:
            os.makedirs(storage_dir, exist_ok=True)
            config = {
                "llm": {
                    "provider": "openai",
                    "config": {"model": extraction_model, "api_key": api_key},
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small", "api_key": api_key},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "dobby_speaker_memories",
                        "path": storage_dir,
                    },
                },
            }
            self._mem = _Mem0Memory.from_config(config)
            logger.info("SpeakerMemory ready (model=%s, dir=%s)", extraction_model, storage_dir)
        except Exception as exc:
            logger.warning("SpeakerMemory init failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._mem is not None

    def load(self, speaker: str) -> str:
        """Return bullet-list of known facts for prompt injection. Empty if none."""
        if not self.available:
            return ""
        try:
            result = self._mem.get_all(user_id=speaker)  # type: ignore[union-attr]
            facts = [m["memory"] for m in (result.get("results") or []) if m.get("memory")]
            if not facts:
                return ""
            return "\n".join(f"- {f}" for f in facts)
        except Exception as exc:
            logger.debug("SpeakerMemory.load failed for '%s': %s", speaker, exc)
            return ""

    def save_async(self, speaker: str, messages: list[dict]) -> None:
        """Extract and persist facts from session messages in a background thread."""
        if not self.available or len(messages) < 2:
            return
        threading.Thread(
            target=self._save, args=(speaker, list(messages)), daemon=True
        ).start()

    def _save(self, speaker: str, messages: list[dict]) -> None:
        try:
            self._mem.add(messages, user_id=speaker)  # type: ignore[union-attr]
            logger.info("SpeakerMemory: saved session for '%s' (%d turns)", speaker, len(messages))
        except Exception as exc:
            logger.warning("SpeakerMemory.save failed for '%s': %s", speaker, exc)
