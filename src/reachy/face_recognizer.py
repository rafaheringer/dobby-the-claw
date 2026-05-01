"""Face recognition for speaker identification."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_INSIGHTFACE_AVAILABLE = False
try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _FaceAnalysis = None  # type: ignore[assignment,misc]


class FaceRecognizer:
    """Enroll and identify speakers by face embedding using InsightFace."""

    UNKNOWN = "visitante"
    _THRESHOLD = 0.40  # cosine similarity minimum to accept a match

    def __init__(self, profiles_dir: str) -> None:
        self._dir = Path(profiles_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._app: Optional[object] = None
        self._profiles: dict[str, NDArray] = {}
        self._lock = threading.Lock()

        if not _INSIGHTFACE_AVAILABLE:
            logger.warning(
                "insightface not installed; speaker face ID disabled. "
                "Install with: pip install insightface onnxruntime"
            )
            return

        try:
            app = _FaceAnalysis(
                name="buffalo_s",
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_size=(320, 320))
            self._app = app
            logger.info("FaceRecognizer ready (buffalo_s, CPU)")
        except Exception as exc:
            logger.warning("FaceRecognizer init failed: %s", exc)

        self._load_profiles()

    @property
    def available(self) -> bool:
        """Return True if face recognition is operational."""
        return self._app is not None

    def enroll(self, name: str, frames: list[NDArray]) -> int:
        """Enroll a person from BGR frames. Returns count of accepted samples."""
        if not self.available:
            return 0
        embeddings = [e for e in (self._embed(f) for f in frames) if e is not None]
        if not embeddings:
            logger.warning("No face detected in any enrollment frame for '%s'", name)
            return 0
        mean = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        with self._lock:
            self._profiles[name] = mean.astype(np.float32)
            self._save_profiles_locked()
        logger.info("Enrolled '%s' with %d face samples", name, len(embeddings))
        return len(embeddings)

    def identify(self, frame: NDArray) -> tuple[str, float]:
        """Return (name, similarity) for closest enrolled profile, or UNKNOWN."""
        if not self.available:
            return self.UNKNOWN, 0.0
        with self._lock:
            if not self._profiles:
                return self.UNKNOWN, 0.0
            profiles_snapshot = dict(self._profiles)

        emb = self._embed(frame)
        if emb is None:
            return self.UNKNOWN, 0.0

        best_name, best_score = self.UNKNOWN, 0.0
        for name, profile in profiles_snapshot.items():
            score = float(np.dot(emb, profile))
            if score > best_score:
                best_score, best_name = score, name

        if best_score < self._THRESHOLD:
            return self.UNKNOWN, best_score
        return best_name, best_score

    def profile_names(self) -> list[str]:
        """Return names of all enrolled profiles."""
        with self._lock:
            return list(self._profiles.keys())

    def _embed(self, frame: NDArray) -> Optional[NDArray]:
        """Extract normalized face embedding from a BGR frame, or None if no face."""
        try:
            faces = self._app.get(frame)  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("Face embedding failed: %s", exc)
            return None
        if not faces:
            return None
        # Use the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.normed_embedding.astype(np.float32)
        return emb

    def _load_profiles(self) -> None:
        """Load enrolled face profiles from disk."""
        index_path = self._dir / "profiles.json"
        if not index_path.exists():
            return
        try:
            index: dict = json.loads(index_path.read_text(encoding="utf-8"))
            loaded = 0
            for name, fname in index.items():
                npy = self._dir / fname
                if npy.exists():
                    self._profiles[name] = np.load(str(npy))
                    loaded += 1
            if loaded:
                logger.info(
                    "Loaded %d face profile(s): %s", loaded, list(self._profiles.keys())
                )
        except Exception as exc:
            logger.warning("Failed to load face profiles: %s", exc)

    def _save_profiles_locked(self) -> None:
        """Write enrolled profiles to disk. Must be called with self._lock held."""
        index: dict[str, str] = {}
        for name, emb in self._profiles.items():
            fname = f"face_{name.lower().replace(' ', '_')}.npy"
            np.save(str(self._dir / fname), emb)
            index[name] = fname
        (self._dir / "profiles.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )
