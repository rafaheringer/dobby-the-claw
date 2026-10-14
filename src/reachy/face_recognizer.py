"""Face recognition for speaker identification."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_INSIGHTFACE_AVAILABLE = False
try:
    from insightface.app import FaceAnalysis as _FaceAnalysis
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _FaceAnalysis = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class EnrollResult:
    """Outcome of an enrollment: accepted samples, or why nothing was saved."""

    count: int = 0
    error: str = ""  # "no_face", "several_faces", "inconsistent" or "already:<name>"


class FaceRecognizer:
    """Enroll and identify speakers by face embedding using InsightFace."""

    UNKNOWN = "visitante"
    # Cosine similarity minimum to accept a match. Distant (~35 px) faces of the same person still
    # score 0.3-0.8, while different people stay below ~0.2.
    _THRESHOLD = 0.35
    # The best profile must beat the runner-up by this much, otherwise the face is ambiguous.
    _MARGIN = 0.10
    # Enrolling a face this close to another profile means it is that person (or a mislabeled frame).
    _DUPLICATE_SIMILARITY = 0.45

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
            # 640 keeps faces a few meters away detectable (320 misses faces under ~40 px).
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._app = app
            logger.info("FaceRecognizer ready (buffalo_s, CPU)")
        except Exception as exc:
            logger.warning("FaceRecognizer init failed: %s", exc)

        self._load_profiles()

    @property
    def available(self) -> bool:
        """Return True if face recognition is operational."""
        return self._app is not None

    def enroll(self, name: str, frames: list[NDArray]) -> EnrollResult:
        """Enroll a person from BGR frames showing only them up close."""
        if not self.available:
            return EnrollResult(error="no_face")
        embeddings = []
        for frame in frames:
            faces = self._faces(frame)
            if not faces:
                continue
            # Someone else just as close could be the one being named: refuse rather than guess.
            if len(faces) > 1 and _area(faces[1]) >= 0.5 * _area(faces[0]):
                logger.warning("Enrollment of '%s' refused: several faces in view", name)
                return EnrollResult(error="several_faces")
            embeddings.append(faces[0].normed_embedding.astype(np.float32))
        if not embeddings:
            logger.warning("No face detected in any enrollment frame for '%s'", name)
            return EnrollResult(error="no_face")
        stacked = np.stack(embeddings)
        if float((stacked @ stacked.T).min()) < 0.4:
            logger.warning("Enrollment of '%s' refused: samples look like different people", name)
            return EnrollResult(error="inconsistent")
        mean = stacked.mean(axis=0)
        mean = mean / max(float(np.linalg.norm(mean)), 1e-6)
        with self._lock:
            for other, profile in self._profiles.items():
                if other.casefold() != name.casefold() and float(np.dot(mean, profile)) >= self._DUPLICATE_SIMILARITY:
                    logger.warning("Enrollment of '%s' refused: face already enrolled as '%s'", name, other)
                    return EnrollResult(error=f"already:{other}")
            self._profiles[name] = mean.astype(np.float32)
            self._save_profiles_locked()
        logger.info("Enrolled '%s' with %d face samples", name, len(embeddings))
        return EnrollResult(count=len(embeddings))

    def identify(self, frame: NDArray) -> tuple[str, float]:
        """Return (name, similarity) for closest enrolled profile, or UNKNOWN."""
        return self.identify_face(frame) or (self.UNKNOWN, 0.0)

    def identify_face(self, frame: NDArray) -> Optional[tuple[str, float]]:
        """Like identify(), but return None when no face is found at all."""
        if not self.available:
            return None
        emb = self._embed(frame)
        if emb is None:
            return None
        with self._lock:
            profiles_snapshot = dict(self._profiles)

        scores = sorted(((float(np.dot(emb, profile)), name) for name, profile in profiles_snapshot.items()), reverse=True)
        if not scores:
            return self.UNKNOWN, 0.0
        best_score, best_name = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        if best_score < self._THRESHOLD or best_score - runner_up < self._MARGIN:
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
        face = max(faces, key=_area)
        emb = face.normed_embedding.astype(np.float32)
        return emb

    def _faces(self, frame: NDArray) -> list:
        """Detected faces, largest first."""
        try:
            faces = self._app.get(frame)  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("Face detection failed: %s", exc)
            return []
        return sorted(faces, key=_area, reverse=True)

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


def _area(face) -> float:
    return float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
