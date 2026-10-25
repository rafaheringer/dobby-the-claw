"""Face recognition for speaker identification.

Built for a robot that moves, watches through a wide lens and sees people from 0.5 m to several
meters. What makes identification trustworthy here:

- Faces are searched *inside each person's own box* (upper body crop), and only a face whose
  center falls in that box counts. The largest face of a loose head crop is often the neighbor's.
- Bad evidence is not evidence: small, blurry or low-confidence faces yield no decision.
- Each person has a gallery of embeddings (several poses/distances), scored by the mean of the
  best matches instead of one blurred average.
- A decision needs an absolute similarity AND, with several people enrolled, a margin over the
  runner-up. Callers (the camera worker) additionally vote across frames per tracked person.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

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


@dataclass(frozen=True)
class FaceObservation:
    """One usable look at a face: who it resembles most and how reliable the look was."""

    name: str  # an enrolled name, or FaceRecognizer.UNKNOWN
    score: float  # similarity to `name` (or to the closest profile when unknown)
    runner_up: str = ""  # second closest profile, for diagnostics


class FaceRecognizer:
    """Enroll and identify speakers by face embedding using InsightFace."""

    UNKNOWN = "visitante"
    # Cosine similarity minimum to accept a match.
    _THRESHOLD = 0.45
    # With several profiles the best must beat the runner-up by this much, else it is ambiguous.
    _MARGIN = 0.08
    # Enrolling a face this close to another profile means it is that person (or a mislabeled frame).
    _DUPLICATE_SIMILARITY = 0.45
    # Evidence quality gates.
    _MIN_DET_SCORE = 0.55
    _MIN_FACE_PX = 36  # face width in the original frame; below this the embedding is mostly noise
    _MIN_SHARPNESS = 25.0  # variance of the Laplacian over the face; motion blur falls under it
    _GALLERY_MAX = 12
    _TOP_K = 3

    def __init__(self, profiles_dir: str) -> None:
        self._dir = Path(profiles_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._app: Optional[object] = None
        self._profiles: dict[str, NDArray] = {}  # name -> (n, 512) gallery of normalized embeddings
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

    # ------------------------------------------------------------------ enrollment

    def enroll(self, name: str, frames: list[NDArray]) -> EnrollResult:
        """Enroll a person from BGR frames (or tight face crops) showing only them."""
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
            if not self._usable(frame, faces[0]):
                continue
            embeddings.append(faces[0].normed_embedding.astype(np.float32))
        if not embeddings:
            logger.warning("No usable face (sharp, large enough) in any enrollment frame for '%s'", name)
            return EnrollResult(error="no_face")
        stacked = np.stack(embeddings)
        if float((stacked @ stacked.T).min()) < 0.4:
            logger.warning("Enrollment of '%s' refused: samples look like different people", name)
            return EnrollResult(error="inconsistent")
        with self._lock:
            for other, gallery in self._profiles.items():
                if other.casefold() == name.casefold():
                    continue
                similarity = float(np.mean([_gallery_score(gallery, e, self._TOP_K) for e in embeddings]))
                if similarity >= self._DUPLICATE_SIMILARITY:
                    logger.warning(
                        "Enrollment of '%s' refused: face already enrolled as '%s' (similarity %.2f)", name, other, similarity
                    )
                    return EnrollResult(error=f"already:{other}")
            # Re-enrolling adds poses/distances to the gallery (older samples drop out first).
            existing = self._profiles.get(name)
            merged = stacked if existing is None else np.concatenate([existing, stacked])
            self._profiles[name] = merged[-self._GALLERY_MAX:].astype(np.float32)
            self._save_profiles_locked()
        logger.info("Enrolled '%s' with %d face samples (gallery %d)", name, len(embeddings), len(self._profiles[name]))
        return EnrollResult(count=len(embeddings))

    # ------------------------------------------------------------------ identification

    def identify(self, frame: NDArray) -> tuple[str, float]:
        """Return (name, similarity) for the largest face in `frame`, or UNKNOWN."""
        result = self.identify_face(frame)
        return result if result is not None else (self.UNKNOWN, 0.0)

    def identify_face(self, frame: NDArray) -> Optional[tuple[str, float]]:
        """Like identify(), but return None when no usable face is found."""
        if not self.available:
            return None
        faces = self._faces(frame)
        face = next((f for f in faces if self._usable(frame, f)), None)
        if face is None:
            return None
        observation = self._match(face.normed_embedding.astype(np.float32))
        return observation.name, observation.score

    def identify_person(self, frame: NDArray, box: tuple[int, int, int, int]) -> Optional[FaceObservation]:
        """Identify the face belonging to one detected person.

        `box` is the person's box in frame pixels (x0, y0, x1, y1). Returns None when that person's
        face is not visible or not good enough to judge: callers must not treat it as "unknown".
        """
        found = self._person_face(frame, box)
        if found is None:
            return None
        return self._match(found[0].normed_embedding.astype(np.float32))

    def face_crop_for_person(self, frame: NDArray, box: tuple[int, int, int, int]) -> Optional[NDArray]:
        """Tight crop around the face that belongs to one person's box (for enrollment), or None."""
        found = self._person_face(frame, box)
        if found is None:
            return None
        face, (ox, oy) = found
        x0, y0, x1, y1 = (float(v) for v in face.bbox)
        margin = 0.35 * max(x1 - x0, y1 - y0)
        h, w = frame.shape[:2]
        left, top = int(max(0, x0 + ox - margin)), int(max(0, y0 + oy - margin))
        right, bottom = int(min(w, x1 + ox + margin)), int(min(h, y1 + oy + margin))
        return frame[top:bottom, left:right] if right > left and bottom > top else None

    def profile_names(self) -> list[str]:
        """Return names of all enrolled profiles."""
        with self._lock:
            return list(self._profiles.keys())

    # ------------------------------------------------------------------ internals

    def _match(self, emb: NDArray) -> FaceObservation:
        with self._lock:
            profiles = dict(self._profiles)
        scores = sorted(((_gallery_score(g, emb, self._TOP_K), name) for name, g in profiles.items()), reverse=True)
        if not scores:
            return FaceObservation(self.UNKNOWN, 0.0)
        best_score, best_name = scores[0]
        runner_up_score, runner_up_name = scores[1] if len(scores) > 1 else (0.0, "")
        if best_score < self._THRESHOLD or (len(scores) > 1 and best_score - runner_up_score < self._MARGIN):
            return FaceObservation(self.UNKNOWN, best_score, best_name)
        return FaceObservation(best_name, best_score, runner_up_name)

    def _person_face(self, frame: NDArray, box: tuple[int, int, int, int]):
        """The usable face inside one person's box: (face, (crop_x, crop_y)) or None."""
        if cv2 is None:
            return None
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = box
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        # Upper body only (the head is in the top of a person box), a bit of margin for box jitter.
        cx0, cx1 = int(max(0, x0 - 0.08 * bw)), int(min(w, x1 + 0.08 * bw))
        cy0, cy1 = int(max(0, y0 - 0.12 * bh)), int(min(h, y0 + 0.62 * bh))
        if cx1 - cx0 < 24 or cy1 - cy0 < 24:
            return None
        crop = frame[cy0:cy1, cx0:cx1]
        faces = self._faces(crop)
        inside = []
        for face in faces:
            fx = (face.bbox[0] + face.bbox[2]) / 2 + cx0
            fy = (face.bbox[1] + face.bbox[3]) / 2 + cy0
            if x0 <= fx <= x1 and cy0 <= fy <= cy1:
                inside.append(face)
        face = next((f for f in inside if self._usable(crop, f)), None)
        return None if face is None else (face, (cx0, cy0))

    def _usable(self, image: NDArray, face) -> bool:
        """Enough pixels, a confident detection and no motion blur."""
        x0, y0, x1, y1 = (int(v) for v in face.bbox)
        if (x1 - x0) < self._MIN_FACE_PX or float(getattr(face, "det_score", 1.0)) < self._MIN_DET_SCORE:
            return False
        if cv2 is None:
            return True
        h, w = image.shape[:2]
        region = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        if region.size == 0:
            return False
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)  # compare sharpness at one scale
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= self._MIN_SHARPNESS

    def _faces(self, frame: NDArray) -> list:
        """Detected faces, largest first."""
        try:
            faces = self._app.get(frame)  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("Face detection failed: %s", exc)
            return []
        return sorted(faces, key=_area, reverse=True)

    def _load_profiles(self) -> None:
        """Load enrolled face profiles from disk (a single vector is a one-sample gallery)."""
        index_path = self._dir / "profiles.json"
        if not index_path.exists():
            return
        try:
            index: dict = json.loads(index_path.read_text(encoding="utf-8"))
            for name, fname in index.items():
                npy = self._dir / fname
                if npy.exists():
                    self._profiles[name] = np.atleast_2d(np.load(str(npy))).astype(np.float32)
            if self._profiles:
                logger.info("Loaded %d face profile(s): %s", len(self._profiles), list(self._profiles.keys()))
        except Exception as exc:
            logger.warning("Failed to load face profiles: %s", exc)

    def _save_profiles_locked(self) -> None:
        """Write enrolled profiles to disk. Must be called with self._lock held."""
        index: dict[str, str] = {}
        for name, gallery in self._profiles.items():
            fname = f"face_{name.lower().replace(' ', '_')}.npy"
            np.save(str(self._dir / fname), gallery)
            index[name] = fname
        (self._dir / "profiles.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _gallery_score(gallery: NDArray, emb: NDArray, top_k: int) -> float:
    """Mean of the `top_k` best similarities to a person's samples (robust to one odd sample)."""
    sims = np.sort(gallery @ emb)[::-1]
    return float(sims[: min(top_k, len(sims))].mean())


def _area(face) -> float:
    return float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
