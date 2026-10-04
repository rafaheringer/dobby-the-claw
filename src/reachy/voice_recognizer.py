"""Voice (speaker embedding) recognition for speaker identification."""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_SHERPA_AVAILABLE = False
try:
    import sherpa_onnx as _sherpa_onnx
    _SHERPA_AVAILABLE = True
except ImportError:
    _sherpa_onnx = None  # type: ignore[assignment]

DEFAULT_MODEL_NAME = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{DEFAULT_MODEL_NAME}"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_RATE = 16000


def default_model_path() -> Path:
    """Return the bundled-on-demand voice model location inside the repo."""
    return _REPO_ROOT / "models" / "speaker" / DEFAULT_MODEL_NAME


def trim_to_voiced(audio: NDArray, sample_rate: int = SAMPLE_RATE) -> NDArray:
    """Drop silent 30 ms frames using an energy gate relative to the loudest frames."""
    frame = int(sample_rate * 0.03)
    usable = (audio.size // frame) * frame
    if usable == 0:
        return audio[:0]
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    loud = float(np.percentile(rms, 90))
    gate = max(0.001, loud * 0.2)
    return frames[rms >= gate].reshape(-1)


class VoiceRecognizer:
    """Identify speakers by voice embedding (sherpa-onnx CAM++), learning profiles over time."""

    UNKNOWN = "visitante"
    _THRESHOLD = 0.50  # cosine similarity minimum to accept a match
    _MIN_VOICED_S = 0.8  # shorter clips give unreliable embeddings
    _MAX_PROFILE_WEIGHT = 50  # cap so profiles keep adapting slowly

    def __init__(self, profiles_dir: str, model_path: str = "") -> None:
        self._dir = Path(profiles_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._extractor: Optional[object] = None
        self._profiles: dict[str, NDArray] = {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._extract_lock = threading.Lock()

        if not _SHERPA_AVAILABLE:
            logger.warning(
                "sherpa-onnx not installed; voice speaker ID disabled. "
                "Install with: pip install sherpa-onnx"
            )
            return

        path = Path(model_path).expanduser() if model_path else default_model_path()
        if not path.is_absolute():
            path = _REPO_ROOT / path
        try:
            if not path.exists():
                if model_path:
                    raise FileNotFoundError(path)
                self._download_model(path)
            config = _sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=2)
            self._extractor = _sherpa_onnx.SpeakerEmbeddingExtractor(config)
            logger.info("VoiceRecognizer ready (%s)", path.name)
        except Exception as exc:
            logger.warning("VoiceRecognizer init failed: %s", exc)

        self._load_profiles()

    @property
    def available(self) -> bool:
        """Return True if voice recognition is operational."""
        return self._extractor is not None

    def embed(self, audio: NDArray) -> Optional[NDArray]:
        """Return a normalized embedding for 16 kHz mono audio, or None if too little speech."""
        if not self.available:
            return None
        audio = np.asarray(audio, dtype=np.float32)
        voiced = trim_to_voiced(audio)
        logger.debug(
            "Voice clip: %.2fs, %.2fs voiced, peak %.4f",
            audio.size / SAMPLE_RATE, voiced.size / SAMPLE_RATE, float(np.abs(audio).max()) if audio.size else 0.0,
        )
        if voiced.size < int(SAMPLE_RATE * self._MIN_VOICED_S):
            return None
        try:
            with self._extract_lock:
                stream = self._extractor.create_stream()  # type: ignore[union-attr]
                stream.accept_waveform(SAMPLE_RATE, voiced)
                stream.input_finished()
                emb = np.array(self._extractor.compute(stream), dtype=np.float32)  # type: ignore[union-attr]
        except Exception as exc:
            logger.debug("Voice embedding failed: %s", exc)
            return None
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else None

    def scores(self, emb: NDArray) -> dict[str, float]:
        """Return cosine similarity of `emb` against every voice profile."""
        with self._lock:
            return {name: float(np.dot(emb, profile)) for name, profile in self._profiles.items()}

    def identify(self, emb: NDArray) -> tuple[str, float]:
        """Return (name, similarity) for the closest voice profile, or UNKNOWN."""
        scores = self.scores(emb)
        if not scores:
            return self.UNKNOWN, 0.0
        name = max(scores, key=scores.__getitem__)
        if scores[name] < self._THRESHOLD:
            return self.UNKNOWN, scores[name]
        return name, scores[name]

    def sample_count(self, name: str) -> int:
        """Return how many utterances have been folded into `name`'s voice profile."""
        with self._lock:
            return self._counts.get(name, 0)

    def add_sample(self, name: str, emb: NDArray) -> None:
        """Fold one utterance embedding into `name`'s profile (running mean)."""
        with self._lock:
            count = self._counts.get(name, 0)
            current = self._profiles.get(name)
            if current is None:
                updated = emb
            else:
                weight = min(count, self._MAX_PROFILE_WEIGHT)
                updated = (current * weight + emb) / (weight + 1)
            norm = np.linalg.norm(updated)
            if norm > 0:
                updated = updated / norm
            self._profiles[name] = updated.astype(np.float32)
            self._counts[name] = count + 1
            self._save_profiles_locked()
        logger.debug("Voice profile '%s' updated (%d samples)", name, count + 1)

    def profile_names(self) -> list[str]:
        """Return names of all voice profiles."""
        with self._lock:
            return list(self._profiles.keys())

    @staticmethod
    def _download_model(path: Path) -> None:
        """Fetch the default CAM++ model from the sherpa-onnx releases."""
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading voice model to %s", path)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(_MODEL_URL, tmp)
        tmp.replace(path)

    def _load_profiles(self) -> None:
        """Load voice profiles from disk."""
        index_path = self._dir / "voice_profiles.json"
        if not index_path.exists():
            return
        try:
            index: dict = json.loads(index_path.read_text(encoding="utf-8"))
            for name, entry in index.items():
                npy = self._dir / entry["file"]
                if npy.exists():
                    self._profiles[name] = np.load(str(npy))
                    self._counts[name] = int(entry.get("samples", 1))
            if self._profiles:
                logger.info("Loaded %d voice profile(s): %s", len(self._profiles), self._counts)
        except Exception as exc:
            logger.warning("Failed to load voice profiles: %s", exc)

    def _save_profiles_locked(self) -> None:
        """Write voice profiles to disk. Must be called with self._lock held."""
        index: dict[str, dict] = {}
        for name, emb in self._profiles.items():
            fname = f"voice_{name.lower().replace(' ', '_')}.npy"
            np.save(str(self._dir / fname), emb)
            index[name] = {"file": fname, "samples": self._counts.get(name, 1)}
        (self._dir / "voice_profiles.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )
