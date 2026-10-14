"""Person and object detection on COCO classes.

Default backend: YOLOX-tiny (ONNX Runtime, Apache-2.0), which finds people across the room with
the Reachy wide-angle camera. MediaPipe EfficientDet-Lite is kept as a lighter alternative.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reachy.presence import Box

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import onnxruntime as _ort
except ImportError:
    _ort = None  # type: ignore[assignment]

_MP_AVAILABLE = False
try:
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _mp = None  # type: ignore[assignment]

DEFAULT_MODEL_NAME = "yolox_tiny"
_YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{0}.onnx"
_MEDIAPIPE_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/{0}/float16/latest/{0}.tflite"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_MP_INPUT_WIDTH = 640  # EfficientDet works at 320-448 px, so larger frames only cost resize time
_NMS_IOU = 0.45

COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class Detection:
    """One detected object."""

    label: str
    score: float
    box: Box


def model_path_for(name_or_path: str) -> Path:
    """Resolve a model name (yolox_tiny, yolox_s, efficientdet_lite0/2) or a path to an .onnx/.tflite file."""
    value = name_or_path or DEFAULT_MODEL_NAME
    if not value.endswith((".onnx", ".tflite")):
        suffix = ".tflite" if value.startswith("efficientdet") else ".onnx"
        return _REPO_ROOT / "models" / "vision" / f"{value}{suffix}"
    path = Path(value).expanduser()
    return path if path.is_absolute() else _REPO_ROOT / path


class PersonDetector:
    """Detect people (and other COCO objects) in BGR frames."""

    PERSON = "person"

    def __init__(self, model: str = "", person_threshold: float = 0.35, object_threshold: float = 0.5) -> None:
        self._detector = None  # MediaPipe ObjectDetector
        self._session = None  # ONNX Runtime session (YOLOX)
        self._input_name = ""
        self._input_size = 416
        self._grid: NDArray[np.float32] | None = None
        self._strides: NDArray[np.float32] | None = None
        self._lock = threading.Lock()
        self._person_threshold = person_threshold
        self._object_threshold = object_threshold
        if cv2 is None:
            logger.warning("opencv not installed; person detection disabled")
            return

        path = model_path_for(model)
        try:
            if not path.exists():
                if model.endswith((".onnx", ".tflite")):
                    raise FileNotFoundError(path)
                self._download_model(path)
            if path.suffix == ".onnx":
                self._init_yolox(path)
            else:
                self._init_mediapipe(path)
            logger.info("PersonDetector ready (%s)", path.name)
        except Exception as exc:
            self._detector = self._session = None
            logger.warning("PersonDetector init failed: %s", exc)

    def _init_yolox(self, path: Path) -> None:
        if _ort is None:
            raise RuntimeError("onnxruntime not installed")
        options = _ort.SessionOptions()
        options.intra_op_num_threads = 2  # leave CPU for audio, face tracking and face recognition
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")  # idle threads must not hog CPU
        self._session = _ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._input_size = int(model_input.shape[2])
        grids, strides = [], []
        for stride in (8, 16, 32):
            cells = self._input_size // stride
            xs, ys = np.meshgrid(np.arange(cells), np.arange(cells))
            grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
            strides.append(np.full((cells * cells, 1), stride))
        self._grid = np.concatenate(grids).astype(np.float32)
        self._strides = np.concatenate(strides).astype(np.float32)

    def _init_mediapipe(self, path: Path) -> None:
        if not _MP_AVAILABLE:
            raise RuntimeError("mediapipe not installed")
        options = _mp_vision.ObjectDetectorOptions(
            base_options=_mp_python.BaseOptions(model_asset_buffer=path.read_bytes()),
            score_threshold=min(self._person_threshold, self._object_threshold),
            max_results=15,
        )
        self._detector = _mp_vision.ObjectDetector.create_from_options(options)

    @property
    def available(self) -> bool:
        """Return True if detection is operational."""
        return self._session is not None or self._detector is not None

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Return people above the person threshold and other objects above the object threshold."""
        if frame is None or not self.available:
            return []
        try:
            with self._lock:
                if self._session is not None:
                    return self._detect_yolox(frame)
                return self._detect_mediapipe(frame)
        except Exception as exc:
            logger.debug("Person detection failed: %s", exc)
            return []

    def _keep(self, label: str, score: float) -> bool:
        return score >= (self._person_threshold if label == self.PERSON else self._object_threshold)

    def _detect_yolox(self, frame: NDArray[np.uint8]) -> list[Detection]:
        height, width = frame.shape[:2]
        size = self._input_size
        ratio = min(size / height, size / width)
        resized = cv2.resize(frame, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        out = self._session.run(None, {self._input_name: blob})[0][0]  # type: ignore[union-attr]

        class_scores = out[:, 4:5] * out[:, 5:]
        classes = class_scores.argmax(1)
        scores = class_scores[np.arange(len(classes)), classes]
        candidates = scores >= min(self._person_threshold, self._object_threshold)
        if not candidates.any():
            return []
        out, classes, scores = out[candidates], classes[candidates], scores[candidates]
        centers = (out[:, :2] + self._grid[candidates]) * self._strides[candidates]  # type: ignore[index]
        sizes = np.exp(out[:, 2:4]) * self._strides[candidates]  # type: ignore[index]
        corners = np.concatenate((centers - sizes / 2, sizes), axis=1) / ratio  # x, y, w, h in frame pixels

        keep = cv2.dnn.NMSBoxesBatched(corners.tolist(), scores.tolist(), classes.tolist(), 0.0, _NMS_IOU)
        detections = []
        for index in np.asarray(keep).reshape(-1):
            label = COCO_LABELS[int(classes[index])]
            score = float(scores[index])
            if not self._keep(label, score):
                continue
            x, y, w, h = corners[index]
            x, y, w, h = float(x), float(y), float(w), float(h)
            box = Box(max(0.0, x / width), max(0.0, y / height), min(1.0, (x + w) / width), min(1.0, (y + h) / height))
            detections.append(Detection(label, score, box))
        return detections

    def _detect_mediapipe(self, frame: NDArray[np.uint8]) -> list[Detection]:
        height, width = frame.shape[:2]
        if width > _MP_INPUT_WIDTH:
            scale = _MP_INPUT_WIDTH / float(width)
            frame = cv2.resize(frame, (_MP_INPUT_WIDTH, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
            height, width = frame.shape[:2]
        image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._detector.detect(image)  # type: ignore[union-attr]

        detections = []
        for item in result.detections:
            if not item.categories:
                continue
            category = item.categories[0]
            label = category.category_name or ""
            score = float(category.score)
            if not self._keep(label, score):
                continue
            bb = item.bounding_box
            box = Box(
                max(0.0, bb.origin_x / width),
                max(0.0, bb.origin_y / height),
                min(1.0, (bb.origin_x + bb.width) / width),
                min(1.0, (bb.origin_y + bb.height) / height),
            )
            detections.append(Detection(label, score, box))
        return detections

    @staticmethod
    def _download_model(path: Path) -> None:
        """Fetch a YOLOX (GitHub release) or EfficientDet-Lite (MediaPipe storage) model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading person-detection model to %s", path)
        url = (_YOLOX_URL if path.suffix == ".onnx" else _MEDIAPIPE_URL).format(path.stem)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
