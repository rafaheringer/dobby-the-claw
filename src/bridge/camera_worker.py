import logging
import threading
import time
from typing import Optional, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


class CameraWorker:
    def __init__(self, reachy_mini, poll_interval_s: float = 0.1) -> None:
        self._reachy_mini = reachy_mini
        self._poll_interval_s = poll_interval_s
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame = None
        self._face_center: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()
        self._face_detector = None
        if cv2 is not None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_detector = cv2.CascadeClassifier(cascade_path)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=2.0)
        self._thread = None

    def get_face_center(self) -> Optional[Tuple[int, int]]:
        with self._lock:
            return self._face_center

    def get_latest_frame(self):
        with self._lock:
            return self._frame

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._reachy_mini.media.get_frame()
            except Exception as exc:
                logging.debug("Camera frame fetch failed: %s", exc)
                time.sleep(self._poll_interval_s)
                continue

            if frame is None:
                time.sleep(self._poll_interval_s)
                continue

            face_center = self._detect_face_center(frame)
            with self._lock:
                self._frame = frame
                self._face_center = face_center

            time.sleep(self._poll_interval_s)

    def _detect_face_center(self, frame):
        if self._face_detector is None or cv2 is None:
            return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_detector.detectMultiScale(gray, 1.3, 5)
        except Exception as exc:
            logging.debug("Face detection failed: %s", exc)
            return None

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (int(x + w / 2), int(y + h / 2))
