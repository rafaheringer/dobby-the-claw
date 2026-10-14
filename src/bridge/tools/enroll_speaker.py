"""Tool that lets Dobby enroll a new speaker face profile during conversation."""

from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Optional

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult
from reachy.face_recognizer import FaceRecognizer
from reachy.presence import Track, choose_enroll_track

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

_CHOICE_TTL_S = 120.0  # numbers shown in an ambiguous answer stay valid this long
_IMAGE_MAX_WIDTH = 1280


class EnrollSpeakerTool:
    """Capture camera frames and enroll a face profile for speaker identification."""

    def __init__(self, camera_worker) -> None:
        self._camera_worker = camera_worker
        self._numbers: dict[int, int] = {}  # number shown to the model -> track id
        self._numbers_ts = 0.0

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="enroll_speaker",
            description=(
                "Enrolls a person's face so Dobby can recognize them in future interactions. "
                "Use when asked to remember someone or when talking to an unrecognized person and wanting to learn their name. "
                "target='speaker' enrolls whoever is talking (\"eu sou a Ana\", \"me cadastra\"); "
                "target='other' enrolls someone else being introduced (\"ela é a Jéssica\", \"esse é o meu filho Pedro\"). "
                "When several people could be meant, it returns ambiguous=true with a numbered image of the people "
                "in view: ask which one (position, clothes, hair...), then call again with person_number. "
                "Only for people: never for pets or other animals (their names are just remembered as facts)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name to associate with this person's face.",
                    },
                    "target": {
                        "type": "string",
                        "enum": ["speaker", "other"],
                        "description": "'speaker' = the person talking now; 'other' = someone else being introduced.",
                    },
                    "person_number": {
                        "type": "integer",
                        "description": "Number of the person in the image from a previous ambiguous answer.",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            runtime_guardrail=(
                "Enroll a speaker only after explicitly learning their name — either because the user asked you "
                "to remember them or because you proactively asked an unrecognized person what to call them. "
                "Never guess a name or enroll without consent. Never enroll a pet or animal. "
                "When someone introduces another person, use target='other'. Never pick person_number yourself "
                "from the image alone when the user did not say who it is: ask first."
            ),
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        name = str(arguments.get("name", "")).strip()
        if not name:
            return ToolExecutionResult(output={"ok": False, "message": "Nome não informado."})
        want_speaker = str(arguments.get("target") or "speaker").strip().lower() != "other"
        number = _as_int(arguments.get("person_number"))

        face_recognizer = self._camera_worker._face_recognizer
        if face_recognizer is None or not face_recognizer.available:
            return ToolExecutionResult(
                output={"ok": False, "message": "Reconhecimento facial não disponível."}
            )

        people, speaker_id, frame = self._camera_worker.get_enrollment_view()
        if not people:
            if not want_speaker:
                return _fail(
                    f"Não estou vendo ninguém além de quem fala para cadastrar '{name}'. "
                    "Peça para a pessoa aparecer na frente da câmera e tente de novo."
                )
            # No person detector: fall back to the face right in front of the camera.
            return self._enroll(name, self._capture_frames(count=4, interval_s=0.35), None, voice_from_turn=True)

        track_id = None
        if number is not None:
            if time.time() - self._numbers_ts > _CHOICE_TTL_S:
                self._remember_numbers(people)
            track_id = self._numbers.get(number)
            if track_id is None:
                return _fail(f"Não há pessoa número {number} na imagem. Pergunte de novo quem é.")

        choice = choose_enroll_track(people, want_speaker, speaker_id, _is_known, track_id)
        if choice.track is None:
            if choice.error == "bad_number":
                return _fail(
                    f"A pessoa número {number} saiu da câmera. Peça para ela voltar e tente de novo."
                )
            if choice.error == "no_other":
                return _fail(
                    f"Só vejo quem está falando ou pessoas já conhecidas, não consigo saber quem é '{name}'. "
                    "Peça para a pessoa aparecer na frente da câmera."
                )
            return self._ask_which(name, people, speaker_id, frame)

        track = choice.track
        voice_from_turn = want_speaker and (track.id == speaker_id or len(people) == 1)
        crops = self._camera_worker.capture_track_faces(track.id)
        return self._enroll(name, crops, track, voice_from_turn)

    def _enroll(self, name: str, frames: List, track: Optional[Track], voice_from_turn: bool) -> ToolExecutionResult:
        if not frames:
            return _fail(
                f"Não consegui ver o rosto de '{name}'. "
                "Peça para a pessoa olhar para a câmera, um pouco mais perto, e tente de novo."
            )
        result = self._camera_worker._face_recognizer.enroll(name, frames)
        if result.count > 0:
            if track is not None:
                self._camera_worker.name_track(track.id, name)
            self._numbers = {}
            return ToolExecutionResult(
                output={
                    "ok": True,
                    "message": f"'{name}' cadastrado(a) com sucesso ({result.count} amostras aceitas).",
                    "profiles": self._camera_worker._face_recognizer.profile_names(),
                    # Only the person who just spoke may get this turn's voice sample.
                    "voice_from_turn": voice_from_turn,
                }
            )

        if result.error.startswith("already:"):
            message = (
                f"Não cadastrei: esse rosto já está cadastrado como '{result.error.split(':', 1)[1]}'. "
                "Confirme com a pessoa quem ela é antes de tentar de novo."
            )
        elif result.error in ("several_faces", "inconsistent"):
            message = (
                f"Não cadastrei '{name}': havia outro rosto muito perto do dela. "
                "Peça para ela ficar um pouco afastada dos outros, olhando para você, e tente de novo."
            )
        else:
            message = (
                f"Não detectei um rosto claro para cadastrar '{name}'. "
                "Peça para a pessoa olhar diretamente para a câmera, um pouco mais perto, e tente novamente."
            )
        return _fail(message)

    def _ask_which(self, name: str, people: List[Track], speaker_id: Optional[int], frame) -> ToolExecutionResult:
        self._remember_numbers(people)
        candidates = []
        for number, track in enumerate(people, start=1):
            entry: Dict[str, Any] = {"person_number": number, "position": _position(track)}
            if _is_known(track):
                entry["recognized_as"] = track.name
            if track.id == speaker_id:
                entry["is_speaker"] = True
            candidates.append(entry)
        return ToolExecutionResult(
            output={
                "ok": False,
                "ambiguous": True,
                "message": (
                    f"Não sei qual das pessoas é '{name}'. A imagem mostra cada uma com um número. "
                    "Pergunte quem é (pela posição, roupa ou cabelo) e chame enroll_speaker de novo com person_number."
                ),
                "people": candidates,
            },
            image_base64=_numbered_image(frame, people),
        )

    def _remember_numbers(self, people: List[Track]) -> None:
        self._numbers = {number: track.id for number, track in enumerate(people, start=1)}
        self._numbers_ts = time.time()

    def _capture_frames(self, count: int, interval_s: float) -> List:
        frames = []
        for _ in range(count):
            frame = self._camera_worker.get_latest_frame()
            if frame is not None:
                frames.append(frame)
            time.sleep(interval_s)
        return frames


def _fail(message: str) -> ToolExecutionResult:
    return ToolExecutionResult(output={"ok": False, "message": message})


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _is_known(track: Track) -> bool:
    return bool(track.name) and track.name != FaceRecognizer.UNKNOWN


def _position(track: Track) -> str:
    x = track.box.center_x
    return "esquerda da imagem" if x < 0.38 else "direita da imagem" if x > 0.62 else "centro da imagem"


def _numbered_image(frame, people: List[Track]) -> Optional[str]:
    """JPEG (base64) of the frame with a numbered box around each person."""
    if frame is None or cv2 is None:
        return None
    vis = frame.copy()
    h, w = vis.shape[:2]
    for number, track in enumerate(people, start=1):
        box = track.box
        p0, p1 = (int(box.x0 * w), int(box.y0 * h)), (int(box.x1 * w), int(box.y1 * h))
        cv2.rectangle(vis, p0, p1, (0, 200, 255), 4)
        label = str(number)
        scale = max(1.2, h / 400)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 4)
        cv2.rectangle(vis, p0, (p0[0] + tw + 16, p0[1] + th + 16), (0, 200, 255), -1)
        cv2.putText(vis, label, (p0[0] + 8, p0[1] + th + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    if w > _IMAGE_MAX_WIDTH:
        vis = cv2.resize(vis, (_IMAGE_MAX_WIDTH, int(h * _IMAGE_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(encoded.tobytes()).decode("utf-8") if ok else None
