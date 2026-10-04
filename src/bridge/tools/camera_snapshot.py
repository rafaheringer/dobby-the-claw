"""Camera snapshot tool that exposes the latest frame to the model."""

from __future__ import annotations

import base64
from typing import Any, Dict

import cv2

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult

_MAX_WIDTH = 1280  # full 1080p frames add latency without helping scene descriptions


class CameraSnapshotTool:
    """Capture and encode the latest camera frame for tool responses."""

    def __init__(self, camera_worker) -> None:
        """Store camera worker used to fetch the latest frame."""
        self.camera_worker = camera_worker

    def definition(self) -> ToolDefinition:
        """Return OpenAI function schema for the camera snapshot tool."""
        return ToolDefinition(
            name="camera_snapshot",
            description=(
                "Capture the current camera frame and make it available as an image input for visual understanding. "
                "The scene changes all the time: call this again every time you are asked what you see, "
                "never reuse an earlier image."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the assistant needs the image now.",
                    }
                },
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Capture a frame, encode as JPEG base64, and return metadata."""
        frame = self.camera_worker.get_latest_frame()
        if frame is None:
            return ToolExecutionResult(output={"ok": False, "message": "No camera frame available"})

        if frame.shape[1] > _MAX_WIDTH:
            scale = _MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (_MAX_WIDTH, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return ToolExecutionResult(output={"ok": False, "message": "Failed to encode camera frame"})

        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        reason = str(arguments.get("reason", "")).strip()
        output = {
            "ok": True,
            "message": "Camera snapshot captured",
            "reason": reason,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }
        # Names come from local face recognition so the model never has to identify people from the image.
        get_speaker = getattr(self.camera_worker, "get_current_speaker", None)
        speaker = get_speaker() if callable(get_speaker) else None
        if speaker:
            output["recognized_person"] = speaker
        return ToolExecutionResult(output=output, image_base64=image_b64)
