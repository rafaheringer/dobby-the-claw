from __future__ import annotations

import base64
from typing import Any, Dict

import cv2

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult


class CameraSnapshotTool:
    def __init__(self, camera_worker) -> None:
        self.camera_worker = camera_worker

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="camera_snapshot",
            description="Capture the latest camera frame and make it available as an image input for visual understanding.",
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
        frame = self.camera_worker.get_latest_frame()
        if frame is None:
            return ToolExecutionResult(output={"ok": False, "message": "No camera frame available"})

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return ToolExecutionResult(output={"ok": False, "message": "Failed to encode camera frame"})

        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        reason = str(arguments.get("reason", "")).strip()
        return ToolExecutionResult(
            output={
                "ok": True,
                "message": "Camera snapshot captured",
                "reason": reason,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            },
            image_base64=image_b64,
        )
