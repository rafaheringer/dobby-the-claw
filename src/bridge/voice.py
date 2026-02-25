import logging
import os
import tempfile
import time
import wave

import requests

try:
    from reachy_mini import ReachyMini
except ImportError:  # pragma: no cover - optional SDK
    ReachyMini = None


class VoicePipeline:
    def __init__(
        self,
        stt_provider: str,
        tts_provider: str,
        wake_word: str,
        tts_api_base: str,
        tts_api_key_env: str,
        tts_model: str,
        tts_voice: str,
        tts_format: str,
        tts_output: str,
        reachy_instance=None,
        speech_animator=None,
    ) -> None:
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider
        self.wake_word = wake_word
        self.tts_api_base = tts_api_base.rstrip("/")
        self.tts_api_key_env = tts_api_key_env
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.tts_format = tts_format
        self.tts_output = tts_output
        self._reachy_mini = ReachyMini
        self._reachy_instance = reachy_instance
        self._speech_animator = speech_animator

    def start_listening(self) -> None:
        # TODO: Integrate wake word + STT.
        raise NotImplementedError("Voice pipeline not implemented yet")

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return

        if self._speech_animator is not None:
            self._speech_animator.start_speaking()

        if self.tts_provider != "openai":
            print(f"Reachy: {text}")
            if self._speech_animator is not None:
                self._speech_animator.stop_speaking()
            return

        api_key = os.getenv(self.tts_api_key_env, "").strip()
        if not api_key:
            print(f"Reachy: {text}")
            if self._speech_animator is not None:
                self._speech_animator.stop_speaking()
            return

        try:
            audio = self._synthesize_openai(text, api_key)
            if self.tts_output == "reachy":
                if self._reachy_mini is None:
                    print(f"Reachy: {text}")
                else:
                    try:
                        self._play_on_reachy(audio)
                    except Exception:
                        print(f"Reachy: {text}")
                return

            print(f"Reachy: {text}")
        finally:
            if self._speech_animator is not None:
                self._speech_animator.stop_speaking()

    def _synthesize_openai(self, text: str, api_key: str) -> bytes:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.tts_model,
            "input": text,
            "voice": self.tts_voice,
            "format": self.tts_format,
        }
        response = requests.post(
            f"{self.tts_api_base}/audio/speech",
            headers=headers,
            json=payload,
            timeout=20,
        )
        if not response.ok:
            raise RuntimeError(
                f"TTS request failed: {response.status_code} {response.text}"
            )
        return response.content

    def _play_on_reachy(self, audio: bytes) -> None:
        suffix = f".{self.tts_format.lower()}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio)
            temp_path = temp_file.name

        try:
            mini = self._get_reachy_instance()
            if mini is None:
                return
            mini.media.play_sound(temp_path)
            duration_s = self._get_wav_duration(temp_path)
            if duration_s is not None:
                time.sleep(duration_s + 0.6)
                mini.media.stop_playing()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _get_wav_duration(self, path: str) -> float | None:
        if self.tts_format.lower() != "wav":
            return None
        try:
            with wave.open(path, "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
        except wave.Error:
            return None

    def _get_reachy_instance(self):
        if self._reachy_instance is not None:
            return self._reachy_instance
        if self._reachy_mini is None:
            return None
        try:
            self._reachy_instance = self._reachy_mini()
        except Exception as exc:
            logging.warning("Failed to connect Reachy Mini: %s", exc)
            return None
        return self._reachy_instance
