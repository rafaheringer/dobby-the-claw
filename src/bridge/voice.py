class VoicePipeline:
    def __init__(self, stt_provider: str, tts_provider: str, wake_word: str) -> None:
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider
        self.wake_word = wake_word

    def start_listening(self) -> None:
        # TODO: Integrate wake word + STT.
        raise NotImplementedError("Voice pipeline not implemented yet")

    def speak(self, text: str) -> None:
        # TODO: Integrate TTS.
        raise NotImplementedError("Voice pipeline not implemented yet")
