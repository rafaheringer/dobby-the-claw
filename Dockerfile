FROM python:3.11-slim

WORKDIR /app

# Build deps (cmake + build-essential needed for insightface on ARM64)
# Runtime libs: OpenMP (onnxruntime), glib/gl (cv2), sndfile + portaudio (audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    pkg-config \
    alsa-utils \
    libopenblas-dev \
    libgomp1 \
    libglib2.0-0 \
    libglib2.0-dev \
    libgl1 \
    libsndfile1 \
    portaudio19-dev \
    libcairo2-dev \
    libgirepository1.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-alsa \
    gstreamer1.0-nice \
    libnice-dev \
    libnice10 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix: VP8 frames have GST_CLOCK_TIME_NONE timestamps via WebRTC; appsink with
# sync=True waits forever for those timestamps → frames never delivered.
RUN sed -i \
    's/self\._appsink_video\.set_property("max-buffers", 1)/self._appsink_video.set_property("max-buffers", 1)\n        self._appsink_video.set_property("sync", False)/' \
    /usr/local/lib/python3.11/site-packages/reachy_mini/media/webrtc_client_gstreamer.py

# Fix: containerized Raspberry Pi deployments expose ALSA devices directly but do
# not provide PulseAudio. Patch Reachy SDK `play_sound()` to fall back to the same
# default sink logic already used by the continuous playback pipeline.
RUN python3 - <<'PY'
from pathlib import Path

path = Path('/usr/local/lib/python3.11/site-packages/reachy_mini/media/audio_gstreamer.py')
content = path.read_text()
old = '''        else:
            id_audio_card = get_audio_device("Sink")
            audiosink = Gst.ElementFactory.make("pulsesink")
            audiosink.set_property("device", f"{id_audio_card}")
            self.logger.info(f"Using audio device {id_audio_card} for playback.")
'''
new = '''        else:
            id_audio_card = get_audio_device("Sink")
            if id_audio_card is None:
                self.logger.warning(
                    "No specific audio card found, using default audio sink for playback."
                )
                audiosink = Gst.ElementFactory.make("autoaudiosink")
            else:
                audiosink = Gst.ElementFactory.make("pulsesink")
                if audiosink is None:
                    self.logger.warning(
                        "pulsesink unavailable, falling back to default audio sink for playback."
                    )
                    audiosink = Gst.ElementFactory.make("autoaudiosink")
                elif audiosink.find_property("device") is not None:
                    audiosink.set_property("device", f"{id_audio_card}")
                self.logger.info(f"Using audio device {id_audio_card} for playback.")
'''
if old not in content:
    raise SystemExit('Expected audio_gstreamer.py block not found')
path.write_text(content.replace(old, new))
PY

# Pre-download openWakeWord built-in models (melspectrogram, embedding, etc.)
RUN python3 -c "import openwakeword; openwakeword.utils.download_models()"

COPY src /app/src
COPY scripts/bridge-entrypoint.sh /app/bridge-entrypoint.sh

RUN chmod +x /app/bridge-entrypoint.sh

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
# Disable GUI debug window — no display in container
ENV VISION_DEBUG_WINDOW=0
ENV REACHY_DIRECT_ALSA_AUDIO=1

ENTRYPOINT ["/app/bridge-entrypoint.sh"]
CMD ["python", "-m", "bridge.main"]
