FROM python:3.11-slim

WORKDIR /app

# Build deps (cmake + build-essential needed for insightface on ARM64)
# Runtime libs: OpenMP (onnxruntime), glib/gl (cv2), sndfile + portaudio (audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    pkg-config \
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
    libnice-dev \
    libnice10 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src

ENV PYTHONPATH=/app/src
# Disable GUI debug window — no display in container
ENV VISION_DEBUG_WINDOW=0

CMD ["python", "-m", "bridge.main"]
