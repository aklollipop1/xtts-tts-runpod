FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System deps: ffmpeg for audio I/O.
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Coqui TTS + runpod. Pin a compatible transformers (XTTS is sensitive to it).
RUN pip install --no-cache-dir \
    "coqui-tts==0.24.1" \
    "transformers>=4.40,<4.46" \
    runpod

# Accept the Coqui model license at build time so download is non-interactive.
ENV COQUI_TOS_AGREED=1

# Pre-download the XTTS-v2 model into the image so cold starts are fast.
RUN python -c "import os; os.environ['COQUI_TOS_AGREED']='1'; from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# OPTIONAL: bundle a default reference voice. Drop a 6-30s clip named
# reference.wav into the repo and uncomment the next line.
# COPY reference.wav /reference.wav

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
