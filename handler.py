"""
RunPod Serverless handler for Coqui XTTS-v2 — Urdu/Pashto (Arabic-script) TTS.

Input (one of):
  { "input": { "text": "اسلام علیکم نیکہ، زہ تیار یم", "language": "ar" } }
  optionally with a base64 reference voice to clone:
  { "input": { "text": "...", "language": "ar", "speaker_wav_base64": "<b64 wav>" } }

Output:
  { "output": { "audio_base64": "<base64 WAV>", "sample_rate": 24000 } }

Notes:
- language defaults to "ar" (Arabic) which is the correct bridge for our
  Urdu rasm-ul-khat text (Pashto written in Urdu script).
- If no speaker_wav is given, a bundled default reference voice is used
  (place a 6-30s clip named `reference.wav` next to this file).
"""
import os
import io
import base64
import tempfile

import runpod
import torch

# Accept Coqui's non-commercial license non-interactively.
os.environ["COQUI_TOS_AGREED"] = "1"

from TTS.api import TTS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Load the model ONCE at cold start (stays warm between requests).
print("Loading XTTS-v2 ...")
tts = TTS(MODEL_NAME).to(DEVICE)
print("XTTS-v2 loaded on", DEVICE)

# Default reference voice bundled in the image (optional but recommended).
DEFAULT_REF = "/reference.wav" if os.path.exists("/reference.wav") else None


def _decode_ref(speaker_wav_base64):
    """Write a base64 reference clip to a temp wav and return its path."""
    if not speaker_wav_base64:
        return None
    data = base64.b64decode(speaker_wav_base64)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.write(data)
    f.close()
    return f.name


def handler(job):
    inp = job.get("input", {}) or {}
    text = (inp.get("text") or "").strip()
    language = (inp.get("language") or "ar").strip()
    speaker_b64 = inp.get("speaker_wav_base64")

    if not text:
        return {"error": "no text provided"}

    ref_path = None
    tmp_ref = None
    out_path = None
    try:
        tmp_ref = _decode_ref(speaker_b64)
        ref_path = tmp_ref or DEFAULT_REF
        if ref_path is None:
            return {"error": "no speaker reference available; include speaker_wav_base64 or bundle reference.wav"}

        out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        tts.tts_to_file(
            text=text,
            speaker_wav=ref_path,
            language=language,
            file_path=out_path,
        )

        with open(out_path, "rb") as fh:
            audio_b64 = base64.b64encode(fh.read()).decode("utf-8")
        return {"audio_base64": audio_b64, "sample_rate": 24000}
    except Exception as e:
        return {"error": str(e)}
    finally:
        for p in (tmp_ref, out_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


runpod.serverless.start({"handler": handler})
