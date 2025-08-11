import os
import subprocess
from pathlib import Path
import json

try:
    import whisper
except Exception:
    whisper = None

try:
    import openai
except Exception:
    openai = None

try:
    from openai import OpenAI as OpenAI_Client
except Exception:
    OpenAI_Client = None

def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc.stdout.decode(errors='ignore')

def ensure_ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and add to PATH.")

def extract_audio(video_path: str, out_audio: str):
    ensure_ffmpeg_available()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", out_audio
    ]
    run_cmd(cmd)
    return out_audio

def transcribe_with_local_whisper(audio_path: str, model_size: str = "base"):
    if whisper is None:
        raise RuntimeError("whisper not installed")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result

def transcribe_with_openai_api(audio_path: str, api_key: str = None):
    if openai is not None:
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise RuntimeError("OPENAI_API_KEY not set")
        with open(audio_path, "rb") as af:
            resp = openai.Audio.transcriptions.create(
                file=af,
                model="whisper-1"
            )
        return {"text": resp.get("text")}
    if OpenAI_Client is not None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI_Client(api_key=api_key)
        with open(audio_path, "rb") as af:
            resp = client.audio.transcriptions.create(model="whisper-1", file=af)
        return {"text": resp.text}
    raise RuntimeError("No OpenAI client available")

def process_video(video_path: str, outdir: str, use_local_whisper=True, whisper_model="base", api_key=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    audio_path = str(outdir / "extracted_audio.wav")
    extract_audio(video_path, audio_path)

    transcript = None
    if use_local_whisper and whisper is not None:
        whisper_result = transcribe_with_local_whisper(audio_path, model_size=whisper_model)
        transcript = whisper_result.get("text")
    else:
        resp = transcribe_with_openai_api(audio_path, api_key=api_key)
        transcript = resp.get("text")

    if transcript is None:
        raise RuntimeError("Failed to get transcript")

    transcript_file = outdir / "transcript.txt"
    transcript_file.write_text(transcript, encoding="utf-8")

    # Optionally generate summary here (omitted for brevity)

    report = {
        "video": str(video_path),
        "audio": audio_path,
        "transcript_file": str(transcript_file),
        "summary_file": None,
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report
