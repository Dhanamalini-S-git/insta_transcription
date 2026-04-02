from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import yt_dlp
import whisper
import os
import uuid

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = whisper.load_model("base")

class VideoURL(BaseModel):
    url: str


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/transcribe")
def transcribe_video(data: VideoURL):
    video_url = data.url
    audio_file = f"{uuid.uuid4()}.mp3"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": audio_file.replace(".mp3", ""),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        return JSONResponse({"error": f"Download failed: {str(e)}"})

    try:
        result = model.transcribe(audio_file)
        transcription = result["text"]
    except Exception as e:
        return JSONResponse({"error": f"Transcription failed: {str(e)}"})
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

    return {
        "status": "success",
        "transcription": transcription
    }
