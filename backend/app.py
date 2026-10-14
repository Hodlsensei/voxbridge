import os, tempfile
import gradio as gr
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI

load_dotenv()
whisper = WhisperModel("base", device="cpu", compute_type="int8")
client = OpenAI()


def transcribe(path):
    if not path:
        return ""
    segments, _ = whisper.transcribe(path, language="en")
    return " ".join(s.text.strip() for s in segments)


def speak(text):
    if not text.strip():
        return None
    out = os.path.join(tempfile.gettempdir(), "voxbridge_out.mp3")
    res = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
    with open(out, "wb") as f:
        f.write(res.read())
    return out


with gr.Blocks(title="VoxBridge") as demo:
    gr.Markdown("# VoxBridge\nRecord, edit the transcript, hear it re-spoken clearly.")
    mic = gr.Audio(sources=["microphone"], type="filepath", label="Your voice")
    text = gr.Textbox(label="Transcript (editable)", lines=4)
    btn = gr.Button("Generate clear version")
    out = gr.Audio(label="Clear version", autoplay=True)
    mic.stop_recording(transcribe, mic, text)
    btn.click(speak, text, out)

demo.launch()