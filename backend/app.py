import os, tempfile
import gradio as gr
import edge_tts
from faster_whisper import WhisperModel

whisper = WhisperModel("base", device="cpu", compute_type="int8")
VOICE = "en-US-AriaNeural"


def transcribe(path):
    if not path:
        return ""
    segments, _ = whisper.transcribe(path, language="en")
    return " ".join(s.text.strip() for s in segments)


async def speak(text):
    if not text.strip():
        return None
    out = os.path.join(tempfile.gettempdir(), "voxbridge_out.mp3")
    await edge_tts.Communicate(text, VOICE).save(out)
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