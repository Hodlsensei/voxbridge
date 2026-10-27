import os, tempfile, glob
import torch
import soundfile as sf
import gradio as gr
import edge_tts
from faster_whisper import WhisperModel, decode_audio

whisper = WhisperModel("base", device="cpu", compute_type="int8")
VOICE = "en-US-AriaNeural"

BASE = os.path.dirname(os.path.abspath(__file__))
REF_FILES = glob.glob(os.path.join(BASE, "reference", "*.wav"))

knn_vc, matching_set = None, None
if REF_FILES:
    print("Loading kNN-VC and your reference voice (slow on first run)...")
    knn_vc = torch.hub.load(
        "bshall/knn-vc", "knn_vc",
        prematched=True, trust_repo=True, pretrained=True, device="cpu",
    )
    refs = [torch.from_numpy(decode_audio(p, sampling_rate=16000)).unsqueeze(0) for p in REF_FILES]
    matching_set = torch.cat([knn_vc.get_features(r) for r in refs], dim=0)
    print("Voice ready.")
else:
    print("No .wav files in backend/reference, using the plain American voice.")


def transcribe(path):
    if not path:
        return ""
    try:
        segments, _ = whisper.transcribe(path, language="en")
        return " ".join(s.text.strip() for s in segments)
    except Exception as e:
        print("TRANSCRIBE ERROR:", repr(e))
        return f"Error: {e}"


async def speak(text):
    if not text.strip():
        return None
    mp3 = os.path.join(tempfile.gettempdir(), "voxbridge_tts.mp3")
    await edge_tts.Communicate(text, VOICE).save(mp3)
    if knn_vc is None:
        return mp3  # no reference recording yet, plain American voice
    wav = torch.from_numpy(decode_audio(mp3, sampling_rate=16000)).unsqueeze(0)
    query = knn_vc.get_features(wav)
    out_wav = knn_vc.match(query, matching_set, topk=4)
    out = os.path.join(tempfile.gettempdir(), "voxbridge_me.wav")
    sf.write(out, out_wav.numpy(), 16000)
    return out


with gr.Blocks(title="VoxBridge") as demo:
    gr.Markdown("# VoxBridge\nRecord, edit the transcript, hear it re-spoken clearly in your own voice.")
    mic = gr.Audio(sources=["microphone"], type="filepath", label="Your voice")
    text = gr.Textbox(label="Transcript (editable)", lines=4)
    btn = gr.Button("Generate clear version")
    out = gr.Audio(label="Clear version", autoplay=True)
    mic.stop_recording(transcribe, mic, text)
    btn.click(speak, text, out)

demo.launch()