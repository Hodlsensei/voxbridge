import argparse, asyncio, io, os, queue, tempfile, threading, time, wave
from collections import deque

import numpy as np
import sounddevice as sd
import edge_tts
from dotenv import load_dotenv
from faster_whisper import WhisperModel, decode_audio

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

SR = 16000                 # mic sample rate for Whisper
BLOCK = int(SR * 0.03)     # 30 ms blocks
SILENCE_END = 0.5          # seconds of silence that ends a sentence
MIN_SPEECH = 0.3           # ignore blips shorter than this
MAX_UTT = 15               # force-flush long speech (seconds)
VOICE = "en-US-AriaNeural"
VOICE_RATE = "+10%"        # faster playback = less lag for the caller
HALLUCINATIONS = {"music", "you", "please", "thank you", "thanks for watching", "bye"}

_local = None


def local_model():
    global _local
    if _local is None:
        print("Loading local Whisper (base.en)...")
        _local = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4)
    return _local


def normalize(audio):
    """Boost quiet audio (Bluetooth mics are quiet) before transcribing."""
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        return audio
    return audio * min(0.9 / peak, 10.0)


def to_wav_bytes(audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def find_device(name_part, kind):
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_output_channels"] if kind == "output" else d["max_input_channels"]
        if ch > 0 and name_part.lower() in d["name"].lower():
            return i
    return None


def resolve(arg, kind):
    if arg is None:
        return None
    return int(arg) if arg.isdigit() else find_device(arg, kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="list audio devices and exit")
    p.add_argument("--mic", default=None, help="mic name part or index (default: system mic)")
    p.add_argument("--out", default="CABLE Input", help="output device name part or index")
    p.add_argument("--local", action="store_true", help="force local Whisper, skip Groq")
    args = p.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    mic_dev = resolve(args.mic, "input")
    out_dev = resolve(args.out, "output")
    if args.mic and mic_dev is None:
        print(f"Mic '{args.mic}' not found. Connect the headset, then run: python live.py --list")
        return
    if out_dev is None:
        print(f"Output device '{args.out}' not found. Install VB-Cable, restart Windows, "
              "then run: python live.py --list")
        return
    out_sr = int(sd.query_devices(out_dev)["default_samplerate"])
    print("Mic:", sd.query_devices(mic_dev, "input")["name"] if mic_dev is not None
          else sd.query_devices(kind="input")["name"])
    print("Output:", sd.query_devices(out_dev)["name"], f"@ {out_sr} Hz")

    groq_client = None
    key = os.getenv("GROQ_API_KEY")
    if key and not args.local:
        from groq import Groq
        groq_client = Groq(api_key=key)
        print("Transcription: Groq whisper-large-v3-turbo")
    else:
        local_model()
        print("Transcription: local Whisper base.en")

    def transcribe(audio):
        audio = normalize(audio)
        if groq_client:
            try:
                r = groq_client.audio.transcriptions.create(
                    file=("speech.wav", to_wav_bytes(audio)),
                    model="whisper-large-v3-turbo",
                    language="en",
                    temperature=0.0,
                )
                return r.text.strip()
            except Exception as e:
                print("Groq failed, using local model:", repr(e))
        segs, _ = local_model().transcribe(
            audio, language="en", beam_size=1,
            vad_filter=True, condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segs).strip()

    q = queue.Queue()
    speaking = threading.Event()

    def worker():
        loop = asyncio.new_event_loop()
        while True:
            audio = q.get()
            try:
                t0 = time.time()
                text = transcribe(audio)
                t_stt = time.time() - t0
                if not text or text.lower().strip(" .!?") in HALLUCINATIONS:
                    continue
                print("You said:", text)

                t1 = time.time()
                mp3 = os.path.join(tempfile.gettempdir(), "voxbridge_live.mp3")
                loop.run_until_complete(
                    edge_tts.Communicate(text, VOICE, rate=VOICE_RATE).save(mp3)
                )
                wav = decode_audio(mp3, sampling_rate=out_sr)
                t_tts = time.time() - t1
                print(f"  stt {t_stt:.1f}s + tts {t_tts:.1f}s = "
                      f"caller hears it after {t_stt + t_tts:.1f}s")

                speaking.set()
                sd.play(wav, samplerate=out_sr, device=out_dev)
                sd.wait()
            except Exception as e:
                print("ERROR:", repr(e))
            finally:
                speaking.clear()

    threading.Thread(target=worker, daemon=True).start()

    state = {"calib": 0, "noise": 0.0, "thr": 0.015, "buf": [], "speech": 0, "last": 0.0}
    preroll = deque(maxlen=7)

    def callback(indata, frames, t, status):
        if speaking.is_set():
            return  # don't listen while the synthetic voice is playing

        chunk = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        s = state

        if s["calib"] < 30:  # first ~1s: measure room noise
            s["noise"] += rms
            s["calib"] += 1
            if s["calib"] == 30:
                s["thr"] = min(0.04, max(0.015, s["noise"] / 30 * 3))
                print(f"Ready. Speak in short sentences and pause (threshold {s['thr']:.4f}).")
            return

        now = time.time()
        if rms > s["thr"]:
            if not s["buf"]:
                s["buf"].extend(preroll)
            s["buf"].append(chunk)
            s["speech"] += 1
            s["last"] = now
        elif s["buf"]:
            s["buf"].append(chunk)
            too_long = len(s["buf"]) * BLOCK / SR > MAX_UTT
            if now - s["last"] >= SILENCE_END or too_long:
                if s["speech"] * BLOCK / SR >= MIN_SPEECH:
                    q.put(np.concatenate(s["buf"]).astype(np.float32))
                s["buf"], s["speech"] = [], 0
        else:
            preroll.append(chunk)

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK, device=mic_dev, callback=callback):
        print("Calibrating room noise, stay silent for 2 seconds... (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()