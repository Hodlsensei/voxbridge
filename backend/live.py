import argparse, asyncio, os, queue, tempfile, threading, time
from collections import deque

import numpy as np
import sounddevice as sd
import edge_tts
from faster_whisper import WhisperModel, decode_audio

SR = 16000                 # mic sample rate for Whisper
BLOCK = int(SR * 0.03)     # 30 ms blocks
SILENCE_END = 0.7          # seconds of silence that ends a sentence
MIN_SPEECH = 0.3           # ignore blips shorter than this
MAX_UTT = 15               # force-flush long speech (seconds)
VOICE = "en-US-AriaNeural"
HALLUCINATIONS = {"music", "you", "please", "thank you", "thanks for watching", "bye"}


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
    args = p.parse_args()

    if args.list:
        print(sd.query_devices())
        return

    mic_dev = resolve(args.mic, "input")
    out_dev = resolve(args.out, "output")
    if out_dev is None:
        print(f"Output device '{args.out}' not found. Install VB-Cable, restart Windows, "
              "then run: python live.py --list")
        return
    out_sr = int(sd.query_devices(out_dev)["default_samplerate"])
    print("Mic:", sd.query_devices(mic_dev, "input")["name"] if mic_dev is not None
          else sd.query_devices(kind="input")["name"])
    print("Output:", sd.query_devices(out_dev)["name"], f"@ {out_sr} Hz")

    print("Loading Whisper...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4)

    q = queue.Queue()
    speaking = threading.Event()

    def worker():
        while True:
            audio = q.get()
            try:
                t0 = time.time()
                segs, _ = model.transcribe(
                    audio, language="en", beam_size=1,
                    vad_filter=True, condition_on_previous_text=False,
                )
                text = " ".join(s.text.strip() for s in segs).strip()
                t_stt = time.time() - t0
                if not text or text.lower().strip(" .!?") in HALLUCINATIONS:
                    continue
                print("You said:", text)

                t1 = time.time()
                mp3 = os.path.join(tempfile.gettempdir(), "voxbridge_live.mp3")
                asyncio.run(edge_tts.Communicate(text, VOICE).save(mp3))
                wav = decode_audio(mp3, sampling_rate=out_sr)
                t_tts = time.time() - t1
                print(f"  whisper {t_stt:.1f}s + tts {t_tts:.1f}s = "
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
                s["thr"] = max(0.015, s["noise"] / 30 * 4)
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
        print("Calibrating room noise, stay quiet for 1 second... (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()  