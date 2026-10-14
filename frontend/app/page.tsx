"use client";

import { useRef, useState } from "react";

const API = "http://localhost:8000";

export default function Home() {
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [text, setText] = useState("");
  const [original, setOriginal] = useState<string | null>(null);
  const [converted, setConverted] = useState<string | null>(null);
  const [error, setError] = useState("");
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  async function start() {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunks.current = [];
      mr.ondataavailable = (e) => chunks.current.push(e.data);
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks.current, { type: "audio/webm" });
        setOriginal(URL.createObjectURL(blob));
        await transcribe(blob);
      };
      mr.start();
      recorder.current = mr;
      setRecording(true);
    } catch {
      setError("Microphone access denied or unavailable.");
    }
  }

  function stop() {
    recorder.current?.stop();
    setRecording(false);
  }

  async function transcribe(blob: Blob) {
    setLoading(true);
    try {
      const form = new FormData();
      form.append("audio", blob, "recording.webm");
      const res = await fetch(`${API}/transcribe`, { method: "POST", body: form });
      if (!res.ok) throw new Error();
      const data = await res.json();
      setText(data.text);
    } catch {
      setError("Transcription failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  async function speak() {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API}/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error();
      setConverted(URL.createObjectURL(await res.blob()));
    } catch {
      setError("Speech generation failed. Check your OpenAI key.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-xl p-8 space-y-6">
      <h1 className="text-3xl font-bold">VoxBridge</h1>
      <p className="text-gray-500">Record, review the transcript, then hear it re-spoken clearly.</p>

      <button
        onClick={recording ? stop : start}
        disabled={loading}
        className={`rounded-lg px-5 py-3 font-semibold text-white disabled:opacity-50 ${
          recording ? "bg-red-600" : "bg-blue-600"
        }`}
      >
        {recording ? "Stop recording" : "Start recording"}
      </button>

      {original && (
        <div>
          <p className="text-sm font-medium mb-1">Your voice</p>
          <audio controls src={original} className="w-full" />
        </div>
      )}

      {(text || loading) && (
        <div>
          <p className="text-sm font-medium mb-1">Transcript (editable)</p>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            className="w-full rounded-lg border p-3 text-black"
            placeholder={loading ? "Working..." : ""}
          />
          <button
            onClick={speak}
            disabled={loading || !text.trim()}
            className="mt-2 rounded-lg bg-green-600 px-5 py-3 font-semibold text-white disabled:opacity-50"
          >
            Generate clear version
          </button>
        </div>
      )}

      {converted && (
        <div>
          <p className="text-sm font-medium mb-1">Clear version</p>
          <audio controls src={converted} className="w-full" autoPlay />
        </div>
      )}

      {error && <p className="text-red-500">{error}</p>}
    </main>
  );
}