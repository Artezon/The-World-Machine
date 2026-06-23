import argparse
import queue
import sys
from typing import Callable

import numpy as np
import silero_vad
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from silero_vad.utils_vad import VADIterator

SAMPLE_RATE = 16000
BLOCK_SIZE = 512

def _build_transcriber(args) -> Callable[[np.ndarray], str]:
    if args.cpu:
        device = "cpu"
        compute_type = "int8"
        print(f"Loading whisper model '{args.model}' on CPU (faster-whisper)...")
    else:
        device = "cuda"
        compute_type = "float16"
        print(f"Loading whisper model '{args.model}' on GPU (faster-whisper)...")

    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception:
        if device == "cuda":
            print("CUDA not available, falling back to CPU...")
            model = WhisperModel(args.model, device="cpu", compute_type="int8")
        else:
            raise

    lang = args.language if args.language else None

    def transcribe(audio: np.ndarray) -> str:
        segments, _ = model.transcribe(audio, language=lang)
        text = "".join(seg.text for seg in segments)
        return text.strip()

    return transcribe


def main():
    parser = argparse.ArgumentParser(
        description="Real-time speech recognition with AI conversation"
    )
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name")
    parser.add_argument(
        "--language", default=None, help="Language code (default: auto-detect)"
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=500,
        help="Silence duration in ms to end utterance",
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=0.5, help="VAD threshold (0-1)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index (use --list-devices to see options)",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio input devices and exit"
    )

    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU mode (faster-whisper on CPU)"
    )

    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    transcribe_fn = _build_transcriber(args)

    print("Loading VAD model...")

    vad_model = silero_vad.load_silero_vad()
    vad = VADIterator(
        vad_model,
        threshold=args.vad_threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=args.silence_ms,
    )

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"Audio error: {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    stream = sd.InputStream(
        device=args.device,
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=BLOCK_SIZE,
        callback=callback,
        dtype="float32",
    )
    stream.start()

    print(f"\nListening... (speak into your microphone, Ctrl+C to stop)\n")

    ring_buffer = []
    speech_segments = []
    speaking = False

    try:
        while True:
            chunk = audio_queue.get()
            ring_buffer.append(chunk)

            while len(ring_buffer) > 0:
                chunk_samples = (
                    np.concatenate(ring_buffer)
                    if len(ring_buffer) > 1
                    else ring_buffer[0]
                )
                total = len(chunk_samples)
                if total < BLOCK_SIZE:
                    break

                vad_input = chunk_samples[:BLOCK_SIZE]
                remaining = chunk_samples[BLOCK_SIZE:]
                ring_buffer = [remaining] if len(remaining) > 0 else []

                tensor = torch.from_numpy(vad_input)
                result = vad(tensor)

                if result and "start" in result:
                    speaking = True
                    speech_segments = [vad_input.copy()]
                    sys.stdout.write("\r[listening...]")
                    sys.stdout.flush()

                elif result and "end" in result:
                    speaking = False
                    speech_segments.append(vad_input.copy())
                    full_audio = np.concatenate(speech_segments)
                    speech_segments = []

                    sys.stdout.write("\r[transcribing...]")
                    sys.stdout.flush()

                    text = transcribe_fn(full_audio)
                    if text:
                        print(f"\r\033[KYou: {text}")
                    else:
                        print("\r\033[K[...no speech detected...]")
                    sys.stdout.flush()

                elif speaking:
                    speech_segments.append(vad_input.copy())

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
