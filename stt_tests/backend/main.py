import asyncio
import concurrent.futures
import json
import logging
import math
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from protocol import (
    AudioPacketError,
    decode_audio_packet,
    require_positive_int,
)
from scipy.signal import resample_poly
from silero_vad import VADIterator, load_silero_vad
from uvicorn.config import LOGGING_CONFIG

LOGGER = logging.getLogger("app")
SERVER_SAMPLE_RATE = 16000
MAX_REC_SECS = 30.0
BLOCK_SIZE = 512

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_HTML = FRONTEND_DIR / "index.html"
ASSETS_DIR = FRONTEND_DIR / "assets"

load_dotenv()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8765"))
MODEL = os.getenv("MODEL", "small")
DEVICE = os.getenv("DEVICE", "cuda")
SILENCE_MS = int(os.getenv("SILENCE_MS", "500"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))


def resample_int16(samples, src_rate, dst_rate):
    if src_rate == dst_rate or samples.size == 0:
        return samples.copy()
    try:
        d = math.gcd(int(src_rate), int(dst_rate))
        resampled = resample_poly(
            samples.astype(np.float32), int(dst_rate // d), int(src_rate // d)
        )
    except Exception:
        duration = samples.size / float(src_rate)
        target = max(1, int(round(duration * dst_rate)))
        src_p = np.linspace(0, 1, samples.size, False)
        dst_p = np.linspace(0, 1, target, False)
        resampled = np.interp(dst_p, src_p, samples.astype(np.float32))
    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)


class ConnMgr:
    def __init__(self):
        self._conns = {}
        self._lock = asyncio.Lock()

    async def connect(self, sess_id, ws):
        await ws.accept()
        async with self._lock:
            self._conns[sess_id] = ws

    async def disconnect(self, sess_id):
        async with self._lock:
            self._conns.pop(sess_id, None)

    async def send(self, sess_id, msg):
        p = json.dumps(msg, separators=(",", ":"))
        async with self._lock:
            ws = self._conns.get(sess_id)
        if ws is None:
            return False
        try:
            await ws.send_text(p)
            return True
        except Exception:
            async with self._lock:
                if self._conns.get(sess_id) is ws:
                    self._conns.pop(sess_id, None)
            return False


class Session:
    _vad_model = None

    def __init__(self, sess_id, lang=""):
        self.sess_id = sess_id
        self.lang = lang
        if Session._vad_model is None:
            Session._vad_model = load_silero_vad()
        self._vad = VADIterator(
            Session._vad_model,
            threshold=VAD_THRESHOLD,
            sampling_rate=SERVER_SAMPLE_RATE,
            min_silence_duration_ms=int(SILENCE_MS),
        )
        self._lock = threading.Lock()
        self._buf = []
        self._buf_n = 0
        self._speaking = False
        self._ring = []

    def feed(self, samples_int16):
        self._ring.append(samples_int16.copy())
        combined = np.concatenate(self._ring) if len(self._ring) > 1 else self._ring[0]
        total = len(combined)

        while total >= BLOCK_SIZE:
            chunk = combined[:BLOCK_SIZE]
            remaining = combined[BLOCK_SIZE:]
            self._ring = [remaining] if len(remaining) > 0 else []
            combined = remaining
            total = len(remaining)

            tensor = torch.from_numpy(chunk).float() / 32768.0
            vad_result = self._vad(tensor)

            if vad_result and "start" in vad_result:
                with self._lock:
                    self._speaking = True
                    self._buf = [chunk.copy()]
                    self._buf_n = chunk.size
            elif vad_result and "end" in vad_result:
                with self._lock:
                    if self._speaking:
                        self._buf.append(chunk.copy())
                        self._buf_n += chunk.size
                        self._speaking = False
                        return self._finish()
            elif self._speaking:
                with self._lock:
                    self._buf.append(chunk.copy())
                    self._buf_n += chunk.size
                    if self._buf_n / SERVER_SAMPLE_RATE >= MAX_REC_SECS:
                        self._speaking = False
                        return self._finish()
        return None

    def _finish(self):
        audio = np.concatenate(self._buf).astype(np.float32) / 32768
        self._buf = []
        self._buf_n = 0
        return audio


def process_packet(packet, max_bytes=524288):
    if len(packet.audio) > max_bytes:
        raise AudioPacketError("packet too large")
    sr = require_positive_int(packet.metadata, "sampleRate")
    ch = packet.metadata.get("channels", 1)
    if not isinstance(ch, int) or isinstance(ch, bool) or ch <= 0:
        raise AudioPacketError("invalid channels")
    if ch > 8:
        raise AudioPacketError("max 8 channels")
    if packet.metadata.get("format", "pcm_s16le") != "pcm_s16le":
        raise AudioPacketError("only pcm_s16le")
    s = np.frombuffer(packet.audio, dtype=np.int16)
    if ch > 1:
        usable = len(s) - (len(s) % ch)
        s = s[:usable].reshape(-1, ch).mean(axis=1).astype(np.int16)
    return resample_int16(s, sr, SERVER_SAMPLE_RATE)


def create_app():
    mgr = ConnMgr()
    engine = {"ref": None}
    ready = threading.Event()
    sessions = {}
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def _transcribe(audio, lang):
        e = engine["ref"]
        if e is None:
            return "", ""
        loop = asyncio.get_event_loop()
        try:

            def run():
                segments, info = e.transcribe(audio, language=lang or None)
                text = " ".join(seg.text for seg in segments)
                return text.strip(), info.language

            text, detected = await loop.run_in_executor(_executor, run)
            return text, detected or lang or ""
        except Exception:
            LOGGER.exception("transcribe failed")
            return "", ""

    @asynccontextmanager
    async def lifespan(_):
        LOGGER.info("Loading model...")
        try:
            engine["ref"] = WhisperModel(MODEL, device=DEVICE, compute_type="default")
            ready.set()
            LOGGER.info("Model ready")
        except Exception as e:
            LOGGER.exception(f"Failed to load model: {e}")
        yield
        _executor.shutdown(wait=False)
        ready.clear()

    app = FastAPI(title="World Machine", version="1.0.0", lifespan=lifespan)

    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

    @app.get("/")
    async def index():
        return HTMLResponse(FRONTEND_HTML.read_text("utf-8"))

    @app.get("/health")
    async def health():
        return {"ok": ready.is_set()}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        sess_id = uuid.uuid4().hex
        await mgr.connect(sess_id, websocket)
        session = None
        try:
            while True:
                msg = await websocket.receive()
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if isinstance(data, dict) and "language" in data:
                        session = Session(sess_id, data["language"])
                        sessions[sess_id] = session
                        await mgr.send(
                            sess_id,
                            {
                                "type": "ready" if ready.is_set() else "status",
                            },
                        )
                elif "bytes" in msg and msg["bytes"]:
                    if session is None:
                        continue
                    try:
                        pkt = decode_audio_packet(msg["bytes"])
                        samples = process_packet(pkt)
                        audio = session.feed(samples)
                        if audio is not None:
                            text, lang = await _transcribe(audio, session.lang)
                            if text:
                                await mgr.send(
                                    sess_id,
                                    {
                                        "type": "final",
                                        "text": text,
                                        "language": lang or "auto",
                                        "sessionId": sess_id,
                                    },
                                )
                    except AudioPacketError as e:
                        await mgr.send(sess_id, {"type": "error", "message": str(e)})
                    except Exception as e:
                        LOGGER.exception("audio error")
                        await mgr.send(sess_id, {"type": "error", "message": str(e)})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sessions.pop(sess_id, None)
            await mgr.disconnect(sess_id)

    return app


if __name__ == "__main__":
    LOGGING_CONFIG["root"] = {"level": "INFO", "handlers": ["default"]}
    uvicorn.run(
        create_app(), host=HOST, port=PORT, log_level="info", log_config=LOGGING_CONFIG
    )
