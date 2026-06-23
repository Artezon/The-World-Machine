import asyncio
import concurrent.futures
import json
import logging
import math
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from protocol import (
    AudioPacketError,
    decode_audio_packet,
    require_positive_int,
)
from scipy.signal import resample_poly
from uvicorn.config import LOGGING_CONFIG

LOGGER = logging.getLogger("app")
SERVER_SAMPLE_RATE = 16000
MIN_REC_SECS = 0.3
MAX_REC_SECS = 30.0

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_HTML = BASE_DIR / "frontend" / "index.html"
DOTENV_PATH = BASE_DIR / ".env"


def load_env():
    if DOTENV_PATH.exists():
        for line in DOTENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


load_env()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8765"))
MODEL = os.getenv("MODEL", "small")
DEVICE = os.getenv("DEVICE", "cuda")
SILENCE_SECS = int(os.getenv("SILENCE_MS", "500")) / 1000.0


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


class VAD:
    def __init__(self):
        self._vad = webrtcvad.Vad(1)

    def is_speech(self, samples_int16):
        if samples_int16 is None or samples_int16.size < 160:
            return False
        frame = 320
        usable = samples_int16.size - (samples_int16.size % frame)
        speech = 0
        checked = 0
        for start in range(0, usable, frame):
            checked += 1
            if self._vad.is_speech(
                samples_int16[start : start + frame].tobytes(),
                SERVER_SAMPLE_RATE,
            ):
                speech += 1
        return checked > 0 and speech / checked >= 0.25


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


# Session (VAD + audio buffering)
class Session:
    def __init__(self, sess_id, lang=""):
        self.sess_id = sess_id
        self.lang = lang
        self.vad = VAD()
        self._lock = threading.Lock()
        self._buf = []
        self._buf_n = 0
        self._rec = False
        self._last_speech = 0.0

    def feed(self, samples_int16):
        now = time.monotonic()
        speech = self.vad.is_speech(samples_int16)
        with self._lock:
            if not self._rec:
                if not speech:
                    return None
                self._rec = True
                self._last_speech = now
                self._buf = [samples_int16.copy()]
                self._buf_n = samples_int16.size
                return None
            self._buf.append(samples_int16.copy())
            self._buf_n += samples_int16.size
            if speech:
                self._last_speech = now
            total = self._buf_n / SERVER_SAMPLE_RATE
            silence = now - self._last_speech
            if total >= MIN_REC_SECS and silence >= SILENCE_SECS:
                return self._finish()
            if total >= MAX_REC_SECS:
                return self._finish()
        return None

    def _finish(self):
        audio = np.concatenate(self._buf).astype(np.float32) / 32768
        self._buf = []
        self._buf_n = 0
        self._rec = False
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
                                "text": "Ready"
                                if ready.is_set()
                                else "Model loading...",
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
