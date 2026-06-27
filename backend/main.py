import asyncio
import concurrent.futures
import json
import logging
import logging.config
import math
import os
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
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
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
SILENCE_MS = int(os.getenv("SILENCE_MS", "500"))
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
LANG_CONFIDENCE_THRESHOLD = float(os.getenv("LANG_CONFIDENCE_THRESHOLD", "0.7"))
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "1") == "1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234").rstrip("/")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "")

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.txt"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text("utf-8").strip()

LMSTUDIO_RESPONSE_IDS = {}


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


class ProviderError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


async def stream_openrouter(messages):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": OPENROUTER_MODEL,
                "messages": messages,
                "stream": True,
                "max_tokens": 1024,
                "reasoning": {"effort": "none"},
            },
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                LOGGER.error("OpenRouter API error %s: %s", response.status_code, body)
                raise ProviderError(response.status_code)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    if not data:
                        continue
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass


async def stream_lm_studio(sess_id, system_prompt, text, reasoning=False):
    url = f"{LMSTUDIO_URL}/api/v1/chat"
    prev_id = LMSTUDIO_RESPONSE_IDS.get(sess_id)

    body = {
        "model": LMSTUDIO_MODEL,
        "input": text,
        "stream": True,
        "max_output_tokens": 1024,
        "store": True,
    }
    if not reasoning:
        body["reasoning"] = "off"
    if system_prompt and not prev_id:
        body["system_prompt"] = system_prompt
    if prev_id:
        body["previous_response_id"] = prev_id

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0), trust_env=False
    ) as client:
        async with client.stream("POST", url, json=body) as response:
            if response.status_code != 200:
                if not reasoning:
                    async for token in stream_lm_studio(
                        sess_id, system_prompt, text, reasoning=True
                    ):
                        yield token
                    return

                body_text = await response.aread()
                LOGGER.error(
                    "LM Studio API error %s: %s", response.status_code, body_text
                )
                raise ProviderError(response.status_code)

            event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:].strip()
                elif line.startswith("data: "):
                    data = json.loads(line[6:].strip())
                    if event == "message.delta":
                        content = data.get("content", "")
                        if content:
                            yield content
                    elif event == "chat.end":
                        rid = data.get("result", {}).get("response_id")
                        if rid:
                            LMSTUDIO_RESPONSE_IDS[sess_id] = rid


def create_app():
    mgr = ConnMgr()
    engine = {"ref": None}
    ready = threading.Event()
    sessions = {}
    chat_histories = {}
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
                return text.strip(), info.language, info.language_probability

            text, detected, prob = await loop.run_in_executor(_executor, run)
            if prob < LANG_CONFIDENCE_THRESHOLD:
                LOGGER.info("Ignoring audio (low language confidence)")
                return "", ""
            return text, detected or lang or ""
        except Exception:
            LOGGER.exception("transcribe failed")
            return "", ""

    async def _preload_lm_studio_model():
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0), trust_env=False
            ) as client:
                r = await client.get(f"{LMSTUDIO_URL}/api/v1/models/")
                if r.status_code == 200:
                    models = r.json().get("models", [])
                    selected = next(
                        (m for m in models if m.get("key") == LMSTUDIO_MODEL), None
                    )
                    if selected and selected.get("loaded_instances"):
                        return

                LOGGER.info("Preloading LM Studio model '%s'...", LMSTUDIO_MODEL)
                r = await client.post(
                    f"{LMSTUDIO_URL}/api/v1/models/load",
                    json={"model": LMSTUDIO_MODEL},
                )
                if r.status_code == 200:
                    LOGGER.info("LM Studio model '%s' loaded", LMSTUDIO_MODEL)
                else:
                    LOGGER.warning(
                        "LM Studio model load returned %s: %s",
                        r.status_code,
                        r.text,
                    )
        except Exception as e:
            LOGGER.warning("Failed to check LM Studio model: %s", e)

    @asynccontextmanager
    async def lifespan(_):
        LOGGER.info("Loading Whisper model...")
        try:
            engine["ref"] = WhisperModel(
                WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="default"
            )
            ready.set()
            LOGGER.info("Whisper model ready")
        except Exception as e:
            LOGGER.exception(f"Failed to load Whisper model: {e}")

        yield
        _executor.shutdown(wait=False)
        ready.clear()

    app = FastAPI(title="World Machine", version="1.0.0", lifespan=lifespan)

    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

    @app.get("/")
    async def index():
        return HTMLResponse(FRONTEND_HTML.read_text("utf-8"))

    async def _check_llm_health() -> bool:
        try:
            if USE_OPENROUTER:
                if not OPENROUTER_API_KEY:
                    return False
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                    r = await client.get(
                        "https://openrouter.ai/api/v1/auth/key",
                        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                    )
            else:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(5.0), trust_env=False
                ) as client:
                    r = await client.get(LMSTUDIO_URL)
            return r.status_code == 200
        except Exception:
            return False

    @app.get("/health")
    async def health():
        return {
            "main": True,
            "stt": ready.is_set(),
            "llm": await _check_llm_health(),
        }

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        sess_id = uuid.uuid4().hex
        await mgr.connect(sess_id, websocket)
        if not USE_OPENROUTER:
            asyncio.create_task(_preload_lm_studio_model())
        session = None
        chat_histories[sess_id] = []
        try:
            while True:
                msg = await websocket.receive()
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if isinstance(data, dict):
                        if "language" in data:
                            session = Session(sess_id, data["language"])
                            sessions[sess_id] = session
                            if ready.is_set():
                                await mgr.send(sess_id, {"type": "ready"})
                        elif data.get("type") == "chat":
                            text = data.get("text", "").strip()
                            if not text:
                                continue
                            if len(text) > 4096:
                                await mgr.send(
                                    sess_id,
                                    {
                                        "type": "error",
                                        "message": "Sorry, we can't deliver such a long message. Please try sending something shorter",
                                    },
                                )
                                continue
                            history = chat_histories.get(sess_id, [])
                            messages = (
                                [{"role": "system", "content": SYSTEM_PROMPT}]
                                + history
                                + [{"role": "user", "content": text}]
                            )
                            history.append({"role": "user", "content": text})
                            full_response = ""
                            provider = "OpenRouter" if USE_OPENROUTER else "LM Studio"
                            try:
                                if USE_OPENROUTER:
                                    async for token in stream_openrouter(messages):
                                        full_response += token
                                        await mgr.send(
                                            sess_id, {"type": "token", "text": token}
                                        )
                                else:
                                    async for token in stream_lm_studio(
                                        sess_id, SYSTEM_PROMPT, text
                                    ):
                                        full_response += token
                                        await mgr.send(
                                            sess_id, {"type": "token", "text": token}
                                        )
                            except ProviderError as e:
                                await mgr.send(
                                    sess_id,
                                    {
                                        "type": "error",
                                        "message": f"{provider} returned {e.status_code}",
                                    },
                                )
                            except httpx.HTTPError as e:
                                LOGGER.error("Connection to %s failed: %s", provider, e)
                                await mgr.send(
                                    sess_id,
                                    {
                                        "type": "error",
                                        "message": "Could not deliver your message to The World Machine. Please try again later",
                                    },
                                )
                            else:
                                history.append(
                                    {"role": "assistant", "content": full_response}
                                )
                                if len(history) > 100:
                                    history[:2] = []
                                chat_histories[sess_id] = history
                                await mgr.send(sess_id, {"type": "done"})
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
                                        "type": "stt",
                                        "text": text,
                                        "lang": lang or "auto",
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
            chat_histories.pop(sess_id, None)
            LMSTUDIO_RESPONSE_IDS.pop(sess_id, None)
            await mgr.disconnect(sess_id)

    return app


if __name__ == "__main__":
    LOGGING_CONFIG["root"] = {"level": "INFO", "handlers": ["default"]}
    logging.config.dictConfig(LOGGING_CONFIG)

    if USE_OPENROUTER and not OPENROUTER_API_KEY:
        LOGGER.error("OPENROUTER_API_KEY is not set, exiting")
        sys.exit(1)
    if not USE_OPENROUTER and not LMSTUDIO_MODEL:
        LOGGER.error("LMSTUDIO_MODEL is not set, exiting")
        sys.exit(1)

    uvicorn.run(
        create_app(), host=HOST, port=PORT, log_level="info", log_config=LOGGING_CONFIG
    )
