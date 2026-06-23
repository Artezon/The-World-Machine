const WS_URL = (() => {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return proto + "//" + location.host + "/ws";
})();

const ui = {
  micBtn: document.getElementById("micBtn"),
  langSelect: document.getElementById("langSelect"),
  clearBtn: document.getElementById("clearBtn"),
  transcriptArea: document.getElementById("transcriptArea"),
  realtimeDisplay: document.getElementById("realtimeDisplay"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
  meterFill: document.getElementById("meterFill"),
};

let currentLevel = 0;

let ws = null;
let audioContext = null;
let stream = null;
let processor = null;
let source = null;
let isRecording = false;
let audioQueue = [];
let flushInterval = null;
let manualDisconnect = false;

function setConnected(connected) {
  ui.statusDot.className = "status-dot" + (connected ? " connected" : "");
  ui.statusText.textContent = connected ? "Connected" : "Disconnected";
  ui.micBtn.disabled = !connected;
  if (!connected && isRecording) stopRecording();
}

function setRecording(state) {
  isRecording = state;
  ui.micBtn.classList.toggle("active", state);
  if (state) {
    ui.statusDot.className = "status-dot recording";
    ui.statusText.textContent = "Recording";
  } else {
    ui.statusDot.className = "status-dot connected";
    ui.statusText.textContent = "Connected";
  }
}

function connect() {
  if (ws) {
    try {
      ws.close();
    } catch (_) {}
    ws = null;
  }
  manualDisconnect = false;
  ws = new WebSocket(WS_URL);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    ws.send(JSON.stringify({ language: ui.langSelect.value }));
    ui.statusDot.className = "status-dot";
    ui.statusText.textContent = "Connecting...";
  };

  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === "ready") {
        setConnected(true);
      } else if (msg.type === "status") {
        ui.statusText.textContent = msg.text;
      } else if (msg.type === "final") {
        ui.realtimeDisplay.textContent = "";
        addEntry(msg.text, msg.language);
      }
    } catch (_) {}
  };

  ws.onclose = () => {
    setConnected(false);
    ws = null;
    if (!manualDisconnect) setTimeout(connect, 1000);
  };

  ws.onerror = () => {
    ws && ws.close();
  };
}

const LANG_MAP = {};
ui.langSelect.querySelectorAll("option").forEach((opt) => {
  LANG_MAP[opt.value] = opt.textContent;
});

function addEntry(text, language) {
  const div = document.createElement("div");
  div.className = "entry final";
  const label = document.createElement("div");
  label.className = "label";
  label.textContent = LANG_MAP[language] || language || "Auto-detect";
  const content = document.createElement("div");
  content.className = "text";
  content.textContent = text;
  div.append(label, content);
  ui.transcriptArea.append(div);
  ui.transcriptArea.scrollTop = ui.transcriptArea.scrollHeight;
}

function encodePacket(metadata, audioBuf) {
  const metaStr = JSON.stringify(metadata);
  const metaBytes = new TextEncoder().encode(metaStr);
  const packet = new ArrayBuffer(
    4 + metaBytes.byteLength + audioBuf.byteLength,
  );
  const view = new DataView(packet);
  view.setUint32(0, metaBytes.byteLength, true);
  new Uint8Array(packet, 4, metaBytes.byteLength).set(metaBytes);
  new Uint8Array(packet, 4 + metaBytes.byteLength).set(
    new Uint8Array(audioBuf),
  );
  return packet;
}

function floatTo16BitPCM(samples) {
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm;
}

function flushAudioQueue() {
  if (
    !audioQueue.length ||
    !ws ||
    ws.readyState !== WebSocket.OPEN ||
    !isRecording
  )
    return;
  const totalLen = audioQueue.reduce((sum, arr) => sum + arr.length, 0);
  const merged = new Float32Array(totalLen);
  let offset = 0;
  for (const arr of audioQueue) {
    merged.set(arr, offset);
    offset += arr.length;
  }
  audioQueue = [];
  const pcm = floatTo16BitPCM(merged);
  const packet = encodePacket(
    {
      sampleRate: audioContext ? audioContext.sampleRate : 16000,
    },
    pcm.buffer,
  );
  ws.send(packet);
}

async function startRecording() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
      },
    });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    source = audioContext.createMediaStreamSource(stream);
    const scriptNode = audioContext.createScriptProcessor(4096, 1, 1);
    scriptNode.onaudioprocess = (e) => {
      if (!isRecording) return;
      const samples = e.inputBuffer.getChannelData(0);
      audioQueue.push(new Float32Array(samples));
      let sum = 0;
      for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
      currentLevel = Math.sqrt(sum / samples.length);
    };
    source.connect(scriptNode);
    scriptNode.connect(audioContext.destination);
    processor = scriptNode;
    setRecording(true);
    flushInterval = setInterval(flushAudioQueue, 100);
  } catch (err) {
    console.error(err);
    setRecording(false);
  }
}

function stopRecording() {
  setRecording(false);
  if (flushInterval) {
    clearInterval(flushInterval);
    flushInterval = null;
  }
  if (processor) {
    processor.disconnect();
    processor = null;
  }
  if (source) {
    source.disconnect();
    source = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  audioQueue = [];
}

function updateMeter() {
  const pct = Math.min(100, currentLevel * 500);
  ui.meterFill.style.width = pct + "%";
  currentLevel *= 0.92;
  requestAnimationFrame(updateMeter);
}
requestAnimationFrame(updateMeter);

ui.micBtn.addEventListener("click", () => {
  if (isRecording) stopRecording();
  else startRecording();
});

ui.clearBtn.addEventListener("click", () => {
  ui.transcriptArea.innerHTML = "";
  ui.realtimeDisplay.textContent = "";
});

ui.langSelect.addEventListener("change", () => {
  if (ws) {
    manualDisconnect = true;
    try {
      ws.close();
    } catch (_) {}
    ws = null;
    setConnected(false);
    connect();
  }
});

connect();
