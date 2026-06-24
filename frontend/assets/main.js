const WS_URL = (() => {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return proto + "//" + location.host + "/ws";
})();

const ui = {
  actionBtn: document.getElementById("actionBtn"),
  micIcon: document.getElementById("micIcon"),
  sendIcon: document.getElementById("sendIcon"),
  textInput: document.getElementById("textInput"),
  userText: document.getElementById("userText"),
  assistantText: document.getElementById("assistantText"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
};

let ws = null;
let audioContext = null;
let textSoundBuffer = null;
let stream = null;
let processor = null;
let source = null;
let isRecording = false;
let isWaiting = false;
let audioQueue = [];
let flushInterval = null;
let manualDisconnect = false;
let hasInputText = false;
let currentLevel = 0;
let currentEmotion = "neutral1";

const EMOTION_VARIANTS = {
  neutral: 5,
  smirk: 1,
  concern: 1,
  crying: 1,
  distressed: 2,
  distressed_meow: 1,
  distressed_talk: 1,
  eyeclosed: 2,
  masked: 1,
  confused: 3,
  slightly_sad: 1,
  excited: 1,
  sad: 1,
  shocked: 1,
  smiling: 1,
  speaking: 1,
  surprised: 1,
  upset: 2,
  upset_meow: 1,
  wtf: 2,
  yawning: 1,
};

async function preloadFaces() {
  const base = "assets/faces/";
  const promises = [];
  for (const [emotion, count] of Object.entries(EMOTION_VARIANTS)) {
    for (let i = 1; i <= count; i++) {
      const img = new Image();
      img.src = base + emotion + i + ".png";
      promises.push(new Promise((r) => (img.onload = img.onerror = r)));
    }
  }
  await Promise.all(promises);
}

const CHAR_INTERVAL_MS = 10;
const SOUND_INTERVAL_MS = 80;
let charQueue = [];
let typewriterTimer = null;
let allTokensReceived = false;
let lastSoundTime = 0;
let thinkingTimeout = null;
let emotionBuffer = "";
let expectEmotion = true;

function setFace(emotion) {
  const count = EMOTION_VARIANTS[emotion];
  if (!count) return;
  const idx = Math.floor(Math.random() * count) + 1;
  const filename = emotion + idx + ".png";
  currentEmotion = filename;
  const faceEl = document.querySelector(".face");
  if (faceEl) faceEl.src = "assets/faces/" + filename;
}

function setStatus(state, text) {
  ui.statusDot.className = "status-dot" + (state ? " " + state : "");
  ui.statusText.textContent = text;
}

function setConnected(connected) {
  if (connected) {
    setStatus("connected", "Connected");
  } else {
    setStatus("", "Disconnected");
    isWaiting = false;
    clearUserText();
    clearAssistantText();
  }
  ui.actionBtn.disabled = !connected;
  if (!connected && isRecording) stopRecording();
  updateInputState();
}

function setWaiting(waiting) {
  isWaiting = waiting;
  if (waiting) {
    setStatus("processing", "The World Machine is thinking...");
    clearTimeout(thinkingTimeout);
    thinkingTimeout = setTimeout(() => {
      if (isWaiting) {
        setWaiting(false);
        clearAssistantText();
        setStatus("connected", "Connected");
      }
    }, 60000);
  } else {
    clearTimeout(thinkingTimeout);
    thinkingTimeout = null;
  }
  updateInputState();
}

function updateInputState() {
  const disabled = isRecording || isWaiting;
  ui.textInput.disabled = disabled;
  if (!disabled && !isRecording) {
    setTimeout(() => ui.textInput.focus(), 50);
  }
}

function clearUserText() {
  ui.userText.textContent = "";
}

function clearAssistantText() {
  ui.assistantText.textContent = "";
  charQueue = [];
  allTokensReceived = false;
  emotionBuffer = "";
  expectEmotion = true;
  if (typewriterTimer) {
    clearInterval(typewriterTimer);
    typewriterTimer = null;
  }
}

function updateActionButton() {
  const val = ui.textInput.value.trim();
  hasInputText = val.length > 0;
  if (isRecording) {
    ui.micIcon.style.display = "";
    ui.sendIcon.style.display = "none";
    ui.actionBtn.className = "action-btn recording";
    ui.actionBtn.title = "Disable microphone";
  } else if (hasInputText) {
    ui.micIcon.style.display = "none";
    ui.sendIcon.style.display = "";
    ui.actionBtn.className = "action-btn send-mode";
    ui.actionBtn.title = "Send message";
  } else {
    ui.micIcon.style.display = "";
    ui.sendIcon.style.display = "none";
    ui.actionBtn.className = "action-btn";
    ui.actionBtn.title = "Enable microphone";
  }
}

async function initAudio() {
  if (audioContext) {
    if (audioContext.state === "suspended") await audioContext.resume();
    return;
  }
  audioContext = new AudioContext();
  try {
    const resp = await fetch("assets/text_robot.wav");
    const buf = await resp.arrayBuffer();
    textSoundBuffer = await audioContext.decodeAudioData(buf);
  } catch (e) {
    console.warn("Could not load textSoundBuffer:", e);
  }
}

function playTextSound() {
  if (!textSoundBuffer || !audioContext) return;
  const now = Date.now();
  if (now - lastSoundTime < SOUND_INTERVAL_MS) return;
  lastSoundTime = now;
  try {
    const src = audioContext.createBufferSource();
    src.buffer = textSoundBuffer;
    src.playbackRate.value = 0.95 + Math.random() * 0.1;
    const gain = audioContext.createGain();
    gain.gain.value = 0.2 + Math.random() * 0.05;
    src.connect(gain);
    gain.connect(audioContext.destination);
    src.start();
  } catch (_) {}
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
    ws.send(JSON.stringify({ language: "" }));
    setStatus("", "Connecting...");
  };

  ws.onmessage = async (e) => {
    try {
      const msg = JSON.parse(e.data);
      switch (msg.type) {
        case "ready":
          setConnected(true);
          setTimeout(() => {
            setFace("neutral");
            const welcome =
              "[You have established a connection with The\xa0World\xa0Machine. State your purpose.";
            clearUserText();
            clearAssistantText();
            allTokensReceived = true;
            queueChars(welcome);
          }, 800);
          break;
        case "status":
          ui.statusText.textContent = msg.text || "Connected";
          break;
        case "final":
          ui.textInput.value = msg.text;
          updateActionButton();
          if (!isWaiting) {
            await initAudio();
            sendChat(msg.text);
          }
          break;
        case "token":
          if (!allTokensReceived) {
            if (ui.assistantText.textContent === "") {
              ui.assistantText.textContent = "[";
              expectEmotion = true;
              emotionBuffer = "";
            }
            queueChars(msg.text);
          }
          break;
        case "done":
          allTokensReceived = true;
          tryFinishTypewriter();
          break;
        case "error":
          clearAssistantText();
          allTokensReceived = true;
          setFace("wtf");
          queueChars(
            "[TRANSMISSION FAILURE. " +
              (msg.message || "UNKNOWN ERROR").toUpperCase() +
              ".",
          );
          break;
      }
    } catch (_) {}
  };

  ws.onclose = () => {
    setConnected(false);
    ws = null;
    if (!manualDisconnect) setTimeout(connect, 3000);
  };

  ws.onerror = () => {
    ws && ws.close();
  };
}

async function sendChat(text) {
  await initAudio();
  if (!text.trim() || !ws || ws.readyState !== WebSocket.OPEN || isWaiting)
    return;
  ui.userText.textContent = "You: " + text.trim();
  clearAssistantText();
  ui.textInput.value = "";
  updateActionButton();
  setWaiting(true);
  ws.send(JSON.stringify({ type: "chat", text: text.trim() }));
}

function queueChars(text) {
  if (expectEmotion) {
    if (text[0] === "{" || emotionBuffer.length > 0) {
      emotionBuffer += text;
      const closeIdx = emotionBuffer.indexOf("}");
      if (closeIdx !== -1) {
        setFace(emotionBuffer.substring(1, closeIdx));
        expectEmotion = false;
        for (let i = closeIdx + 1; i < emotionBuffer.length; i++)
          charQueue.push(emotionBuffer[i]);
        if (!typewriterTimer) startTypewriter();
        emotionBuffer = "";
      }
      return;
    } else {
      expectEmotion = false;
      setFace("neutral");
    }
  }
  for (const ch of text) charQueue.push(ch);
  if (!typewriterTimer) startTypewriter();
}

function startTypewriter() {
  if (typewriterTimer) return;
  typewriterTimer = setInterval(() => {
    if (charQueue.length > 0) {
      const ch = charQueue.shift();
      ui.assistantText.appendChild(document.createTextNode(ch));
      ui.assistantText.scrollTop = ui.assistantText.scrollHeight;
      playTextSound();
    } else if (allTokensReceived) {
      clearInterval(typewriterTimer);
      typewriterTimer = null;
      finishResponse();
    }
  }, CHAR_INTERVAL_MS);
}

function tryFinishTypewriter() {
  if (!typewriterTimer && charQueue.length === 0) {
    finishResponse();
  }
}

function finishResponse() {
  ui.assistantText.appendChild(document.createTextNode("]"));
  setWaiting(false);
  if (isRecording) {
    setStatus("recording", "Listening...");
  } else {
    setStatus("connected", "Connected");
  }
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
    { sampleRate: audioContext ? audioContext.sampleRate : 16000 },
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
    await initAudio();
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
    isRecording = true;
    setStatus("recording", "Listening...");
    flushInterval = setInterval(flushAudioQueue, 100);
    updateActionButton();
    updateInputState();
    requestAnimationFrame(updateMicFill);
  } catch (err) {
    stopRecording();
    clearUserText();
    clearAssistantText();
    allTokensReceived = true;
    setFace("eyeclosed");
    queueChars(
      "[I cannot hear you. The microphone permission may not be granted. Please try again.",
    );
  }
}

function updateMicFill() {
  if (isRecording) {
    const pct = Math.min(100, currentLevel * 500);
    document.getElementById("micFill").style.height = pct + "%";
    currentLevel *= 0.92;
    requestAnimationFrame(updateMicFill);
  } else {
    document.getElementById("micFill").style.height = "0%";
  }
}

function stopRecording() {
  isRecording = false;
  if (flushInterval) {
    clearInterval(flushInterval);
    flushInterval = null;
  }
  flushAudioQueue();
  if (processor) {
    processor.disconnect();
    processor = null;
  }
  if (source) {
    source.disconnect();
    source = null;
  }
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  audioQueue = [];
  updateActionButton();
  if (ws && ws.readyState === WebSocket.OPEN) {
    setStatus("connected", "Connected");
  } else {
    setStatus("", "Disconnected");
  }
  updateInputState();
}

ui.actionBtn.addEventListener("click", async (e) => {
  e.preventDefault();
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  await initAudio();

  if (isRecording) {
    stopRecording();
  } else if (hasInputText) {
    sendChat(ui.textInput.value.trim());
  } else {
    startRecording();
  }
});

ui.textInput.addEventListener("input", updateActionButton);

ui.textInput.addEventListener("keydown", async (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    const text = ui.textInput.value.trim();
    if (text && !isRecording && !isWaiting) {
      await initAudio();
      sendChat(text);
    }
  }
});

preloadFaces();
updateActionButton();
connect();
