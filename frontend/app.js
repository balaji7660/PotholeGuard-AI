/**
 * PotholeGuard-AI Frontend Application
 * Real-Time Mobile HUD, WebSocket Streaming, Voice Synthesis & Research Analytics
 */

// State variables
let ws = null;
let stream = null;
let isDetecting = false;
let isMuted = false;
let currentFacingMode = 'environment';
let lastFrameTime = performance.now();
let detectionHistory = [];
let recentRiskScores = [];

// DOM Elements
const video = document.getElementById('camera-feed');
const canvas = document.getElementById('hud-overlay');
const ctx = canvas.getContext('2d');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnMute = document.getElementById('btn-mute');
const btnSwitchCam = document.getElementById('btn-switch-cam');
const fpsVal = document.getElementById('fps-val');
const latModel = document.getElementById('lat-model');
const latNet = document.getElementById('lat-net');
const statusIndicator = document.getElementById('status-indicator');
const systemStatus = document.getElementById('system-status');
const modelModeBadge = document.getElementById('model-mode-badge');
const riskBanner = document.getElementById('risk-banner');
const riskTitle = document.getElementById('risk-title');
const recommendationText = document.getElementById('recommendation-text');
const riskIcon = document.getElementById('risk-icon');
const reasonsContainer = document.getElementById('reasons-container');
const reasonsList = document.getElementById('reasons-list');

// Badges
const mSize = document.getElementById('m-size');
const mDepth = document.getElementById('m-depth');
const mPath = document.getElementById('m-path');
const mApproach = document.getElementById('m-approach');
const mConf = document.getElementById('m-conf');
const mQuality = document.getElementById('m-quality');
const historyBody = document.getElementById('history-body');

// Navigation Tabs
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    const tabId = btn.getAttribute('data-tab');
    document.getElementById(tabId).classList.add('active');

    if (tabId === 'tab-history') loadHistory();
    if (tabId === 'tab-stats') {
      loadStats();
      drawRiskTimeline();
    }
  });
});

// Audio Mute Toggle
btnMute.addEventListener('click', () => {
  isMuted = !isMuted;
  btnMute.innerText = isMuted ? '🔇 Audio MUTED' : '🔊 Audio ON';
  if (isMuted && window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
});

// Camera Flipping
btnSwitchCam.addEventListener('click', async () => {
  currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
  if (isDetecting) {
    await initCamera();
  }
});

// Start Detection
btnStart.addEventListener('click', async () => {
  await initCamera();
  connectWebSocket();
  isDetecting = true;
  btnStart.style.display = 'none';
  btnStop.style.display = 'inline-block';
  systemStatus.innerText = 'STREAMING';
  statusIndicator.style.background = '#00f0ff';
});

// Stop Detection
btnStop.addEventListener('click', () => {
  isDetecting = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
  }
  if (ws) {
    ws.close();
  }
  btnStart.style.display = 'inline-block';
  btnStop.style.display = 'none';
  systemStatus.innerText = 'STANDBY';
  statusIndicator.style.background = '#10b981';
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (reasonsContainer) reasonsContainer.style.display = 'none';
});

// Initialize Camera
async function initCamera() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
  }
  try {
    const constraints = {
      video: {
        facingMode: currentFacingMode,
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();

    // Adjust canvas resolution
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
  } catch (err) {
    console.error('Camera access failed:', err);
    alert('Camera permission denied or camera unavailable: ' + err.message);
  }
}

// Connect WebSocket
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/detect`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('[+] WebSocket connected to backend');
    sendNextFrame();
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    renderHUD(data);
    if (isDetecting) {
      requestAnimationFrame(sendNextFrame);
    }
  };

  ws.onerror = (err) => {
    console.error('[!] WebSocket error:', err);
  };

  ws.onclose = () => {
    console.log('[-] WebSocket closed');
    if (isDetecting) {
      setTimeout(connectWebSocket, 1500);
    }
  };
}

// Capture & Send Frame
function sendNextFrame() {
  if (!isDetecting || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    requestAnimationFrame(sendNextFrame);
    return;
  }

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 480;
  tempCanvas.height = 360;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

  const base64Img = tempCanvas.toDataURL('image/jpeg', 0.65);
  const payload = {
    image: base64Img,
    timestamp: performance.now()
  };
  ws.send(JSON.stringify(payload));
}

// Render AR HUD Overlays & Telemetry
function renderHUD(data) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Measure Client FPS
  const now = performance.now();
  const fps = 1000.0 / (now - lastFrameTime);
  lastFrameTime = now;
  fpsVal.innerText = (data.processing_fps || fps).toFixed(1);

  // Latencies
  latModel.innerText = `${data.latency_ms || 0} ms`;
  latNet.innerText = `${data.network_latency_ms || 0} ms`;
  modelModeBadge.innerText = data.mode || 'REAL_MODEL';

  // Overall Risk UI
  const risk = data.overall_risk || 'SAFE';
  riskBanner.className = `risk-banner ${risk.toLowerCase()}`;
  riskTitle.innerText = risk === 'SAFE' ? 'ROAD CLEAR' : `${risk.replace('_', ' ')} DETECTED`;
  recommendationText.innerText = data.overall_recommendation || 'MAINTAIN COURSE';

  const riskIconMap = {
    SAFE: '🟢',
    WARNING: '🟡',
    HIGH_WARNING: '🟠',
    DANGER: '🔴',
    UNCERTAIN: '🟣'
  };
  riskIcon.innerText = riskIconMap[risk] || '🟢';

  // Explainable AI Reasons Card
  const reasons = data.overall_reasons || [];
  if (reasons.length > 0 && risk !== 'SAFE') {
    reasonsContainer.style.display = 'block';
    reasonsList.innerHTML = reasons.map(r => `<li>${r}</li>`).join('');
  } else {
    reasonsContainer.style.display = 'none';
  }

  // Quality badge
  if (data.image_quality && mQuality) {
    mQuality.innerText = data.image_quality.quality_tier || 'HIGH';
  }

  // Draw Trajectory Corridor Guideline
  const W = canvas.width;
  const H = canvas.height;
  ctx.strokeStyle = 'rgba(0, 240, 255, 0.25)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(W * 0.35, H);
  ctx.lineTo(W * 0.45, H * 0.4);
  ctx.moveTo(W * 0.65, H);
  ctx.lineTo(W * 0.55, H * 0.4);
  ctx.stroke();
  ctx.setLineDash([]);

  // Render Detections
  if (data.detections && data.detections.length > 0) {
    const primary = data.detections[0];
    mSize.innerText = primary.size_class || 'NONE';
    mDepth.innerText = primary.depth_class || 'NONE';
    mPath.innerText = primary.position_class ? primary.position_class.split('/')[0].trim() : 'CENTER';
    mApproach.innerText = primary.approach_state || 'FAR';
    mConf.innerText = `${Math.round((primary.confidence || 0.9) * 100)}%`;

    // Record risk for timeline
    recentRiskScores.push({
      time: Date.now(),
      score: primary.risk_score || 0.0,
      risk_class: primary.risk_class || 'SAFE'
    });
    if (recentRiskScores.length > 40) recentRiskScores.shift();

    data.detections.forEach(det => {
      const [x1, y1, x2, y2] = det.bbox;
      const w = x2 - x1;
      const h = y2 - y1;

      let color = '#10b981';
      if (det.risk_class === 'WARNING') color = '#f59e0b';
      if (det.risk_class === 'HIGH_WARNING') color = '#f97316';
      if (det.risk_class === 'DANGER') color = '#ef4444';
      if (det.risk_class === 'UNCERTAIN') color = '#8b5cf6';

      // Bounding Box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, w, h);

      // Label background
      const labelText = `ID #${det.track_id} [${det.range_category || det.approach_state || 'NEAR'}] - ${det.risk_class}`;
      ctx.font = 'bold 12px Inter, sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - 24, Math.max(textWidth + 14, w), 24);

      // Label text
      ctx.fillStyle = '#000';
      ctx.fillText(labelText, x1 + 6, y1 - 7);
    });
  } else {
    mSize.innerText = 'NONE';
    mDepth.innerText = 'NONE';
    mPath.innerText = 'CLEAR';
    mApproach.innerText = 'FAR';
    mConf.innerText = '100%';
  }

  // Voice Alert Triggering
  if (data.voice_alert && !isMuted) {
    speakAlert(data.voice_alert.voice_text);
  }
}

// Web SpeechSynthesis Engine
function speakAlert(text) {
  if (!('speechSynthesis' in window) || isMuted) return;
  if (window.speechSynthesis.speaking) return;

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.05;
  utterance.pitch = 1.0;
  window.speechSynthesis.speak(utterance);
}

// Draw Risk vs Time Timeline Chart
function drawRiskTimeline() {
  const tCanvas = document.getElementById('risk-timeline-canvas');
  if (!tCanvas) return;
  const tCtx = tCanvas.getContext('2d');
  tCtx.clearRect(0, 0, tCanvas.width, tCanvas.height);

  if (recentRiskScores.length < 2) {
    tCtx.fillStyle = '#94a3b8';
    tCtx.font = '12px Inter, sans-serif';
    tCtx.fillText('Accumulating live tracking risk points...', 20, 90);
    return;
  }

  const W = tCanvas.width;
  const H = tCanvas.height;
  const pad = 30;

  // Draw Grid Lines
  tCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  tCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (H - 2 * pad) * (i / 4.0);
    tCtx.beginPath();
    tCtx.moveTo(pad, y);
    tCtx.lineTo(W - pad, y);
    tCtx.stroke();
    tCtx.fillStyle = '#64748b';
    tCtx.font = '10px monospace';
    tCtx.fillText(((1.0 - i * 0.25)).toFixed(2), 5, y + 3);
  }

  // Plot Risk Line
  tCtx.beginPath();
  tCtx.lineWidth = 2.5;
  tCtx.strokeStyle = '#00f0ff';

  const n = recentRiskScores.length;
  recentRiskScores.forEach((pt, idx) => {
    const x = pad + (W - 2 * pad) * (idx / (n - 1));
    const y = (H - pad) - (pt.score * (H - 2 * pad));
    if (idx === 0) tCtx.moveTo(x, y);
    else tCtx.lineTo(x, y);
  });
  tCtx.stroke();
}

// Fetch and Render History Logs
async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const logs = await res.json();
    detectionHistory = logs;

    if (logs.length === 0) {
      historyBody.innerHTML = '<tr><td colspan="10" style="text-align: center; color: var(--text-muted);">No detections logged yet.</td></tr>';
      return;
    }

    historyBody.innerHTML = logs.map(l => `
      <tr>
        <td>${new Date(l.timestamp * 1000).toLocaleTimeString()}</td>
        <td>#${l.track_id}</td>
        <td><b style="color: ${getRiskColor(l.risk_class)}">${l.risk_class}</b></td>
        <td>${l.size_class}</td>
        <td>${l.depth_class}</td>
        <td>${l.approach_state || 'NEAR'}</td>
        <td>${Math.round((l.confidence || 0.9) * 100)}%</td>
        <td>${Math.round((l.uncertainty || 0.1) * 100)}%</td>
        <td>${l.path_relevance > 0.6 ? 'DIRECT' : 'OFF-PATH'}</td>
        <td>${l.recommendation}</td>
      </tr>
    `).join('');
  } catch (e) {
    console.error('History load error:', e);
  }
}

function getRiskColor(risk) {
  if (risk === 'DANGER') return '#ef4444';
  if (risk === 'HIGH_WARNING') return '#f97316';
  if (risk === 'WARNING') return '#f59e0b';
  if (risk === 'UNCERTAIN') return '#8b5cf6';
  return '#10b981';
}

// Fetch Researcher Stats
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const st = await res.json();
    document.getElementById('st-total').innerText = st.total_detection_logs || 0;
    document.getElementById('st-unique').innerText = st.unique_potholes_tracked || 0;
    document.getElementById('st-danger').innerText = (st.risk_distribution?.DANGER || 0) + (st.risk_distribution?.HIGH_WARNING || 0);
    document.getElementById('st-lat').innerText = `${st.average_latency_ms || 0} ms`;
  } catch (e) {
    console.error('Stats load error:', e);
  }
}

// Export CSV
document.getElementById('btn-export-csv').addEventListener('click', () => {
  if (detectionHistory.length === 0) {
    alert('No detection logs available to export.');
    return;
  }
  const headers = Object.keys(detectionHistory[0]).join(',');
  const rows = detectionHistory.map(r => Object.values(r).join(',')).join('\n');
  const csvContent = 'data:text/csv;charset=utf-8,' + headers + '\n' + rows;
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `potholeguard_logs_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

// Calibration Form Submission
document.getElementById('calib-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    camera_height_m: parseFloat(document.getElementById('c-height').value),
    camera_pitch_deg: parseFloat(document.getElementById('c-pitch').value),
    reference_distance_m: parseFloat(document.getElementById('c-dist').value),
    pixel_to_cm_ratio: parseFloat(document.getElementById('c-ratio').value)
  };
  try {
    const res = await fetch('/api/calibrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    alert('Calibration parameters updated successfully!');
  } catch (err) {
    alert('Calibration update failed: ' + err.message);
  }
});

// Video / Image Test Upload
document.getElementById('file-uploader').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const isVideo = file.type.startsWith('video/');
  const isImage = file.type.startsWith('image/');

  if (isVideo) {
    const videoElem = document.getElementById('uploaded-video');
    videoElem.src = URL.createObjectURL(file);
    document.getElementById('video-preview-container').style.display = 'block';
    document.getElementById('image-preview-container').style.display = 'none';
  } else if (isImage) {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/test/image', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      
      // Update Raw JSON
      document.getElementById('image-result-json').innerText = JSON.stringify(data, null, 2);
      document.getElementById('video-preview-container').style.display = 'none';
      document.getElementById('image-preview-container').style.display = 'block';

      // Update Result Risk Banner
      const risk = data.overall_risk || 'SAFE';
      const tBanner = document.getElementById('test-risk-banner');
      tBanner.className = `risk-banner ${risk.toLowerCase()}`;
      document.getElementById('test-risk-title').innerText = risk === 'SAFE' ? 'ROAD CLEAR' : `${risk.replace('_', ' ')} DETECTED`;
      document.getElementById('test-recommendation-text').innerText = data.overall_recommendation || 'MAINTAIN COURSE';
      
      const riskIconMap = { SAFE: '🟢', WARNING: '🟡', HIGH_WARNING: '🟠', DANGER: '🔴', UNCERTAIN: '🟣' };
      document.getElementById('test-risk-icon').innerText = riskIconMap[risk] || '🟢';

      // Update Explainable Rationale
      const reasons = data.overall_reasons || [];
      const rList = document.getElementById('test-reasons-list');
      if (reasons.length > 0) {
        document.getElementById('test-reasons-container').style.display = 'block';
        rList.innerHTML = reasons.map(r => `<li>${r}</li>`).join('');
      } else {
        document.getElementById('test-reasons-container').style.display = 'none';
      }

      // Update Badges
      if (data.detections && data.detections.length > 0) {
        const primary = data.detections[0];
        document.getElementById('test-m-size').innerText = primary.size_class || 'NONE';
        document.getElementById('test-m-depth').innerText = primary.depth_class || 'NONE';
        document.getElementById('test-m-path').innerText = primary.position_class ? primary.position_class.split('/')[0].trim() : 'CENTER';
        document.getElementById('test-m-approach').innerText = primary.approach_state || 'FAR';
      }

      // Draw Image & Detections on Canvas
      const img = new Image();
      img.onload = () => {
        const c = document.getElementById('test-image-canvas');
        c.width = img.width;
        c.height = img.height;
        const cCtx = c.getContext('2d');
        cCtx.drawImage(img, 0, 0);

        // Draw Trajectory Guideline Corridor
        cCtx.strokeStyle = 'rgba(0, 240, 255, 0.35)';
        cCtx.lineWidth = 3;
        cCtx.setLineDash([8, 8]);
        cCtx.beginPath();
        cCtx.moveTo(c.width * 0.35, c.height);
        cCtx.lineTo(c.width * 0.45, c.height * 0.4);
        cCtx.moveTo(c.width * 0.65, c.height);
        cCtx.lineTo(c.width * 0.55, c.height * 0.4);
        cCtx.stroke();
        cCtx.setLineDash([]);

        // Draw Bounding Boxes
        if (data.detections) {
          data.detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;
            const w = x2 - x1;
            const h = y2 - y1;

            let color = '#10b981';
            if (det.risk_class === 'WARNING') color = '#f59e0b';
            if (det.risk_class === 'HIGH_WARNING') color = '#f97316';
            if (det.risk_class === 'DANGER') color = '#ef4444';
            if (det.risk_class === 'UNCERTAIN') color = '#8b5cf6';

            cCtx.strokeStyle = color;
            cCtx.lineWidth = 4;
            cCtx.strokeRect(x1, y1, w, h);

            const labelText = `POTHOLE #${det.track_id} [${det.range_category || det.approach_state || 'NEAR'}] - ${det.risk_class} (${det.depth_class})`;
            cCtx.font = 'bold 13px Inter, sans-serif';
            const textWidth = cCtx.measureText(labelText).width;
            cCtx.fillStyle = color;
            cCtx.fillRect(x1, y1 - 28, Math.max(textWidth + 16, w), 28);

            cCtx.fillStyle = '#000';
            cCtx.fillText(labelText, x1 + 8, y1 - 8);
          });
        }
      };
      img.src = URL.createObjectURL(file);

      // Voice warning if unmuted
      if (data.voice_alert && !isMuted) {
        speakAlert(data.voice_alert.voice_text);
      }

    } catch (err) {
      alert('Image test failed: ' + err.message);
    }
  }
});

// Register Service Worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(err => console.log('SW registration error:', err));
}
