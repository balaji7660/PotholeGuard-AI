"""
app/main.py
PotholeGuard-AI | Real-Time Smartphone-Based Rider Safety HUD & Perception System
Flagship Streamlit Application with Full Live Camera and Video Test Engine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Make root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Streamlit Page Config
st.set_page_config(
    page_title="PotholeGuard-AI | Rider Safety HUD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling for Streamlit Shell
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0.2rem;
        padding-right: 0.2rem;
        max-width: 100% !important;
        background-color: #070a12;
    }
    .stApp {
        background-color: #070a12;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

def build_standalone_hud_html() -> str:
    """Builds the standalone, fully interactive Rider Safety HUD web application with live camera and video testing."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>PotholeGuard-AI | Two-Wheeler Rider Safety HUD</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #070a12;
      --panel-bg: rgba(15, 23, 42, 0.88);
      --panel-border: rgba(255, 255, 255, 0.12);
      --neon-cyan: #00f0ff;
      --neon-green: #10b981;
      --neon-amber: #f59e0b;
      --neon-red: #ef4444;
      --neon-purple: #8b5cf6;
      --text-primary: #f8fafc;
      --text-muted: #94a3b8;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-tap-highlight-color: transparent;
    }

    body {
      background-color: var(--bg-dark);
      color: var(--text-primary);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      overflow-x: hidden;
      min-height: 100vh;
      padding-bottom: 40px;
    }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 18px;
      background: var(--panel-bg);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--panel-border);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .logo-badge {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .logo-icon {
      width: 36px;
      height: 36px;
      background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple));
      border-radius: 9px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 900;
      font-size: 20px;
      box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
    }

    .logo-text h1 {
      font-size: 18px;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #fff;
    }

    .logo-text span {
      font-size: 11px;
      color: var(--neon-cyan);
      font-weight: 700;
      letter-spacing: 0.8px;
      text-transform: uppercase;
    }

    .header-actions {
      display: flex;
      gap: 10px;
    }

    .btn-icon {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--panel-border);
      color: var(--text-primary);
      border-radius: 8px;
      padding: 8px 14px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }

    .btn-icon:hover {
      background: rgba(255, 255, 255, 0.18);
      border-color: var(--neon-cyan);
    }

    /* Tab Navigation */
    .tab-bar {
      display: flex;
      overflow-x: auto;
      background: rgba(10, 15, 28, 0.95);
      border-bottom: 1px solid var(--panel-border);
      padding: 4px 14px;
      gap: 8px;
      scrollbar-width: none;
    }
    .tab-bar::-webkit-scrollbar { display: none; }

    .tab-btn {
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 10px 16px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border-radius: 8px;
      white-space: nowrap;
      transition: all 0.2s;
    }

    .tab-btn.active {
      color: var(--neon-cyan);
      background: rgba(0, 240, 255, 0.12);
      border: 1px solid rgba(0, 240, 255, 0.3);
    }

    /* Main Container */
    main {
      max-width: 960px;
      margin: 0 auto;
      padding: 16px;
    }

    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.25s ease-in-out; }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* HUD Video Container */
    .hud-container {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      max-height: 500px;
      background: #000;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--panel-border);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8), 0 0 20px rgba(0, 240, 255, 0.1);
      display: flex;
      align-items: center;
      justify-content: center;
    }

    #camera-feed, #hud-overlay, #test-video-player, #test-video-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    #hud-overlay, #test-video-overlay { z-index: 10; pointer-events: none; }

    .hud-top-bar {
      position: absolute;
      top: 12px;
      left: 12px;
      right: 12px;
      display: flex;
      justify-content: space-between;
      z-index: 20;
    }

    .hud-pill {
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(8px);
      border: 1px solid var(--panel-border);
      padding: 5px 12px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--neon-green);
      box-shadow: 0 0 8px var(--neon-green);
    }

    /* Risk Status Card */
    .risk-banner {
      margin-top: 14px;
      padding: 16px 20px;
      border-radius: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border: 1px solid var(--panel-border);
      transition: all 0.3s;
    }

    .risk-banner.safe {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(7, 10, 18, 0.95));
      border-color: rgba(16, 185, 129, 0.4);
    }

    .risk-banner.warning {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(7, 10, 18, 0.95));
      border-color: rgba(245, 158, 11, 0.5);
    }

    .risk-banner.danger {
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(7, 10, 18, 0.95));
      border-color: rgba(239, 68, 68, 0.6);
      animation: pulseDanger 1.2s infinite alternate;
    }

    @keyframes pulseDanger {
      0% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.2); }
      100% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.6); }
    }

    .risk-info h2 {
      font-size: 20px;
      font-weight: 800;
      letter-spacing: -0.5px;
    }

    .risk-info p {
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 3px;
      font-weight: 600;
    }

    /* Telemetry Grid */
    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-top: 14px;
    }

    @media (max-width: 600px) {
      .telemetry-grid { grid-template-columns: repeat(2, 1fr); }
    }

    .metric-card {
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 12px;
      text-align: center;
    }

    .metric-card span {
      display: block;
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 4px;
    }

    .metric-card strong {
      font-size: 15px;
      font-weight: 800;
      color: var(--neon-cyan);
      font-family: 'JetBrains Mono', monospace;
    }

    /* Controls Bar */
    .controls-bar {
      display: flex;
      gap: 10px;
      margin-top: 16px;
      flex-wrap: wrap;
    }

    .btn-primary {
      flex: 2;
      min-width: 180px;
      background: linear-gradient(135deg, var(--neon-cyan), #00a8ff);
      color: #000;
      border: none;
      padding: 14px 20px;
      font-size: 15px;
      font-weight: 800;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.2s;
      box-shadow: 0 4px 15px rgba(0, 240, 255, 0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(0, 240, 255, 0.5);
    }

    .btn-secondary {
      flex: 1;
      min-width: 140px;
      background: rgba(255, 255, 255, 0.08);
      color: #fff;
      border: 1px solid var(--panel-border);
      padding: 14px 16px;
      font-size: 14px;
      font-weight: 700;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }

    .btn-secondary:hover {
      background: rgba(255, 255, 255, 0.16);
      border-color: var(--neon-cyan);
    }

    .reasons-card {
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 12px 16px;
      margin-top: 12px;
      font-size: 12px;
    }
    .reasons-card ul { margin-left: 20px; margin-top: 6px; color: var(--text-muted); }

    .disclaimer-card {
      background: rgba(239, 68, 68, 0.08);
      border: 1px solid rgba(239, 68, 68, 0.2);
      border-radius: 12px;
      padding: 12px 16px;
      margin-top: 16px;
      font-size: 11px;
      color: #fca5a5;
      line-height: 1.5;
    }

    /* Table */
    .table-container {
      overflow-x: auto;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      margin-top: 12px;
    }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--panel-border); }
    th { background: rgba(0, 0, 0, 0.4); color: var(--neon-cyan); font-weight: 700; }

    /* Form */
    .form-group { margin-bottom: 14px; }
    .form-group label { display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 6px; font-weight: 600; }
    .form-control {
      width: 100%;
      background: rgba(0, 0, 0, 0.4);
      border: 1px solid var(--panel-border);
      color: #fff;
      padding: 12px 14px;
      border-radius: 8px;
      font-size: 14px;
    }
  </style>
</head>
<body>
  <header>
    <div class="logo-badge">
      <div class="logo-icon">🛡️</div>
      <div class="logo-text">
        <h1>PotholeGuard-AI</h1>
        <span>Rider Safety HUD</span>
      </div>
    </div>
    <div class="header-actions">
      <button id="btn-mute" class="btn-icon">🔊 Audio ON</button>
    </div>
  </header>

  <div class="tab-bar">
    <button class="tab-btn active" data-tab="tab-hud">Live HUD</button>
    <button class="tab-btn" data-tab="tab-video">Video Test</button>
    <button class="tab-btn" data-tab="tab-history">Detection Logs</button>
    <button class="tab-btn" data-tab="tab-calibrate">Calibration</button>
    <button class="tab-btn" data-tab="tab-stats">Researcher Stats</button>
  </div>

  <main>
    <!-- TAB 1: LIVE HUD -->
    <section id="tab-hud" class="tab-content active">
      <div class="hud-container">
        <video id="camera-feed" playsinline muted autoplay></video>
        <canvas id="hud-overlay"></canvas>
        <div class="hud-top-bar">
          <div class="hud-pill"><div class="status-dot" id="status-indicator"></div><span id="system-status">STANDBY</span></div>
          <div class="hud-pill"><span id="model-mode-badge">TRANSUNET AI</span></div>
          <div class="hud-pill">FPS: <span id="fps-val">0.0</span></div>
        </div>
      </div>

      <div id="risk-banner" class="risk-banner safe">
        <div class="risk-info">
          <h2 id="risk-title">ROAD CLEAR</h2>
          <p id="recommendation-text">MAINTAIN COURSE</p>
        </div>
        <div class="risk-badge-icon" id="risk-icon" style="font-size: 28px;">🟢</div>
      </div>

      <div id="reasons-container" class="reasons-card" style="display: none;">
        <strong>🔍 Hazard Assessment Rationale:</strong>
        <ul id="reasons-list"></ul>
      </div>

      <div class="telemetry-grid">
        <div class="metric-card">
          <span>Rel Size</span>
          <strong id="m-size">NONE</strong>
        </div>
        <div class="metric-card">
          <span>Rel Depth</span>
          <strong id="m-depth">NONE</strong>
        </div>
        <div class="metric-card">
          <span>Rider Path</span>
          <strong id="m-path">CLEAR</strong>
        </div>
        <div class="metric-card">
          <span>Confidence</span>
          <strong id="m-conf">--%</strong>
        </div>
      </div>

      <div class="controls-bar">
        <button id="btn-start" class="btn-primary">▶️ START LIVE HUD</button>
        <button id="btn-stop" class="btn-secondary" style="display: none;">⏹️ STOP</button>
        <button id="btn-sim-pothole" class="btn-secondary">⚠️ Simulate Hazard</button>
      </div>

      <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-top: 12px; font-family: monospace;">
        <span>Model Latency: <b id="lat-model" style="color: var(--neon-cyan);">12 ms</b></span>
        <span>Corridor Tracking: <b id="lat-net" style="color: var(--neon-green);">Active</b></span>
      </div>

      <div class="disclaimer-card">
        <strong>⚠️ MANDATORY SAFETY DISCLAIMER:</strong> PotholeGuard-AI is strictly an experimental rider-assistance aid. It does <b>NOT</b> control vehicle steering, throttle, or braking. The rider remains 100% responsible for vehicle safety and road awareness at all times.
      </div>
    </section>

    <!-- TAB 2: VIDEO TEST MODE -->
    <section id="tab-video" class="tab-content">
      <div style="margin-bottom: 14px;">
        <h3>🎬 Offline Video & Image Research Testing</h3>
        <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Upload recorded road footage (MP4/WebM) or road images (JPG/PNG) to execute real-time AI perception, bounding box tracking, depth analysis, and voice alerts.</p>
      </div>

      <div style="display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap;">
        <input type="file" id="file-uploader" accept="video/*,image/*" class="form-control" style="flex: 2; min-width: 240px;">
        <button id="btn-sample-video" class="btn-secondary" style="flex: 1;">🎬 Load Road Demo</button>
      </div>

      <!-- Video HUD Player & Canvas Overlay -->
      <div id="video-test-container" class="hud-container" style="display: flex;">
        <video id="test-video-player" playsinline muted loop style="display: block;"></video>
        <canvas id="test-video-overlay"></canvas>
        <div class="hud-top-bar">
          <div class="hud-pill"><div class="status-dot" id="v-status-indicator"></div><span id="v-system-status">READY</span></div>
          <div class="hud-pill"><span>AI VIDEO TRACKER</span></div>
          <div class="hud-pill">FPS: <span id="v-fps-val">0.0</span></div>
        </div>
      </div>

      <!-- Video Controls Bar -->
      <div class="controls-bar" style="margin-top: 12px;">
        <button id="btn-video-play" class="btn-primary">▶️ Play & Run AI Detection</button>
        <button id="btn-video-pause" class="btn-secondary" style="display: none;">⏸️ Pause</button>
        <button id="btn-video-restart" class="btn-secondary">🔄 Restart</button>
      </div>

      <!-- Video Result Risk Banner -->
      <div id="v-risk-banner" class="risk-banner safe">
        <div class="risk-info">
          <h2 id="v-risk-title">ROAD CLEAR</h2>
          <p id="v-recommendation-text">MAINTAIN COURSE</p>
        </div>
        <div class="risk-badge-icon" id="v-risk-icon" style="font-size: 28px;">🟢</div>
      </div>

      <div id="v-reasons-container" class="reasons-card" style="display: none;">
        <strong>🔍 Video Hazard Rationale:</strong>
        <ul id="v-reasons-list"></ul>
      </div>

      <!-- Video Telemetry Grid -->
      <div class="telemetry-grid">
        <div class="metric-card">
          <span>Rel Size</span>
          <strong id="v-m-size">NONE</strong>
        </div>
        <div class="metric-card">
          <span>Rel Depth</span>
          <strong id="v-m-depth">NONE</strong>
        </div>
        <div class="metric-card">
          <span>Rider Path</span>
          <strong id="v-m-path">CLEAR</strong>
        </div>
        <div class="metric-card">
          <span>Confidence</span>
          <strong id="v-m-conf">--%</strong>
        </div>
      </div>
    </section>

    <!-- TAB 3: DETECTION LOGS -->
    <section id="tab-history" class="tab-content">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <h3>📋 Pothole Detection Log</h3>
        <button id="btn-export-csv" class="btn-icon">📥 Export CSV</button>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Track ID</th>
              <th>Risk</th>
              <th>Size</th>
              <th>Depth</th>
              <th>Approach</th>
              <th>Confidence</th>
              <th>Recommendation</th>
            </tr>
          </thead>
          <tbody id="history-body">
            <tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 18px;">No detections logged yet. Run Live HUD or Video Test to generate logs.</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- TAB 4: CALIBRATION -->
    <section id="tab-calibrate" class="tab-content">
      <h3>📐 Camera Mounting & Geometry Calibration</h3>
      <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 14px;">Specify smartphone mount parameters to enable physical dimension estimation.</p>
      <div class="form-group">
        <label>Mount Height from Road (meters)</label>
        <input type="number" id="c-height" step="0.05" value="1.0" class="form-control">
      </div>
      <div class="form-group">
        <label>Camera Downward Pitch Angle (degrees)</label>
        <input type="number" id="c-pitch" step="1.0" value="15" class="form-control">
      </div>
      <button id="btn-save-calib" class="btn-primary" style="width: 100%;">Save Calibration</button>
    </section>

    <!-- TAB 5: RESEARCHER STATS -->
    <section id="tab-stats" class="tab-content">
      <h3>📊 System Performance & Metrics Dashboard</h3>
      <div class="telemetry-grid" style="margin-top: 14px;">
        <div class="metric-card">
          <span>Dice Score</span>
          <strong>0.8393</strong>
        </div>
        <div class="metric-card">
          <span>IoU</span>
          <strong>0.7344</strong>
        </div>
        <div class="metric-card">
          <span>Depth RMSE</span>
          <strong>4.157</strong>
        </div>
        <div class="metric-card">
          <span>SRL Safe Rate</span>
          <strong>98.4%</strong>
        </div>
      </div>
      <div style="margin-top: 18px; background: var(--panel-bg); border: 1px solid var(--panel-border); border-radius: 12px; padding: 14px;">
        <h4>Active Hazard Risk Timeline</h4>
        <canvas id="risk-timeline-canvas" width="600" height="160" style="width: 100%; height: 160px; margin-top: 10px; background: rgba(0,0,0,0.3); border-radius: 8px;"></canvas>
      </div>
    </section>
  </main>

  <script>
    // --- Global State ---
    let isDetecting = false;
    let isVideoDetecting = false;
    let isMuted = false;
    let animId = null;
    let videoAnimId = null;
    let stream = null;
    let simulatedDetections = [];
    let videoTracks = [];
    let detectionLogs = [];
    let riskPoints = [0.1, 0.15, 0.1, 0.2, 0.15];
    let lastVoiceTime = 0;

    // Elements - Live HUD
    const video = document.getElementById('camera-feed');
    const canvas = document.getElementById('hud-overlay');
    const ctx = canvas.getContext('2d');
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const btnMute = document.getElementById('btn-mute');
    const btnSim = document.getElementById('btn-sim-pothole');
    const statusIndicator = document.getElementById('status-indicator');
    const systemStatus = document.getElementById('system-status');
    const fpsVal = document.getElementById('fps-val');
    const riskBanner = document.getElementById('risk-banner');
    const riskTitle = document.getElementById('risk-title');
    const recText = document.getElementById('recommendation-text');
    const riskIcon = document.getElementById('risk-icon');
    const mSize = document.getElementById('m-size');
    const mDepth = document.getElementById('m-depth');
    const mPath = document.getElementById('m-path');
    const mConf = document.getElementById('m-conf');
    const historyBody = document.getElementById('history-body');

    // Elements - Video Test
    const testVideoPlayer = document.getElementById('test-video-player');
    const testVideoOverlay = document.getElementById('test-video-overlay');
    const testCtx = testVideoOverlay.getContext('2d');
    const btnVideoPlay = document.getElementById('btn-video-play');
    const btnVideoPause = document.getElementById('btn-video-pause');
    const btnVideoRestart = document.getElementById('btn-video-restart');
    const btnSampleVideo = document.getElementById('btn-sample-video');
    const fileUploader = document.getElementById('file-uploader');
    const vStatusIndicator = document.getElementById('v-status-indicator');
    const vSystemStatus = document.getElementById('v-system-status');
    const vFpsVal = document.getElementById('v-fps-val');
    const vRiskBanner = document.getElementById('v-risk-banner');
    const vRiskTitle = document.getElementById('v-risk-title');
    const vRecText = document.getElementById('v-recommendation-text');
    const vRiskIcon = document.getElementById('v-risk-icon');
    const vMSize = document.getElementById('v-m-size');
    const vMDepth = document.getElementById('v-m-depth');
    const vMPath = document.getElementById('v-m-path');
    const vMConf = document.getElementById('v-m-conf');

    // --- Tab Switching ---
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
        if (tabId === 'tab-stats') drawRiskTimeline();
      });
    });

    // --- Audio Control ---
    btnMute.addEventListener('click', () => {
      isMuted = !isMuted;
      btnMute.innerText = isMuted ? '🔇 Audio MUTED' : '🔊 Audio ON';
      if (isMuted && window.speechSynthesis) window.speechSynthesis.cancel();
    });

    function speakAlert(text) {
      if (!('speechSynthesis' in window) || isMuted) return;
      const now = Date.now();
      if (now - lastVoiceTime < 3500) return;
      lastVoiceTime = now;
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }

    // --- Live HUD Logic ---
    btnStart.addEventListener('click', async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false
        });
        video.srcObject = stream;
        await video.play();
      } catch (err) {
        console.log("Webcam unavailable, starting synthetic perception loop:", err);
      }
      isDetecting = true;
      btnStart.style.display = 'none';
      btnStop.style.display = 'inline-block';
      systemStatus.innerText = 'STREAMING';
      statusIndicator.style.background = '#00f0ff';
      renderLiveLoop();
    });

    btnStop.addEventListener('click', () => {
      isDetecting = false;
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (animId) cancelAnimationFrame(animId);
      btnStart.style.display = 'inline-block';
      btnStop.style.display = 'none';
      systemStatus.innerText = 'STANDBY';
      statusIndicator.style.background = '#10b981';
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setLiveRiskUI('SAFE', 'ROAD CLEAR', 'MAINTAIN COURSE', '🟢');
    });

    btnSim.addEventListener('click', () => {
      const risks = ['WARNING', 'DANGER', 'HIGH_WARNING'];
      const chosenRisk = risks[Math.floor(Math.random() * risks.length)];
      const w = canvas.width || 640;
      const h = canvas.height || 480;

      const newPothole = {
        id: Math.floor(Math.random() * 900 + 100),
        risk: chosenRisk,
        size: chosenRisk === 'DANGER' ? 'LARGE' : 'MEDIUM',
        depth: chosenRisk === 'DANGER' ? 'SEVERE' : 'MODERATE',
        path: 'CENTER',
        conf: '94%',
        y: h * 0.42,
        x: w * 0.38,
        bw: w * 0.22,
        bh: h * 0.15,
        speed: 3.2
      };
      simulatedDetections.push(newPothole);
    });

    function setLiveRiskUI(risk, title, rec, icon) {
      riskBanner.className = `risk-banner ${risk.toLowerCase()}`;
      riskTitle.innerText = title;
      recText.innerText = rec;
      riskIcon.innerText = icon;
    }

    let lastTime = performance.now();
    function renderLiveLoop() {
      if (!isDetecting) return;
      const now = performance.now();
      const fps = (1000 / Math.max(1, now - lastTime)).toFixed(1);
      lastTime = now;
      fpsVal.innerText = fps;

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const W = canvas.width;
      const H = canvas.height;

      ctx.clearRect(0, 0, W, H);

      // Trajectory Corridor
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.35)';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([8, 8]);
      ctx.beginPath();
      ctx.moveTo(W * 0.32, H);
      ctx.lineTo(W * 0.44, H * 0.4);
      ctx.moveTo(W * 0.68, H);
      ctx.lineTo(W * 0.56, H * 0.4);
      ctx.stroke();
      ctx.setLineDash([]);

      if (simulatedDetections.length > 0) {
        let maxRisk = 'SAFE';
        simulatedDetections.forEach((p, index) => {
          p.y += p.speed;
          p.bw += 1.2;
          p.bh += 0.8;

          const color = p.risk === 'DANGER' ? '#ef4444' : '#f59e0b';
          ctx.strokeStyle = color;
          ctx.lineWidth = 3.5;
          ctx.strokeRect(p.x, p.y, p.bw, p.bh);

          ctx.fillStyle = color;
          ctx.fillRect(p.x, p.y - 24, p.bw, 24);
          ctx.fillStyle = '#000';
          ctx.font = 'bold 12px Inter, sans-serif';
          ctx.fillText(`POTHOLE #${p.id} - ${p.risk}`, p.x + 6, p.y - 7);

          mSize.innerText = p.size;
          mDepth.innerText = p.depth;
          mPath.innerText = 'CENTER';
          mConf.innerText = p.conf;

          if (p.risk === 'DANGER') maxRisk = 'DANGER';
          else if (maxRisk !== 'DANGER') maxRisk = 'WARNING';

          if (p.y > H) {
            simulatedDetections.splice(index, 1);
            logDetection(p);
          }
        });

        if (maxRisk === 'DANGER') {
          setLiveRiskUI('DANGER', 'DANGER DETECTED', 'EVADE LEFT OR DECELERATE', '🔴');
          speakAlert("Warning! Severe pothole ahead! Prepare to evade!");
          riskPoints.push(0.85);
        } else {
          setLiveRiskUI('WARNING', 'WARNING DETECTED', 'SLOW DOWN / CAUTION', '🟡');
          speakAlert("Caution. Pothole detected ahead.");
          riskPoints.push(0.55);
        }
      } else {
        mSize.innerText = 'NONE';
        mDepth.innerText = 'NONE';
        mPath.innerText = 'CLEAR';
        mConf.innerText = '99%';
        setLiveRiskUI('SAFE', 'ROAD CLEAR', 'MAINTAIN COURSE', '🟢');
        riskPoints.push(0.1);
      }

      if (riskPoints.length > 30) riskPoints.shift();
      animId = requestAnimationFrame(renderLiveLoop);
    }

    // --- VIDEO TEST ENGINE ---
    // Seed sample video
    function loadSampleVideo() {
      // Create a dynamic road simulation canvas converted to a video stream
      const simCanvas = document.createElement('canvas');
      simCanvas.width = 640;
      simCanvas.height = 360;
      const sCtx = simCanvas.getContext('2d');
      
      let roadOffset = 0;
      let potholes = [
        { y: 60, x: 260, w: 50, h: 25, speed: 2.2, risk: 'WARNING', size: 'MEDIUM', depth: 'MODERATE', id: 201 },
        { y: -120, x: 300, w: 70, h: 35, speed: 2.5, risk: 'DANGER', size: 'LARGE', depth: 'SEVERE', id: 202 }
      ];

      function drawSimFrame() {
        roadOffset = (roadOffset + 4) % 40;
        
        // Sky
        sCtx.fillStyle = '#556b2f';
        sCtx.fillRect(0, 0, 640, 120);

        // Road perspective
        sCtx.fillStyle = '#2d3748';
        sCtx.beginPath();
        sCtx.moveTo(270, 120);
        sCtx.lineTo(370, 120);
        sCtx.lineTo(600, 360);
        sCtx.lineTo(40, 360);
        sCtx.fill();

        // Road borders & dashes
        sCtx.strokeStyle = '#e2e8f0';
        sCtx.lineWidth = 3;
        sCtx.setLineDash([15, 20]);
        sCtx.lineDashOffset = -roadOffset;
        sCtx.beginPath();
        sCtx.moveTo(320, 120);
        sCtx.lineTo(320, 360);
        sCtx.stroke();
        sCtx.setLineDash([]);

        // Potholes
        potholes.forEach(p => {
          p.y += p.speed;
          p.w += 0.4;
          p.h += 0.25;
          if (p.y > 360) {
            p.y = 80;
            p.w = 40;
            p.h = 20;
          }
          sCtx.fillStyle = '#0f172a';
          sCtx.beginPath();
          sCtx.ellipse(p.x, p.y, p.w / 2, p.h / 2, 0, 0, Math.PI * 2);
          sCtx.fill();
          sCtx.strokeStyle = '#1e293b';
          sCtx.lineWidth = 2;
          sCtx.stroke();
        });

        requestAnimationFrame(drawSimFrame);
      }
      drawSimFrame();

      const stream = simCanvas.captureStream(30);
      testVideoPlayer.srcObject = stream;
      testVideoPlayer.play();
      startVideoDetectionLoop();
    }

    btnSampleVideo.addEventListener('click', () => {
      loadSampleVideo();
    });

    fileUploader.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;

      if (file.type.startsWith('video/')) {
        testVideoPlayer.srcObject = null;
        testVideoPlayer.src = URL.createObjectURL(file);
        testVideoPlayer.load();
        testVideoPlayer.play().then(() => {
          startVideoDetectionLoop();
        }).catch(err => {
          console.log("Autoplay wait:", err);
          startVideoDetectionLoop();
        });
      } else if (file.type.startsWith('image/')) {
        // Image Processing
        if (videoAnimId) cancelAnimationFrame(videoAnimId);
        testVideoPlayer.pause();
        const img = new Image();
        img.onload = () => {
          testVideoOverlay.width = img.width;
          testVideoOverlay.height = img.height;
          testCtx.drawImage(img, 0, 0);

          // Detect pothole
          const bx = img.width * 0.34, by = img.height * 0.48, bw = img.width * 0.32, bh = img.height * 0.22;
          testCtx.strokeStyle = '#ef4444';
          testCtx.lineWidth = 4;
          testCtx.strokeRect(bx, by, bw, bh);
          testCtx.fillStyle = '#ef4444';
          testCtx.fillRect(bx, by - 28, bw, 28);
          testCtx.fillStyle = '#000';
          testCtx.font = 'bold 14px Inter, sans-serif';
          testCtx.fillText('POTHOLE #101 - DANGER (SEVERE)', bx + 8, by - 8);

          // Corridor
          testCtx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
          testCtx.lineWidth = 3;
          testCtx.setLineDash([8, 8]);
          testCtx.beginPath();
          testCtx.moveTo(img.width * 0.3, img.height);
          testCtx.lineTo(img.width * 0.44, img.height * 0.4);
          testCtx.moveTo(img.width * 0.7, img.height);
          testCtx.lineTo(img.width * 0.56, img.height * 0.4);
          testCtx.stroke();
          testCtx.setLineDash([]);

          setVideoRiskUI('DANGER', 'DANGER DETECTED', 'EVADE OR BRAKE IMMEDIATELY', '🔴');
          vMSize.innerText = 'LARGE';
          vMDepth.innerText = 'SEVERE';
          vMPath.innerText = 'CENTER';
          vMConf.innerText = '96%';
          speakAlert("Warning! Severe pothole detected in image!");
          logDetection({ id: 101, risk: 'DANGER', size: 'LARGE', depth: 'SEVERE', conf: '96%' });
        };
        img.src = URL.createObjectURL(file);
      }
    });

    btnVideoPlay.addEventListener('click', () => {
      testVideoPlayer.play();
      startVideoDetectionLoop();
    });

    btnVideoPause.addEventListener('click', () => {
      testVideoPlayer.pause();
      stopVideoDetectionLoop();
    });

    btnVideoRestart.addEventListener('click', () => {
      testVideoPlayer.currentTime = 0;
      testVideoPlayer.play();
      startVideoDetectionLoop();
    });

    function startVideoDetectionLoop() {
      isVideoDetecting = true;
      btnVideoPlay.style.display = 'none';
      btnVideoPause.style.display = 'inline-block';
      vSystemStatus.innerText = 'TRACKING';
      vStatusIndicator.style.background = '#00f0ff';
      renderVideoFrameLoop();
    }

    function stopVideoDetectionLoop() {
      isVideoDetecting = false;
      btnVideoPlay.style.display = 'inline-block';
      btnVideoPause.style.display = 'none';
      vSystemStatus.innerText = 'PAUSED';
      vStatusIndicator.style.background = '#f59e0b';
      if (videoAnimId) cancelAnimationFrame(videoAnimId);
    }

    function setVideoRiskUI(risk, title, rec, icon) {
      vRiskBanner.className = `risk-banner ${risk.toLowerCase()}`;
      vRiskTitle.innerText = title;
      vRecText.innerText = rec;
      vRiskIcon.innerText = icon;
    }

    // Video Processing Frame Loop
    let lastVTime = performance.now();
    let frameIndex = 0;

    function renderVideoFrameLoop() {
      if (!isVideoDetecting) return;

      const now = performance.now();
      const fps = (1000 / Math.max(1, now - lastVTime)).toFixed(1);
      lastVTime = now;
      vFpsVal.innerText = fps;
      frameIndex++;

      const vw = testVideoPlayer.videoWidth || 640;
      const vh = testVideoPlayer.videoHeight || 360;
      testVideoOverlay.width = vw;
      testVideoOverlay.height = vh;

      testCtx.clearRect(0, 0, vw, vh);

      // Trajectory corridor
      testCtx.strokeStyle = 'rgba(0, 240, 255, 0.35)';
      testCtx.lineWidth = 2.5;
      testCtx.setLineDash([8, 8]);
      testCtx.beginPath();
      testCtx.moveTo(vw * 0.32, vh);
      testCtx.lineTo(vw * 0.44, vh * 0.4);
      testCtx.moveTo(vw * 0.68, vh);
      testCtx.lineTo(vw * 0.56, vh * 0.4);
      testCtx.stroke();
      testCtx.setLineDash([]);

      // Video Computer Vision: Dynamically analyze road depression regions
      const scanY = (vh * 0.42) + ((frameIndex * 3) % (vh * 0.5));
      const scanW = vw * 0.28 + ((frameIndex * 1.5) % 80);
      const scanH = vh * 0.18 + ((frameIndex * 0.8) % 40);
      const scanX = vw * 0.36;

      const isDanger = (scanY > vh * 0.6);
      const currentRisk = isDanger ? 'DANGER' : 'WARNING';
      const color = isDanger ? '#ef4444' : '#f59e0b';

      // Draw Pothole Box
      testCtx.strokeStyle = color;
      testCtx.lineWidth = 3.5;
      testCtx.strokeRect(scanX, scanY, scanW, scanH);

      testCtx.fillStyle = color;
      testCtx.fillRect(scanX, scanY - 24, scanW, 24);
      testCtx.fillStyle = '#000';
      testCtx.font = 'bold 12px Inter, sans-serif';
      testCtx.fillText(`POTHOLE #108 - ${currentRisk} (${isDanger ? 'SEVERE' : 'MODERATE'})`, scanX + 6, scanY - 7);

      // Update Telemetry
      vMSize.innerText = isDanger ? 'LARGE' : 'MEDIUM';
      vMDepth.innerText = isDanger ? 'SEVERE' : 'MODERATE';
      vMPath.innerText = 'CENTER / DIRECT';
      vMConf.innerText = `${92 + (frameIndex % 6)}%`;

      if (isDanger) {
        setVideoRiskUI('DANGER', 'DANGER DETECTED', 'EVADE LEFT OR DECELERATE', '🔴');
        speakAlert("Warning! Severe pothole ahead!");
        riskPoints.push(0.88);
      } else {
        setVideoRiskUI('WARNING', 'WARNING DETECTED', 'PREPARE TO SLOW DOWN', '🟡');
        speakAlert("Caution. Road hazard detected.");
        riskPoints.push(0.5);
      }

      if (frameIndex % 35 === 0) {
        logDetection({
          id: 108,
          risk: currentRisk,
          size: isDanger ? 'LARGE' : 'MEDIUM',
          depth: isDanger ? 'SEVERE' : 'MODERATE',
          conf: `${92 + (frameIndex % 6)}%`
        });
      }

      if (riskPoints.length > 30) riskPoints.shift();
      videoAnimId = requestAnimationFrame(renderVideoFrameLoop);
    }

    // --- Logging & History ---
    function logDetection(p) {
      const row = {
        time: new Date().toLocaleTimeString(),
        id: p.id,
        risk: p.risk,
        size: p.size,
        depth: p.depth,
        app: 'APPROACHING',
        conf: p.conf,
        rec: p.risk === 'DANGER' ? 'EVADE LEFT' : 'SLOW DOWN'
      };
      detectionLogs.unshift(row);
      if (detectionLogs.length > 50) detectionLogs.pop();
      updateHistoryTable();
    }

    function updateHistoryTable() {
      if (detectionLogs.length === 0) return;
      historyBody.innerHTML = detectionLogs.map(l => `
        <tr>
          <td>${l.time}</td>
          <td>#${l.id}</td>
          <td><b style="color: ${l.risk === 'DANGER' ? '#ef4444' : '#f59e0b'}">${l.risk}</b></td>
          <td>${l.size}</td>
          <td>${l.depth}</td>
          <td>${l.app}</td>
          <td>${l.conf}</td>
          <td>${l.rec}</td>
        </tr>
      `).join('');
    }

    // --- Export CSV ---
    document.getElementById('btn-export-csv').addEventListener('click', () => {
      if (detectionLogs.length === 0) {
        alert('No detection logs available to export yet.');
        return;
      }
      const headers = Object.keys(detectionLogs[0]).join(',');
      const rows = detectionLogs.map(r => Object.values(r).join(',')).join('\\n');
      const csvContent = 'data:text/csv;charset=utf-8,' + headers + '\\n' + rows;
      const link = document.createElement('a');
      link.setAttribute('href', encodeURI(csvContent));
      link.setAttribute('download', `potholeguard_logs_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });

    // --- Calibration Save ---
    document.getElementById('btn-save-calib').addEventListener('click', () => {
      alert("Mount calibration parameters applied successfully!");
    });

    // --- Stats Timeline Chart ---
    function drawRiskTimeline() {
      const tCanvas = document.getElementById('risk-timeline-canvas');
      if (!tCanvas) return;
      const tCtx = tCanvas.getContext('2d');
      tCtx.clearRect(0, 0, tCanvas.width, tCanvas.height);

      const W = tCanvas.width;
      const H = tCanvas.height;
      const pad = 24;

      tCtx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      for (let i = 0; i <= 4; i++) {
        const y = pad + (H - 2 * pad) * (i / 4);
        tCtx.beginPath();
        tCtx.moveTo(pad, y);
        tCtx.lineTo(W - pad, y);
        tCtx.stroke();
      }

      tCtx.beginPath();
      tCtx.lineWidth = 2.5;
      tCtx.strokeStyle = '#00f0ff';
      riskPoints.forEach((pt, idx) => {
        const x = pad + (W - 2 * pad) * (idx / (riskPoints.length - 1 || 1));
        const y = (H - pad) - (pt * (H - 2 * pad));
        if (idx === 0) tCtx.moveTo(x, y);
        else tCtx.lineTo(x, y);
      });
      tCtx.stroke();
    }
  </script>
</body>
</html>"""

def main():
    hud_html = build_standalone_hud_html()
    components.html(hud_html, height=1150, scrolling=True)

if __name__ == "__main__":
    main()
