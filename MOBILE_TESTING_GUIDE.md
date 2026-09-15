# 📱 PotholeGuard-AI: Real-Time Mobile Testing Guide

This guide walks you through deploying and testing the **Pothole Avoidance AI System** in real time on your smartphone (Android or iOS).

---

## 🌟 What You Get on Your Mobile Device

- **Live Camera AR Stream**: Streams your smartphone's back camera in real-time at 25–30 FPS.
- **Dynamic Steering Action HUD**: Prominent illuminated indicators showing real-time decisions:
  - ⬆️ **MAINTAIN LANE** (Safe trajectory ahead)
  - ⬅️ **SHIFT LEFT** (Avoids pothole on the right)
  - ➡️ **SHIFT RIGHT** (Avoids pothole on the left)
  - 🛑 **EMERGENCY BRAKE** (Critical hazard intervention)
- **Audio Voice Warnings**: Speaks voice guidance via your phone's speaker or car Bluetooth (*"Pothole detected, shift right"*, *"Emergency brake activated"*).
- **Haptic Vibration Feedback**: Mobile device vibrates when severe potholes or safety overrides trigger.
- **Multi-Modal Heatmap Views**:
  - 🎥 **AR Cam**: Camera feed + polygon hazard bounding contours + dynamic path arc.
  - 🌌 **Depth Map**: Real-time pseudo-depth estimation (Plasma colormap).
  - 🔥 **Uncertainty Map**: MC-Dropout epistemic uncertainty visualization.
  - 🛣️ **Road Sim**: Top-down bird's-eye road view with vehicle trajectory.
- **Torch / Flashlight Control**: Turn on your smartphone camera flash for night/low-light testing.

---

## 🚀 Quick Start (1 Command)

### Step 1: Start the Mobile Server on your Computer
Open your terminal in the project directory and run:

```bash
python run_mobile.py
```

### Step 2: Connect Your Smartphone
1. Ensure your smartphone and PC are connected to the **same Wi-Fi network** (or connect your PC to your phone's **Mobile Hotspot**).
2. The terminal will print a **Scannable QR Code** and your mobile URL, e.g.:
   ```
   https://10.249.136.71:8000
   ```
3. Open your phone camera or QR scanner and scan the QR code (or type the URL into Chrome / Safari).

---

## 🔒 Handling the HTTPS / SSL Certificate Notice

Mobile browsers (Chrome & Safari) require **HTTPS** to allow web applications to access the device's camera. `run_mobile.py` automatically generates a secure self-signed certificate for your local IP address.

When opening the URL on your mobile browser for the first time:

- **Google Chrome (Android / iOS)**:
  1. Tap **"Advanced"** (or *Details*).
  2. Tap **"Proceed to [IP Address] (unsafe)"**.
- **Apple Safari (iOS)**:
  1. Tap **"Show Details"**.
  2. Tap **"visit this website"** and confirm with Face ID / Passcode.

---

## 📱 Using the Mobile HUD Interface

1. **Start Stream**: Tap the large **"START REAL-TIME HUD"** button.
2. **Allow Permissions**: Tap **"Allow"** when prompted for Camera access.
3. **Point at Road or Test Media**:
   - **Method A (Live Testing)**: Mount the phone on your car dashboard or hold it while walking to test real road surfaces.
   - **Method B (Desk Testing)**: Point your phone camera at pothole images or dashcam videos playing on your computer screen or iPad.
   - **Method C (Photo Gallery Upload)**: Tap the 📁 folder icon in the HUD to select any photo from your phone's photo library.

---

## ⚙️ Advanced Options

### Run on Custom Port:
```bash
python run_mobile.py --port 8080
```

### Run on Plain HTTP (if using an external proxy / tunnel):
```bash
python run_mobile.py --http
```

### Using Streamlit on Mobile:
You can also access the Streamlit dashboard from your phone:
```bash
streamlit run app/main.py --server.address 0.0.0.0
```
Then select **"📷 Live Camera"** in the sidebar to capture live road photos.
