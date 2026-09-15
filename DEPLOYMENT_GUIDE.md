# 🚀 PotholeGuard-AI: Cloud & Mobile Deployment Guide

Deploy **PotholeGuard-AI** to the cloud so anyone can access the application from anywhere on any smartphone or computer with a public HTTPS link.

---

## 🌟 Option 1: Streamlit Community Cloud (Recommended • 100% Free)

Streamlit Community Cloud hosts your app directly from your GitHub repository for free with automatic updates on every git push.

### Step-by-Step Instructions:
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account (`balaji7660`).
2. Click **"New app"** (or **"Create app"**).
3. Fill in the repository details:
   - **Repository**: `balaji7660/PotholeGuard-AI`
   - **Branch**: `main`
   - **Main file path**: `app/main.py`
   - **App URL** (optional): `potholeguard-ai.streamlit.app`
4. Click **"Deploy!"**
5. Within 1–2 minutes, your web application will be live at:
   $$\text{\bf https://potholeguard-ai.streamlit.app}$$
6. You can open this link directly in any mobile browser (iOS Safari / Android Chrome) to test the live camera and pipeline anywhere!

---

## 🤗 Option 2: Hugging Face Spaces (Free Cloud Hosting)

1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and click **"Create new Space"**.
2. Set:
   - **Space name**: `potholeguard-ai`
   - **License**: `mit`
   - **Space SDK**: Select **Streamlit** (or **Docker**).
3. Connect your GitHub repository `balaji7660/PotholeGuard-AI`.
4. Your space will automatically build and deploy with free cloud compute.

---

## ⚡ Option 3: Render.com (Free Web Service)

1. Go to **[render.com](https://render.com/)** and sign in with GitHub.
2. Click **"New"** $\rightarrow$ **"Web Service"**.
3. Select the `balaji7660/PotholeGuard-AI` repository.
4. Render will automatically detect [`render.yaml`](render.yaml) and configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0`
5. Click **"Create Web Service"**.

---

## 🌐 Option 4: Instant Public Mobile Tunnel (Localtunnel / Ngrok)

If you want to test the **Mobile HUD** (`run_mobile.py` on port 8000) over cellular data (4G/5G) from anywhere in the world without being on the same local Wi-Fi:

### Using Localtunnel (No signup needed):
In a new terminal window, run:
```bash
npx localtunnel --port 8000
```
It will output a public HTTPS URL (e.g. `https://pothole-ai-test.loca.lt`). Open that link on your smartphone!

### Using Ngrok:
```bash
ngrok http 8000
```
Open the provided `https://xxxx.ngrok-free.app` URL on your phone.

---

## 🐳 Option 5: Self-Hosted Docker Container

To run in a production Docker container locally or on any VPS (AWS, GCP, DigitalOcean):

```bash
# Build the Docker container
docker build -t potholeguard-ai .

# Run the container exposing port 8501
docker run -p 8501:8501 potholeguard-ai
```
Open: `http://localhost:8501`
