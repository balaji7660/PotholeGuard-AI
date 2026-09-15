import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import socket
import qrcode
import uvicorn

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def print_qr(url: str):
    qr = qrcode.QRCode()
    qr.add_data(url)
    qr.print_ascii(invert=True)

if __name__ == "__main__":
    ip = get_local_ip()
    port = 8000
    mobile_url = f"http://{ip}:{port}"

    print("\n" + "="*64)
    print("      [+] POTHOLEGUARD-AI: SMARTPHONE HUD TESTER  ")
    print("="*64)
    print(f"\n1. Connect your smartphone and laptop to the SAME Wi-Fi network.")
    print(f"2. Scan the QR code below or navigate to:\n   -> {mobile_url}\n")
    try:
        print_qr(mobile_url)
    except Exception:
        pass
    print("3. Tap 'START LIVE HUD' and allow rear camera permission.")
    print("="*64 + "\n")

    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)
