import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uvicorn
import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    ip = get_local_ip()
    port = 8000
    print("================================================================")
    print("           [+] POTHOLEGUARD-AI FASTAPI BACKEND SERVER           ")
    print("================================================================")
    print(f"[*] Local URL:    http://localhost:{port}")
    print(f"[*] Network URL:  http://{ip}:{port}")
    print("================================================================")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)
