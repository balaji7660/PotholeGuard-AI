"""
run_mobile.py
One-Click Launcher for PotholeGuard-AI Real-Time Mobile Testing.

Features:
- Discovers your computer's local Wi-Fi / LAN IP address automatically
- Generates local self-signed SSL certificates for HTTPS (required by mobile browsers for camera access)
- Prints a scannable QR Code in your terminal to open directly on your smartphone
- Starts the high-performance FastAPI + WebSocket backend server

Usage:
    python run_mobile.py
    python run_mobile.py --port 8000
    python run_mobile.py --http   (Run over HTTP instead of HTTPS)
"""
from __future__ import annotations

import argparse
import datetime
import ipaddress
import os
import socket
import sys
from pathlib import Path

# Ensure UTF-8 stdout on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).parent.resolve()
CERTS_DIR = ROOT / "data" / "certs"


def get_local_ip() -> str:
    """Find the primary local network IP address of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Does not send actual traffic, just determines routing interface
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def generate_ssl_certificate(local_ip: str) -> tuple[Path, Path]:
    """
    Generate self-signed SSL certificate with Subject Alternative Names (SAN)
    for localhost and the local LAN IP address.
    """
    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    cert_path = CERTS_DIR / "cert.pem"
    key_path = CERTS_DIR / "key.pem"

    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    print("🔐 Generating SSL certificate for secure mobile camera access...")
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization

        # Generate RSA private key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Subject & Issuer
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "PotholeGuard-AI"),
            x509.NameAttribute(NameOID.COMMON_NAME, local_ip),
        ])

        # SAN items
        san_items = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]
        try:
            san_items.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))
        except ValueError:
            san_items.append(x509.DNSName(local_ip))

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_items), critical=False)
            .sign(key, hashes.SHA256())
        )

        # Write private key
        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))

        # Write certificate
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print(f"✅ SSL Certificate saved to: {CERTS_DIR}")
        return cert_path, key_path
    except Exception as e:
        print(f"⚠️ Could not generate SSL certificate automatically: {e}")
        return None, None


def print_qr_code(url: str):
    """Print ASCII QR Code to terminal."""
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        print("\n" + "─" * 50)
        print("📱 SCAN WITH YOUR SMARTPHONE CAMERA TO CONNECT:")
        print("─" * 50)
        qr.print_ascii(invert=True)
        print("─" * 50)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="PotholeGuard-AI Real-Time Mobile Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--http", action="store_true", help="Run with HTTP instead of HTTPS")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    local_ip = get_local_ip()
    use_https = not args.http

    cert_path, key_path = None, None
    if use_https:
        cert_path, key_path = generate_ssl_certificate(local_ip)
        if cert_path is None or key_path is None:
            print("⚠️ Falling back to HTTP mode.")
            use_https = False

    protocol = "https" if use_https else "http"
    mobile_url = f"{protocol}://{local_ip}:{args.port}"
    local_url = f"{protocol}://localhost:{args.port}"

    print("\n" + "=" * 60)
    print(" 🚗  POTHOLEGUARD-AI: REAL-TIME MOBILE HUD SERVER")
    print("=" * 60)
    print(f" 🌐 Local Browser URL  : {local_url}")
    print(f" 📱 Mobile Device URL   : {mobile_url}")
    print("=" * 60)

    print_qr_code(mobile_url)

    print("\n💡 MOBILE CONNECTION INSTRUCTIONS:")
    print(f" 1. Ensure your phone and PC are connected to the SAME Wi-Fi or Hotspot.")
    print(f" 2. Open Chrome (Android) or Safari (iOS) on your phone and go to:")
    print(f"    👉  {mobile_url}")
    if use_https:
        print(f" 3. Because of the local self-signed SSL certificate:")
        print(f"    - On Chrome: Click 'Advanced' -> 'Proceed to {local_ip} (unsafe)'")
        print(f"    - On Safari: Tap 'Show Details' -> 'visit this website' -> Confirm")
    print(f" 4. Tap 'START REAL-TIME HUD' and allow Camera permission.")
    print("=" * 60 + "\n")

    import uvicorn
    uvicorn_kwargs = {
        "app": "app.mobile_server:app",
        "host": args.host,
        "port": args.port,
        "reload": args.reload,
        "log_level": "info",
    }
    if use_https and cert_path and key_path:
        uvicorn_kwargs["ssl_certfile"] = str(cert_path)
        uvicorn_kwargs["ssl_keyfile"] = str(key_path)

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
