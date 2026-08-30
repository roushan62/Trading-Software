#!/usr/bin/env python3
"""One-click desktop launcher.

Starts the Streamlit dashboard on a free local port and opens your browser.
Safe to double-click from the desktop shortcut. Ctrl-C / close window = exit.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "dashboard" / "app.py"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_healthy(port: int, timeout: float = 90.0) -> bool:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> None:
    if not APP.exists():
        print(f"[launcher] app not found: {APP}")
        input("Press Enter to exit…")
        return

    port = free_port()
    url = f"http://localhost:{port}"
    print("=" * 60)
    print("  Market Analysis & Trade Signal Software")
    print("  Decision-support + paper trading. Not financial advice.")
    print("=" * 60)
    print(f"[launcher] starting dashboard on {url} …")

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    try:
        if wait_healthy(port):
            print(f"[launcher] ready → opening browser at {url}")
            webbrowser.open(url)
        else:
            print(f"[launcher] server slow to start; manually open {url}")
        proc.wait()
    except KeyboardInterrupt:
        print("\n[launcher] stopping…")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("[launcher] bye.")


if __name__ == "__main__":
    main()
