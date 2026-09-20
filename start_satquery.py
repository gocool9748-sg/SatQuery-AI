import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

PROJECT_DIR = Path(__file__).resolve().parent
VOICE_DIR = PROJECT_DIR / "voice"
LOG_DIR = PROJECT_DIR / "runs" / "launcher_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

STREAMLIT_PORT = 8501
FASTAPI_PORT = 8000
NGROK_API = "http://127.0.0.1:4040/api/tunnels"


def port_open(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def start_process(command, cwd, log_name):
    log_path = LOG_DIR / log_name
    log = open(log_path, "a", encoding="utf-8", buffering=1)

    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW

    print("Starting:", " ".join(map(str, command)))
    print("Log:", log_path)

    return subprocess.Popen(
        command,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )


def wait_for_port(port, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if port_open(port):
            return True
        time.sleep(0.5)
    return False


def get_ngrok_url(timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urlopen(NGROK_API, timeout=2) as response:
                import json
                data = json.loads(response.read().decode("utf-8"))
            for tunnel in data.get("tunnels", []):
                url = tunnel.get("public_url", "")
                if url.startswith("https://"):
                    return url
        except Exception:
            pass
        time.sleep(0.5)
    return None


def find_server():
    candidates = [VOICE_DIR / "server.py", VOICE_DIR / "server_object_voice.py"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find voice/server.py or voice/server_object_voice.py"
    )


def main():
    print("=" * 70)
    print("SATQUERY AI — ONE WEBSITE LAUNCHER")
    print("=" * 70)

    processes = []

    try:
        server_file = find_server()

        # 1. FastAPI voice/tool backend
        if not port_open(FASTAPI_PORT):
            processes.append(
                start_process(
                    [sys.executable, str(server_file)],
                    VOICE_DIR,
                    "fastapi.log",
                )
            )
            if not wait_for_port(FASTAPI_PORT, 60):
                raise RuntimeError(
                    "FastAPI did not start on port 8000. Check runs/launcher_logs/fastapi.log"
                )
        else:
            print("FastAPI is already running on port 8000.")

        # 2. ngrok — Vapi needs a public HTTPS endpoint for the tool webhook.
        if not port_open(4040):
            processes.append(
                start_process(
                    ["ngrok", "http", str(FASTAPI_PORT)],
                    PROJECT_DIR,
                    "ngrok.log",
                )
            )
            time.sleep(2)

        public_url = get_ngrok_url()
        if public_url:
            print("\nNgrok public URL:")
            print(public_url)
            print("Vapi tool URL should be:")
            print(public_url + "/vapi/tools")
        else:
            print("\nWARNING: Could not read the ngrok public URL.")
            print("Check runs/launcher_logs/ngrok.log")

        # 3. Streamlit website
        if not port_open(STREAMLIT_PORT):
            processes.append(
                start_process(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        "app.py",
                        "--server.port",
                        str(STREAMLIT_PORT),
                    ],
                    PROJECT_DIR,
                    "streamlit.log",
                )
            )
            if not wait_for_port(STREAMLIT_PORT, 90):
                raise RuntimeError(
                    "Streamlit did not start on port 8501. Check runs/launcher_logs/streamlit.log"
                )
        else:
            print("Streamlit is already running on port 8501.")

        url = f"http://localhost:{STREAMLIT_PORT}"
        print("\n" + "=" * 70)
        print("SATQUERY AI IS READY")
        print("=" * 70)
        print("Website:", url)
        print("\nKeep this launcher terminal open while using SatQuery AI.")
        print("Press Ctrl+C here to stop the services started by this launcher.")
        print("=" * 70)

        time.sleep(1)

        while True:
            time.sleep(2)
            # If a child process exits unexpectedly, report it but keep the
            # launcher alive so the user can inspect the logs.
            for p in processes:
                if p.poll() is not None:
                    print(f"\nA background service exited with code {p.returncode}.")

    except KeyboardInterrupt:
        print("\nStopping SatQuery AI...")
    except Exception as exc:
        print("\nLAUNCH ERROR:", exc)
        input("Press Enter to close...")
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        time.sleep(1)
        for p in reversed(processes):
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
        print("SatQuery AI launcher stopped.")


if __name__ == "__main__":
    main()
