import subprocess
import sys
import signal

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def start_fastapi():
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"], cwd=PROJECT_DIR)

def start_streamlit():
    return subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", "8501"], cwd=PROJECT_DIR)

if __name__ == "__main__":
    fastapi_proc = start_fastapi()
    streamlit_proc = start_streamlit()
    print("[INFO] FastAPI and Streamlit started. Press Ctrl+C to stop.")
    try:
        fastapi_proc.wait()
        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down services…")
        fastapi_proc.send_signal(signal.SIGINT)
        streamlit_proc.send_signal(signal.SIGINT)
        fastapi_proc.wait()
        streamlit_proc.wait()
        print("[INFO] All services stopped.")
