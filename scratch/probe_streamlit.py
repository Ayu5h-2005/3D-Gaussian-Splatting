import subprocess
import time
import urllib.request

proc = subprocess.Popen(
    [
        r"D:\miniconda3\envs\3dgs_research\python.exe",
        "-m", "streamlit", "run",
        r"D:\3dgs-incremental-research\ui\app.py",
        "--server.port", "8504",
        "--server.headless", "true"
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

try:
    time.sleep(6)
    req = urllib.request.urlopen("http://localhost:8504", timeout=5)
    print("STATUS:", req.getcode())
except Exception as e:
    print("ERROR:", e)
finally:
    proc.terminate()
    proc.wait()
