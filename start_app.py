
import subprocess
import time
import os
import sys

def start_backend():
    print("Starting Backend (FastAPI)...")
    # Using python directly to avoid path issues
    return subprocess.Popen([sys.executable, "backend/main.py"], cwd=".")

def start_frontend():
    print("Starting Frontend (Vite)...")
    # npm run dev
    return subprocess.Popen(["npm", "run", "dev"], cwd="frontend", shell=True)

if __name__ == "__main__":
    try:
        backend_process = start_backend()
        time.sleep(2) # Give backend a moment
        frontend_process = start_frontend()
        
        print("\n" + "="*50)
        print(" APP STARTED SUCCESSFULLY")
        print(" Backend: http://localhost:8000/docs")
        print(" Frontend: http://localhost:5173")
        print(" Press Ctrl+C to stop both servers.")
        print("="*50 + "\n")
        
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)
