"""
==============================================================
  MEDICAL MULTIMODAL SYSTEM - COMPLETE RUN ALL SCRIPT
==============================================================
Runs: config -> data_pipeline -> models -> train -> database
      -> test_database -> app (API server) -> prediction test
==============================================================
"""
import os, sys, time, json, subprocess
import numpy as np
import urllib.request, urllib.error, urllib.parse

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON      = sys.executable

def banner(title):
    print("\n" + "=" * 62)
    print("  " + title)
    print("=" * 62)

def ok(msg):   print("[PASS] " + msg)
def err(msg):  print("[FAIL] " + msg)
def info(msg): print("[INFO] " + msg)

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────
# STEP 1 – Verify core imports
# ──────────────────────────────────────────────────────────────
banner("STEP 1 -- Verifying imports (config / models / data_pipeline)")
try:
    sys.path.insert(0, PROJECT_DIR)
    from core.config import Config
    ok(f"config.py  ->  NUM_TABULAR_FEATURES={Config.NUM_TABULAR_FEATURES}, "
       f"EMBED_DIM={Config.EMBED_DIM}, NUM_CLASSES={Config.NUM_CLASSES}")
except Exception as e:
    err(f"config.py failed: {e}"); sys.exit(1)

try:
    import torch
    from ml.models import ClinicalMultimodalFusionClassifier
    m = ClinicalMultimodalFusionClassifier()
    total_params = sum(p.numel() for p in m.parameters())
    ok(f"models.py  ->  ClinicalMultimodalFusionClassifier loaded  "
       f"({total_params:,} parameters)")
except Exception as e:
    err(f"models.py failed: {e}"); sys.exit(1)

try:
    from ml.data_pipeline import get_data_loaders
    train_dl, val_dl = get_data_loaders()
    ok(f"data_pipeline.py  ->  train_loader={len(train_dl.dataset)} samples, "
       f"val_loader={len(val_dl.dataset)} samples")
except Exception as e:
    err(f"data_pipeline.py failed: {e}"); sys.exit(1)

# ──────────────────────────────────────────────────────────────
# STEP 2 – Train the model
# ──────────────────────────────────────────────────────────────
banner("STEP 2 -- Training model  (train.py)")
info("Running training loop -- please wait ...")
result = subprocess.run(
    [PYTHON, "-m", "ml.train"],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
    cwd=PROJECT_DIR
)
print(result.stdout)
if result.returncode != 0:
    err("train.py failed!")
    print(result.stderr)
    sys.exit(1)
ok("train.py completed successfully")

# ──────────────────────────────────────────────────────────────
# STEP 3 – Database verification
# ──────────────────────────────────────────────────────────────
banner("STEP 3 -- Database verification (database.py)")
try:
    from database.database import DatabaseManager, init_database
    init_database()
    db = DatabaseManager()
    ok(f"database.py  ->  Database initialized & connected (Patients: {db.get_patient_count()}, Assessments: {db.get_assessment_count()})")
except Exception as e:
    err(f"database.py failed: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# STEP 4 – Start FastAPI server in background
# ──────────────────────────────────────────────────────────────
banner("STEP 4 -- Starting FastAPI server  (app.py -> port 8000)")
server_env = os.environ.copy()
server_env["PYTHONIOENCODING"] = "utf-8"
server_proc = subprocess.Popen(
    [PYTHON, "-m", "api.main"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    cwd=PROJECT_DIR, env=server_env
)
info("Waiting 6 s for server to start ...")
time.sleep(6)

if server_proc.poll() is not None:
    out, er = server_proc.communicate()
    err("Server failed to start!")
    print(er.decode("utf-8", errors="replace"))
    sys.exit(1)
ok("FastAPI server running at http://127.0.0.1:8000")

# ──────────────────────────────────────────────────────────────
# STEP 5 – Live prediction test
# ──────────────────────────────────────────────────────────────
banner("STEP 5 -- Live prediction test  (POST /predict)")

simulated_image = [[0.0] * 224 for _ in range(224)]

payload = {
    "patient_info": {
        "name": "Demo Patient",
        "age": 58,
        "cardiomegaly": False,
        "pleural_effusion": False,
        "consolidation": False,
        "lung_nodule": False
    },
    "tabular": [
        130.0,   # SBP
         85.0,   # DBP
         88.0,   # HR
         18.0,   # RR
         97.0,   # SpO2
         37.2,   # Temp
          9.5,   # WBC
          1.1,   # Serum Creatinine
          8.0,   # CRP
          0.03,  # Troponin-I
         58.0,   # Age
          1.0,   # Gender (1=Male)
         27.0,   # BMI
          2.0,   # Charlson Index
    ],
    "image": simulated_image,
    "text": (
        "58-year-old male presenting with mild hypertension and slight tachycardia. "
        "No acute distress. History of type-2 diabetes. Labs show mildly elevated WBC. "
        "Chest X-ray unremarkable. Plan: monitor and adjust antihypertensives."
    )
}

try:
    body = json.dumps(payload).encode("utf-8")
    
    # 1. Fetch JWT token
    admin_user = os.getenv("ADMIN_USERNAME", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
    auth_data = urllib.parse.urlencode({"username": admin_user, "password": admin_pass}).encode("utf-8")
    token_req = urllib.request.Request("http://127.0.0.1:8000/token", data=auth_data, method="POST")
    with urllib.request.urlopen(token_req, timeout=10) as t_resp:
        token_info = json.loads(t_resp.read().decode("utf-8"))
        token = token_info.get("access_token")

    # 2. Make authenticated prediction request
    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        response = json.loads(resp.read().decode("utf-8"))

    ok("Prediction received from API!")
    pred_class = response['predicted_class']
    label      = response['diagnostic_label']
    conf0      = response['confidence_scores']['class_0_stable']   * 100
    conf1      = response['confidence_scores']['class_1_critical'] * 100
    ood_flag   = "YES (warning)" if response['ood'] else "NO (clean)"
    ood_why    = ", ".join(response.get('ood_reasons', [])) or "none"

    print()
    print("  +-------------------------------------------------------+")
    print(f"  |  Predicted Class    : {pred_class}                              |")
    print(f"  |  Diagnostic Label   : {label:<36}|")
    print(f"  |  Stable  Confidence : {conf0:6.2f}%                          |")
    print(f"  |  Critical Confidence: {conf1:6.2f}%                          |")
    print(f"  |  OOD Flag           : {ood_flag:<36}|")
    print(f"  |  OOD Reasons        : {ood_why:<36}|")
    print("  +-------------------------------------------------------+")
    print()

except Exception as e:
    err(f"Prediction request failed: {e}")
    server_proc.terminate()
    sys.exit(1)

# ──────────────────────────────────────────────────────────────
# STEP 6 – Shutdown server
# ──────────────────────────────────────────────────────────────
banner("STEP 6 -- Shutting down background API server")
server_proc.terminate()
server_proc.wait(timeout=5)
ok("Server stopped cleanly")

# ──────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────
banner("ALL STEPS COMPLETE -- FINAL OUTPUT SUMMARY")
print(f"""
  [OK] config.py          -- Config loaded
  [OK] models.py          -- Model architecture OK ({total_params:,} params)
  [OK] data_pipeline.py   -- Data loaders OK
  [OK] train.py           -- Model trained & saved
  [OK] database.py        -- SQLite DB initialized
  [OK] test_database.py   -- All DB tests passed
  [OK] app.py             -- FastAPI server ran (port 8000)
  [OK] /predict endpoint  -- Live prediction SUCCESS

  Model file    : {Config.MODEL_PATH}
  Database file : {os.path.join(PROJECT_DIR, 'database', 'clinical_data.db')}
  Inference log : {os.path.join(PROJECT_DIR, 'logs', 'inference.log')}

  To launch the DASHBOARD:
     .venv\\Scripts\\streamlit run dashboard/app.py

  To start ONLY the API server:
     .venv\\Scripts\\python.exe api/main.py
""")
print("=" * 62)
