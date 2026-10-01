import json
import PySimpleGUI as sg
from datetime import datetime
import os
import sys

# Add project directory to path to allow importing core, ml, utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.prediction_service import run_prediction
from database.database import DatabaseManager

# Initialize database manager
_db_manager = DatabaseManager()

# Helper functions
def add_patient_and_assessment(payload, prediction_result):
    """Add patient record and clinical assessment to the SQLite DB.

    Args:
        payload (dict): Original input payload.
        prediction_result (dict): Output from run_prediction.
    """
    patient_info = payload.get("patient_info", {})
    name = patient_info.get("name", "Unknown")
    age = patient_info.get("age", 0)
    gender = patient_info.get("gender", "U")
    # Generate a simple MRN using timestamp
    mrn = f"MRN-{int(datetime.now().timestamp())}"
    patient_id = _db_manager.add_patient(name, age, gender, mrn)

    # Extract vitals and labs from the tabular payload
    tab = payload.get("tabular", [])
    vitals = {
        "systolic_bp": tab[0] if len(tab) > 0 else None,
        "diastolic_bp": tab[1] if len(tab) > 1 else None,
        "heart_rate": tab[2] if len(tab) > 2 else None,
        "respiratory_rate": tab[3] if len(tab) > 3 else None,
        "spo2": tab[4] if len(tab) > 4 else None,
        "body_temperature": tab[5] if len(tab) > 5 else None,
    }
    labs = {
        "wbc": tab[6] if len(tab) > 6 else None,
        "serum_creatinine": tab[7] if len(tab) > 7 else None,
        "troponin_i": tab[9] if len(tab) > 9 else None,
        "charlson_index": tab[13] if len(tab) > 13 else None,
    }
    # Store the assessment (clinical notes left empty for now)
    assessment_id = _db_manager.add_assessment(
        patient_id,
        vitals=vitals,
        labs=labs,
        clinical_notes=json.dumps(payload.get("text", "")),
        diagnosis=prediction_result.get("risk_status", "")
    )

    # Record the model prediction history
    _db_manager.add_prediction_record(
        assessment_id=assessment_id,
        model_version="v1.0",
        tabular_input=json.dumps(payload.get("tabular", [])),
        prediction_output=json.dumps(prediction_result),
        execution_time_ms=0.0  # placeholder; you could time the call if desired
    )
    return patient_id, assessment_id

# GUI layout
def make_window():
    sg.theme("DarkBlue13")
    layout = [
        [sg.Text("Paste JSON payload for prediction:")],
        [sg.Multiline(size=(80, 15), key="-PAYLOAD-")],
        [sg.Button("Run Prediction", key="-RUN-"), sg.Button("Save to DB", key="-SAVE-", disabled=True), sg.Button("Clear", key="-CLEAR-")],
        [sg.Text("Prediction Output:"), sg.Multiline(size=(80, 12), key="-OUTPUT-", disabled=True)],
    ]
    return sg.Window("Medical Multimodal Prediction", layout, finalize=True)

def main():
    window = make_window()
    prediction_result = None
    payload_data = None
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == "-RUN-":
            raw = values["-PAYLOAD-"].strip()
            if not raw:
                sg.popup_error("Please paste a JSON payload.")
                continue
            try:
                payload_data = json.loads(raw)
            except json.JSONDecodeError as e:
                sg.popup_error(f"Invalid JSON: {e}")
                continue
            try:
                prediction_result = run_prediction(payload_data)
                pretty = json.dumps(prediction_result, indent=2)
                window["-OUTPUT-"].update(pretty)
                window["-SAVE-"].update(disabled=False)
            except Exception as e:
                sg.popup_error(f"Prediction failed: {e}")
                prediction_result = None
        if event == "-SAVE-" and prediction_result and payload_data:
            try:
                patient_id, assessment_id = add_patient_and_assessment(payload_data, prediction_result)
                sg.popup_ok(f"Saved to DB. Patient ID: {patient_id}, Assessment ID: {assessment_id}")
                window["-SAVE-"].update(disabled=True)
            except Exception as e:
                sg.popup_error(f"Failed to save to DB: {e}")
        if event == "-CLEAR-":
            window["-PAYLOAD-"].update("")
            window["-OUTPUT-"].update("")
            window["-SAVE-"].update(disabled=True)
    window.close()

if __name__ == "__main__":
    main()
