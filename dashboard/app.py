import streamlit as st
import numpy as np
import requests
import pandas as pd
import re
import altair as alt
import plotly.express as px
from datetime import datetime
from PIL import Image
import sys
import os
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import Config
from core.clinical_reference import check_value_status, CLINICAL_REFERENCE_RANGES
from database.database import DatabaseManager, init_database
from utils.ocr_utils import set_tesseract_path, image_to_text
from ml.prediction_service import run_prediction
from api.chat_service import get_chat_response, get_ai_recommendations

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical AI Command Center", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# --- INITIALIZATION ---
init_database()
db = DatabaseManager()
set_tesseract_path()

# --- CSS STYLING ---
st.markdown("""
<style>
/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* Make header transparent so the sidebar toggle arrow is still visible */
header {background: transparent !important;}
/* Custom Font */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;800&display=swap');
html, body, [class*="css"] {
font-family: 'Outfit', sans-serif !important;
}
/* Premium Dark Navy Background */
.stApp {
background: #0B101E !important;
background-image: radial-gradient(circle at 50% 0%, #172554, #0B101E 70%) !important;
}
/* Sidebar Styling */
[data-testid="stSidebar"] {
background: rgba(15, 23, 42, 0.4) !important;
backdrop-filter: blur(20px) !important;
-webkit-backdrop-filter: blur(20px) !important;
border-right: 1px solid rgba(0, 240, 255, 0.15) !important;
box-shadow: 10px 0 40px rgba(0, 0, 0, 0.7) !important;
}
/* Inputs Styling */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > textarea,
.stSelectbox > div > div > div {
border-radius: 8px;
border: 1px solid rgba(255, 255, 255, 0.1) !important;
background: rgba(0, 0, 0, 0.3) !important;
color: #E2E8F0 !important;
transition: all 0.3s ease;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > textarea:focus,
.stSelectbox > div > div > div:focus {
box-shadow: inset 2px 2px 5px rgba(0,0,0,0.9), 0 0 10px rgba(0, 240, 255, 0.4) !important;
border: 1px solid rgba(0, 240, 255, 0.6) !important;
background: rgba(15, 23, 42, 0.8) !important;
}
/* Buttons */
.stButton > button {
background: linear-gradient(145deg, #0077FF, #00F0FF) !important;
color: #020617 !important;
font-weight: 800 !important;
border-radius: 8px !important;
border: none !important;
padding: 0.6rem 1.2rem !important;
transition: all 0.3s ease !important;
box-shadow: 0 4px 15px rgba(0, 119, 255, 0.4) !important;
}
.stButton > button:hover {
box-shadow: 0 6px 20px rgba(0, 240, 255, 0.6) !important;
transform: translateY(-2px);
}
/* Secondary Buttons */
button[kind="secondary"] {
background: rgba(30, 41, 59, 0.7) !important;
color: #00F0FF !important;
border: 1px solid rgba(0, 240, 255, 0.3) !important;
box-shadow: none !important;
}
button[kind="secondary"]:hover {
background: rgba(0, 240, 255, 0.1) !important;
border: 1px solid #00F0FF !important;
}
/* Glassmorphism Cards */
.css-1r6slb0, .css-12oz5g7 {
background: rgba(30, 41, 59, 0.4) !important;
backdrop-filter: blur(12px) !important;
border-radius: 12px !important;
border: 1px solid rgba(255, 255, 255, 0.05) !important;
padding: 1rem !important;
box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
}
/* Typography */
h1 { font-size: 2.5rem !important; color: #FFFFFF !important; font-weight: 800 !important; text-shadow: 0 4px 15px rgba(0, 240, 255, 0.3); }
h2 { font-size: 1.8rem !important; color: #E2E8F0 !important; }
h3 { font-size: 1.4rem !important; color: #94A3B8 !important; }
/* Metrics */
[data-testid="stMetricValue"] {
background: -webkit-linear-gradient(45deg, #00F0FF, #FFFFFF);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
font-size: 2.5rem !important;
font-weight: 800 !important;
}
/* Tabs */
button[data-baseweb="tab"] {
font-size: 1.1rem !important;
font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# --- NLP HELPERS (EXISTING) ---
def _extract_number(text, patterns, cast=float):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return cast(match.group(1))
            except (TypeError, ValueError):
                return None
    return None

def _positive_finding(text, term):
    negative = rf"\b(no|without|absent|negative for)\s+[\w\s,;:-]{{0,30}}\b{re.escape(term)}\b"
    positive = rf"\b{re.escape(term)}\b"
    return bool(re.search(positive, text, re.IGNORECASE)) and not bool(re.search(negative, text, re.IGNORECASE))

def parse_medical_report_text(text):
    parsed = {}
    normalized = re.sub(r"\s+", " ", text or "")
    bp = re.search(r"\b(?:b\.?p\.?|blood\s*pressure)\s*[:\-=]?\s*(\d{2,3})\s*/\s*(\d{2,3})", normalized, re.IGNORECASE)
    if bp:
        parsed["sbp"] = int(bp.group(1))
        parsed["dbp"] = int(bp.group(2))
    
    # Minimal version of field patterns just to keep functionality alive without bulking up the script
    # To keep it exact with original, I will include the full dict
    _SEP = r"[\s:=\-–—]*"
    _UNIT = r"(?:\s*(?:mg/dl|g/dl|u/l|iu/l|mm/hr|mmhg|bpm|/min|%|x10[³3]/[µu]l|meq/l|nmol/l|ng/ml|ng/dl|miu/l|mm|µmol/l|mmol/l|10\^3|thousand|k/ul|cells))?\b"
    field_patterns = {
        "sbp": ([r"(?:systolic(?:\s*(?:bp|blood\s*pressure))?|s\.?b\.?p)" + _SEP + r"(\d{2,3})" + _UNIT], int),
        "dbp": ([r"(?:diastolic(?:\s*(?:bp|blood\s*pressure))?|d\.?b\.?p)" + _SEP + r"(\d{2,3})" + _UNIT], int),
        "hr": ([r"(?:heart\s*rate|h\.?r|pulse\s*rate|pulse|p\.?r)" + _SEP + r"(\d{2,3})" + _UNIT], int),
        "rr": ([r"(?:respiratory\s*rate|respiration\s*rate|r\.?r|resp\s*rate|breathing\s*rate)" + _SEP + r"(\d{1,2})" + _UNIT], int),
        "spo2": ([r"(?:spo\s*2|sp\s*o2|oxygen\s*saturation|o2\s*sat(?:uration)?|sao2)" + _SEP + r"(\d{2,3})" + _UNIT], int),
        "temp": ([r"(?:temperature|temp|body\s*temp)" + _SEP + r"(\d{2,3}(?:\.\d+)?)"], float),
        "wbc": ([r"(?:white\s*blood\s*(?:cell|corpuscle)(?:\s*count)?|w\.?b\.?c(?:\.?\s*count)?|tlc|total\s*leukocyte\s*count|total\s*wbc)" + _SEP + r"(\d+(?:[.,]\d+)?(?:,\d{3})?)"], float),
        "scr": ([r"(?:serum\s*)?creatinine" + _SEP + r"(\d+(?:\.\d+)?)", r"s\.?\s*creatinine" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "crp": ([r"(?:c\s*[\-–]?\s*reactive\s*protein|c\.?r\.?p)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "trop": ([r"(?:troponin\s*[\-–]?\s*[it]?|trop(?:onin)?)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "age": ([r"\bage" + _SEP + r"(\d{1,3})(?:\s*(?:years?|yrs?|y\.?o\.?))?", r"(\d{1,3})\s*(?:years?|yrs?)\s*(?:old|male|female|m|f)\b"], int),
        "bmi": ([r"\bb\.?m\.?i" + _SEP + r"(\d+(?:\.\d+)?)", r"body\s*mass\s*index" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "charlson": ([r"(?:charlson(?:\s*comorbidity)?\s*(?:index|score|cci)?|comorbidity\s*(?:index|score))" + _SEP + r"(\d{1,2})"], int),
        "hemoglobin": ([r"(?:hemoglobin|haemoglobin|hb|hgb)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "platelets": ([r"(?:platelet(?:\s*count)?|plt|thrombocytes?)" + _SEP + r"(\d+(?:[.,]\d+)?)"], float),
        "esr": ([r"\be\.?s\.?r" + _SEP + r"(\d+(?:\.\d+)?)", r"erythrocyte\s*sedimentation\s*rate" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "pt_inr": ([r"(?:pt\s*/\s*inr|inr|prothrombin\s*(?:time\s*)?(?:ratio|index)?)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "aptt": ([r"(?:a\s*?ptt|aptt|activated\s*(?:partial\s*)?thromboplastin\s*time)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "glucose": ([r"(?:fasting\s*(?:blood\s*)?(?:glucose|sugar|bs)|fbs|fbg)" + _SEP + r"(\d+(?:\.\d+)?)", r"(?:blood\s*glucose|glucose)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "hba1c": ([r"(?:hba\s*1\s*c|a1c|glycated\s*haemoglobin|glyco(?:sylated)?\s*hb)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "tsh": ([r"\bt\.?s\.?h" + _SEP + r"(\d+(?:\.\d+)?)", r"thyroid\s*stimulating\s*hormone" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "t4": ([r"(?:free\s*t\s*4|f\.?t\s*4|t\s*4(?!\d))" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "calcium": ([r"(?:serum\s*)?(?:calcium|ca)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "vitamin_d": ([r"(?:vitamin\s*d(?:\s*3)?|25\s*(?:oh|hydroxy)\s*(?:vitamin\s*)?d|25-oh-d)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "egfr": ([r"(?:e\.?g\.?f\.?r|estimated\s*gfr|glomerular\s*filtration\s*rate)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "microalbumin": ([r"(?:micro\s*albumin(?:uria)?|urinary\s*albumin(?:\s*creatinine\s*ratio)?)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "alt": ([r"\ba\.?l\.?t" + _SEP + r"(\d+(?:\.\d+)?)", r"(?:alanine\s*(?:amino)?transaminase|sgpt)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "ast": ([r"\ba\.?s\.?t" + _SEP + r"(\d+(?:\.\d+)?)", r"(?:aspartate\s*(?:amino)?transaminase|sgot)" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "alp": ([r"\ba\.?l\.?p" + _SEP + r"(\d+(?:\.\d+)?)", r"alkaline\s*phosphatase" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "albumin": ([r"(?:serum\s*)?\balbumin" + _SEP + r"(\d+(?:\.\d+)?)"], float),
        "bilirubin": ([r"(?:total\s*)?bilirubin" + _SEP + r"(\d+(?:\.\d+)?)", r"t\.?\s*bili(?:rubin)?" + _SEP + r"(\d+(?:\.\d+)?)"], float),
    }

    for field, (patterns, cast) in field_patterns.items():
        if field not in parsed:
            value = _extract_number(normalized, patterns, cast)
            if value is None:
                value = _extract_number(normalized, patterns, str)
                if value is not None:
                    try: value = cast(str(value).replace(",", ""))
                    except (ValueError, TypeError): value = None
            if value is not None: parsed[field] = value

    finding_terms = {"infiltrates": "infiltrates", "cardiomegaly": "cardiomegaly", "pleural_effusion": "pleural effusion", "consolidation": "consolidation", "lung_nodule": "lung nodule"}
    for field, term in finding_terms.items():
        if _positive_finding(normalized, term): parsed[field] = True

    if normalized: parsed["notes"] = normalized[:1500]
    return parsed

def apply_scan_values(values):
    widget_keys = {
        "sbp": "scan_sbp", "dbp": "scan_dbp", "hr": "scan_hr", "rr": "scan_rr", "spo2": "scan_spo2",
        "temp": "scan_temp", "wbc": "scan_wbc", "scr": "scan_scr", "crp": "scan_crp", "trop": "scan_trop",
        "age": "scan_age", "bmi": "scan_bmi", "charlson": "scan_charlson", "infiltrates": "scan_infiltrates",
        "cardiomegaly": "scan_cardiomegaly", "pleural_effusion": "scan_pleural_effusion", "consolidation": "scan_consolidation",
        "lung_nodule": "scan_lung_nodule", "hemoglobin": "scan_hemoglobin", "platelets": "scan_platelets",
        "esr": "scan_esr", "pt_inr": "scan_pt_inr", "aptt": "scan_aptt", "glucose": "scan_glucose",
        "hba1c": "scan_hba1c", "tsh": "scan_tsh", "t4": "scan_t4", "calcium": "scan_calcium",
        "vitamin_d": "scan_vitamin_d", "egfr": "scan_egfr", "microalbumin": "scan_microalbumin", "alt": "scan_alt",
        "ast": "scan_ast", "alp": "scan_alp", "albumin": "scan_albumin", "bilirubin": "scan_bilirubin", "notes": "scan_notes"
    }
    bounds = {
        "sbp": (50, 220), "dbp": (30, 130), "hr": (40, 180), "rr": (8, 40), "spo2": (70, 100),
        "temp": (35.0, 42.0), "wbc": (1.0, 30.0), "scr": (0.2, 10.0), "crp": (0.0, 200.0), "trop": (0.0, 50.0),
        "age": (1, 110), "bmi": (10.0, 50.0), "charlson": (0, 12), "hemoglobin": (5.0, 20.0), "platelets": (10.0, 500.0),
        "esr": (0.0, 100.0), "pt_inr": (0.8, 8.0), "aptt": (15.0, 50.0), "glucose": (40.0, 500.0), "hba1c": (4.0, 14.0),
        "tsh": (0.0, 10.0), "t4": (0.0, 5.0), "calcium": (6.0, 12.0), "vitamin_d": (0.0, 150.0), "egfr": (10.0, 120.0),
        "microalbumin": (0.0, 300.0), "alt": (5.0, 500.0), "ast": (5.0, 500.0), "alp": (10.0, 300.0), "albumin": (1.0, 6.0), "bilirubin": (0.0, 10.0)
    }
    for field, key in widget_keys.items():
        if field in values:
            value = values[field]
            if field in bounds:
                lo, hi = bounds[field]
                value = min(hi, max(lo, value))
            st.session_state[key] = value

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=60)
    st.markdown("## AI Command Center")
    mode = st.radio("Navigation", [
        "🏠 Main Dashboard", 
        "➕ New AI Analysis", 
        "👥 Patient Management", 
        "📜 Analysis History", 
        "📚 Clinical Reference",
        "📂 Batch Processing (CSV)", 
        "🗄️ Database Inspector"
    ])
    
    st.markdown("---")
    
    with st.popover("💬 Open AI Assistant", use_container_width=True):
        st.markdown("### 🤖 AI Medical Assistant")
        
        lang = st.selectbox("Language / ભાષા", ["English 🇺🇸", "Gujarati 🕉️", "Hindi 🇮🇳"], label_visibility="collapsed", key="chat_lang")
        
        if not os.getenv("GEMINI_API_KEY"):
            st.error("⚠️ Missing 'GEMINI_API_KEY' in backend `.env` file.")
        else:
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = [{"role": "model", "content": "Hello Doctor. How can I assist you with the patient? (નમસ્તે ડૉક્ટર, હું પેશન્ટ વિશે તમારી શું મદદ કરી શકું?)"}]
                
            patient_context = "No specific patient data is currently active in the form."
            has_data = False
            context_lines = []
            for key in ["scan_age", "scan_sbp", "scan_dbp", "scan_hr", "scan_spo2", "scan_temp", "scan_wbc", "scan_hemoglobin"]:
                if key in st.session_state and st.session_state[key] != 0:
                    context_lines.append(f"- {key.replace('scan_', '').upper()}: {st.session_state[key]}")
                    has_data = True
                    
            if has_data:
                patient_context = "Current Dashboard Patient Data:\n" + "\n".join(context_lines)
                if "prediction_result" in st.session_state:
                    patient_context += f"\nAI Model Prediction Risk Score: {st.session_state['prediction_result'].get('risk_score', 'N/A')}\n"
            else:
                patient_context = "The doctor hasn't loaded a patient yet."

            chat_container = st.container(height=400)
            with chat_container:
                for msg in st.session_state.chat_messages:
                    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                        st.markdown(msg["content"])
                        
            if prompt := st.chat_input("Ask a medical query..."):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = get_chat_response(st.session_state.chat_messages, patient_context, lang)
                            st.markdown(response)
                    st.session_state.chat_messages.append({"role": "model", "content": response})

    st.markdown("---")
    st.markdown("### System Status")
    st.markdown("🟢 AI Engine Online")
    st.markdown("🟢 Database Connected")
    st.markdown("🟢 Clinical Module Ready")

# --- VIEWS ---

if mode == "🏠 Main Dashboard":
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- REAL DATA FETCHING ---
    patients = db.get_all_patients()
    all_assessments = []
    for p in patients:
        all_assessments.extend(db.get_patient_assessments(p['patient_id']))
    all_assessments.sort(key=lambda x: x.get('assessment_date', ''), reverse=True)
    high_risk = len([a for a in all_assessments if 'CRITICAL' in str(a.get('severity_level', '')).upper() or 'CRITICAL' in str(a.get('risk_category', '')).upper() or 'CRITICAL' in str(a.get('diagnosis', '')).upper()])
    
    # --- GLASSMORPHISM STATISTICS ---
    st.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
<div class="glass-stat" style="background: rgba(15,23,42,0.6); border: 1px solid rgba(0,240,255,0.2); border-radius: 16px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); backdrop-filter: blur(10px); transition: transform 0.3s; cursor: default;" onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,240,255,0.2)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 10px 30px rgba(0,0,0,0.5)';">
<div style="color: #94A3B8; font-size: 14px; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px;">👥 REGISTERED PATIENTS</div>
<div style="color: #FFFFFF; font-size: 36px; font-weight: 800; text-shadow: 0 0 15px rgba(0,240,255,0.4);">{len(patients)}</div>
</div>
<div class="glass-stat" style="background: rgba(15,23,42,0.6); border: 1px solid rgba(0,240,255,0.2); border-radius: 16px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); backdrop-filter: blur(10px); transition: transform 0.3s; cursor: default;" onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,240,255,0.2)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 10px 30px rgba(0,0,0,0.5)';">
<div style="color: #94A3B8; font-size: 14px; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px;">🧠 AI ANALYSES RUN</div>
<div style="color: #FFFFFF; font-size: 36px; font-weight: 800; text-shadow: 0 0 15px rgba(0,240,255,0.4);">{len(all_assessments)}</div>
</div>
<div class="glass-stat" style="background: rgba(15,23,42,0.6); border: 1px solid rgba(239,85,59,0.3); border-radius: 16px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); backdrop-filter: blur(10px); transition: transform 0.3s; cursor: default;" onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(239,85,59,0.2)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 10px 30px rgba(0,0,0,0.5)';">
<div style="color: #94A3B8; font-size: 14px; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px;">⚠️ HIGH RISK CASES</div>
<div style="color: #ef553b; font-size: 36px; font-weight: 800; text-shadow: 0 0 15px rgba(239,85,59,0.6);">{high_risk}</div>
</div>
<div class="glass-stat" style="background: rgba(15,23,42,0.6); border: 1px solid rgba(0,204,150,0.3); border-radius: 16px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); backdrop-filter: blur(10px); transition: transform 0.3s; cursor: default;" onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 15px 35px rgba(0,204,150,0.2)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 10px 30px rgba(0,0,0,0.5)';">
<div style="color: #94A3B8; font-size: 14px; font-weight: 700; letter-spacing: 1px; margin-bottom: 10px;">🟢 SYSTEM STATUS</div>
<div style="color: #00cc96; font-size: 36px; font-weight: 800; text-shadow: 0 0 15px rgba(0,204,150,0.6);">ONLINE</div>
</div>
</div>""", unsafe_allow_html=True)
    
    # --- BEST MEDICAL AI HERO ANIMATION ---
    st.markdown("""
<style>
@keyframes ecgMove { 0% { stroke-dashoffset: 600; } 100% { stroke-dashoffset: 0; } }
@keyframes coreGlow {
  0% { box-shadow: 0 0 10px #00F0FF, 0 0 20px rgba(0,240,255,0.15), inset 0 0 10px rgba(0,240,255,0.05); }
  100% { box-shadow: 0 0 25px #00F0FF, 0 0 50px rgba(0,85,255,0.3), 0 0 70px rgba(0,240,255,0.1), inset 0 0 20px rgba(0,240,255,0.15); }
}
@keyframes coreScale {
  0%, 20% { transform: scale(1);    box-shadow: 0 0 15px #00F0FF; filter: hue-rotate(0deg); }
  46%  { transform: scale(1.25); box-shadow: 0 0 40px #00F0FF, 0 0 80px rgba(0,240,255,0.5); filter: hue-rotate(0deg); }
  51%  { transform: scale(0.8);  box-shadow: 0 0 8px #a855f7; filter: hue-rotate(150deg); }
  55%  { transform: scale(1.05); box-shadow: 0 0 20px #00cc96; filter: hue-rotate(280deg); }
  61%  { transform: scale(1.2);  box-shadow: 0 0 35px #00F0FF, 0 0 70px rgba(0,85,255,0.6); filter: hue-rotate(360deg); }
  93%, 100% { transform: scale(1);    box-shadow: 0 0 15px #00F0FF; filter: hue-rotate(360deg); }
}
@keyframes rippleOut {
  0%   { r: 10; opacity: 0.9; stroke-width: 3; }
  100% { r: 120; opacity: 0; stroke-width: 0.5; }
}
@keyframes pathPulseIn {
  0%, 60%, 100% { opacity: 0.15; }
  25%           { opacity: 0.7; }
}
@keyframes pathPulseOut {
  0%, 50%, 100% { opacity: 0.15; }
  75%           { opacity: 0.7; }
}
@keyframes neuronPulse { 0%,100% { opacity:0.4; transform:scale(1); } 50% { opacity:1; transform:scale(1.4); box-shadow:0 0 15px currentColor; } }
@keyframes spinFwd { to { transform: rotate(360deg); } }
@keyframes spinRev { to { transform: rotate(-360deg); } }
@keyframes icuAlarm {
  0%, 85%, 100% { box-shadow:0 0 10px rgba(239,85,59,0.2); border-color:rgba(239,85,59,0.4); transform: scale(1); }
  93% { box-shadow:0 0 40px rgba(239,85,59,0.8), 0 0 80px rgba(239,85,59,0.4); border-color:#ef553b; transform: scale(1.15); }
}
@keyframes stableGlow {
  0%, 85%, 100% { box-shadow:0 0 8px rgba(0,204,150,0.2); transform: scale(1); }
  93% { box-shadow:0 0 30px rgba(0,204,150,0.8); border-color:#00cc96; transform: scale(1.15); }
}
@keyframes inputPulse {
  0%, 20%, 100% { opacity:0.3; transform:scale(1); box-shadow:none; }
  5% { opacity:1; transform:scale(1.4); box-shadow:0 0 20px currentColor; }
}
@keyframes floatUp { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-8px); } }
@keyframes dataFlow { 0% { opacity:0; transform:scaleX(0); transform-origin:left; } 50% { opacity:1; transform:scaleX(1); } 100% { opacity:0; transform:scaleX(1); } }
@keyframes slideLeft { 0% { stroke-dashoffset:500; } 100% { stroke-dashoffset:0; } }
@keyframes vitalsScroll { 0% { transform:translateX(0); } 100% { transform:translateX(-50%); } }
@keyframes fadeInUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@keyframes scanLine { 0%,100% { top:0%; opacity:0.6; } 50% { top:90%; opacity:0.2; } }
.hero-wrapper { position:relative; width:100%; height:620px; background:radial-gradient(ellipse at 50% 100%, #060f1e 0%, #010410 60%); border-radius:20px; border:1px solid rgba(0,240,255,0.08); box-shadow:0 0 80px rgba(0,0,0,0.95), inset 0 1px 0 rgba(255,255,255,0.03); overflow:hidden; }
.hero-grid { position:absolute; inset:0; background-image:linear-gradient(rgba(0,240,255,0.04) 1px,transparent 1px), linear-gradient(90deg,rgba(0,240,255,0.04) 1px,transparent 1px); background-size:40px 40px; }
.scan-line { position:absolute; left:0; width:100%; height:2px; background:linear-gradient(90deg,transparent,rgba(0,240,255,0.3),transparent); animation:scanLine 4s ease-in-out infinite; z-index:2; }
.hero-title-wrap { position:absolute; top:28px; left:50%; transform:translateX(-50%); text-align:center; z-index:20; animation:fadeInUp 1s ease; }
.hero-badge { display:inline-block; background:rgba(0,240,255,0.1); border:1px solid rgba(0,240,255,0.3); border-radius:50px; padding:4px 16px; color:#00F0FF; font-size:11px; font-weight:800; letter-spacing:4px; margin-bottom:10px; }
.hero-title { color:#fff; font-size:28px; font-weight:900; letter-spacing:1px; text-shadow:0 4px 30px rgba(0,240,255,0.5); }
.hero-subtitle { color:rgba(255,255,255,0.5); font-size:13px; margin-top:4px; letter-spacing:2px; }

/* ECG Banner */
.ecg-bar { position:absolute; bottom:0; left:0; width:100%; height:60px; z-index:3; overflow:hidden; }
.ecg-bar svg { width:200%; height:100%; animation:vitalsScroll 6s linear infinite; }

/* Input Nodes */
.input-node { position:absolute; display:flex; flex-direction:column; align-items:center; gap:6px; z-index:15; animation:floatUp 3s ease-in-out infinite; }
.node-icon { width:56px; height:56px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:24px; border:2px solid; position:relative; }
.node-icon::after { content:''; position:absolute; inset:-4px; border-radius:18px; border:1px solid; opacity:0.3; animation:neuronPulse 2s infinite; }
.node-label { font-size:10px; font-weight:800; letter-spacing:2px; }

/* AI Core */
.ai-core-wrap { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:180px; height:180px; display:flex; align-items:center; justify-content:center; z-index:15; }
.ring-outer { position:absolute; width:180px; height:180px; border-radius:50%; border:2px solid transparent; border-top:2px solid #00F0FF; border-right:2px solid rgba(0,240,255,0.3); animation:spinFwd 4s linear infinite; }
.ring-mid { position:absolute; width:148px; height:148px; border-radius:50%; border:2px solid transparent; border-bottom:2px solid #a855f7; border-left:2px solid rgba(168,85,247,0.3); animation:spinRev 3s linear infinite; }
.ring-inner { position:absolute; width:116px; height:116px; border-radius:50%; border:1px dashed rgba(0,240,255,0.25); animation:spinFwd 8s linear infinite; }
.core-ball { width:88px; height:88px; background:radial-gradient(circle at 35% 35%, rgba(255,255,255,0.9) 0%, #00F0FF 30%, #0055FF 65%, #020617 100%); border-radius:50%; animation:coreScale 6s ease-in-out infinite; display:flex; align-items:center; justify-content:center; position:relative; z-index:20; }
.core-inner-text { font-size:10px; font-weight:900; color:#fff; letter-spacing:1px; text-shadow:0 0 10px #fff; }
.core-label { position:absolute; top:107%; left:50%; transform:translateX(-50%); white-space:nowrap; color:#00F0FF; font-size:11px; font-weight:800; letter-spacing:3px; text-shadow:0 0 10px #00F0FF; animation:neuronPulse 1.5s infinite; }

/* Neuron dots on ring */
.neuron { position:absolute; width:10px; height:10px; border-radius:50%; animation:neuronPulse 1.5s infinite; }

/* Output Cards */
.output-card { position:absolute; display:flex; flex-direction:column; align-items:center; gap:6px; z-index:15; }
.out-box { width:76px; height:76px; border-radius:16px; display:flex; align-items:center; justify-content:center; font-size:30px; border:2px solid; }
.out-label { font-size:11px; font-weight:800; text-align:center; line-height:1.4; white-space:nowrap; }
</style>

<div class="hero-wrapper">
<div class="hero-grid"></div>
<div class="scan-line"></div>

<!-- Title -->
<div style="position:absolute; top:28px; left:0; right:0; text-align:center; z-index:20; padding:0 20px;">
  <div style="color:#ffffff; font-size:26px; font-weight:900; letter-spacing:1px; text-shadow:0 0 30px rgba(0,240,255,0.6), 0 0 60px rgba(0,240,255,0.2);">Multimodal Patient Triage &amp; Decision Engine</div>
</div>

<!-- SVG Paths + Animated Data Packets -->
<svg style="position:absolute;top:0;left:0;width:100%;height:100%;z-index:5;pointer-events:none;" viewBox="0 0 900 620" preserveAspectRatio="none">
  <defs>
    <filter id="glow-c"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <filter id="glow-r"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>

  <!-- Input paths to core (pulsing glow) -->
  <path id="p1" d="M 130,120 C 250,120 350,310 420,310" fill="none" stroke="rgba(0,240,255,0.8)" stroke-width="2" stroke-dasharray="6 4" style="animation:pathPulseIn 6s infinite;"/>
  <path id="p2" d="M 130,310 L 420,310"               fill="none" stroke="rgba(168,85,247,0.8)" stroke-width="2" stroke-dasharray="6 4" style="animation:pathPulseIn 6s infinite;"/>
  <path id="p3" d="M 130,500 C 250,500 350,310 420,310" fill="none" stroke="rgba(0,204,150,0.8)" stroke-width="2" stroke-dasharray="6 4" style="animation:pathPulseIn 6s infinite;"/>

  <!-- Output paths from core -->
  <path id="p4" d="M 480,310 C 610,310 680,140 770,140" fill="none" stroke="rgba(239,85,59,0.8)"  stroke-width="3" stroke-dasharray="8 5" style="animation:pathPulseOut 6s infinite;"/>
  <path id="p5" d="M 480,310 C 610,310 680,480 770,480" fill="none" stroke="rgba(0,204,150,0.8)" stroke-width="3" stroke-dasharray="8 5" style="animation:pathPulseOut 6s infinite;"/>

  <!-- ===== ONE FULL SEQUENTIAL CYCLE (6s total) ===== -->
  <!-- ALL begin="0s" so the 6s loop stays perfectly synchronized. Delay is handled by keyTimes -->
  
  <!-- Clinical IN -->
  <circle r="6" fill="#00F0FF" filter="url(#glow-c)">
    <animateMotion dur="6s" repeatCount="indefinite" keyPoints="0;1;1" keyTimes="0;0.466;1" calcMode="linear"><mpath href="#p1"/></animateMotion>
    <animate attributeName="opacity" values="0;1;1;0;0" keyTimes="0;0.033;0.433;0.466;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>
  <circle r="3" fill="#00F0FF" filter="url(#glow-c)">
    <animateMotion dur="6s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;0.058;0.525;1" calcMode="linear"><mpath href="#p1"/></animateMotion>
    <animate attributeName="opacity" values="0;0;0.45;0.45;0;0" keyTimes="0;0.058;0.091;0.491;0.525;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>

  <!-- Lab IN -->
  <circle r="6" fill="#a855f7" filter="url(#glow-c)">
    <animateMotion dur="6s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;0.033;0.5;1" calcMode="linear"><mpath href="#p2"/></animateMotion>
    <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.033;0.066;0.466;0.5;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>
  <circle r="3" fill="#a855f7" filter="url(#glow-c)">
    <animateMotion dur="6s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;0.091;0.558;1" calcMode="linear"><mpath href="#p2"/></animateMotion>
    <animate attributeName="opacity" values="0;0;0.45;0.45;0;0" keyTimes="0;0.091;0.125;0.525;0.558;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>

  <!-- Imaging IN -->
  <circle r="6" fill="#00cc96" filter="url(#glow-c)">
    <animateMotion dur="6s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;0.066;0.533;1" calcMode="linear"><mpath href="#p3"/></animateMotion>
    <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.066;0.1;0.5;0.533;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>
  <circle r="3" fill="#00cc96" filter="url(#glow-c)">
    <animateMotion dur="6s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;0.125;0.591;1" calcMode="linear"><mpath href="#p3"/></animateMotion>
    <animate attributeName="opacity" values="0;0;0.45;0.45;0;0" keyTimes="0;0.125;0.158;0.558;0.591;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>

  <!-- Processing Ripple Rings (fire at 2.8s = 46%) -->
  <circle cx="450" cy="310" r="10" fill="none" stroke="rgba(0,240,255,0.9)" stroke-width="3">
    <animate attributeName="r"       values="10;10;110;110" keyTimes="0;0.466;0.933;1" dur="6s" begin="0s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0;0;0.9;0;0"   keyTimes="0;0.465;0.466;0.933;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>
  <circle cx="450" cy="310" r="10" fill="none" stroke="rgba(168,85,247,0.7)" stroke-width="2">
    <animate attributeName="r"       values="10;10;90;90"  keyTimes="0;0.5;0.933;1" dur="6s" begin="0s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0;0;0.7;0;0"  keyTimes="0;0.499;0.5;0.933;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>
  <circle cx="450" cy="310" r="10" fill="none" stroke="rgba(0,204,150,0.5)" stroke-width="1.5">
    <animate attributeName="r"       values="10;10;70;70"  keyTimes="0;0.533;0.933;1" dur="6s" begin="0s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0;0;0.5;0;0"  keyTimes="0;0.532;0.533;0.933;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>

  <!-- ICU OUT (fires at 3.4s = 56%, arrives at 5.6s = 93%) -->
  <circle r="7" fill="#ef553b" filter="url(#glow-r)">
    <animateMotion dur="6s" begin="0s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;0.566;0.933;1" calcMode="linear"><mpath href="#p4"/></animateMotion>
    <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.566;0.583;0.916;0.933;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#ef553b" filter="url(#glow-r)">
    <animateMotion dur="6s" begin="0s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;0.616;0.983;1" calcMode="linear"><mpath href="#p4"/></animateMotion>
    <animate attributeName="opacity" values="0;0;0.45;0.45;0;0" keyTimes="0;0.616;0.633;0.966;0.983;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>

  <!-- Stable OUT -->
  <circle r="7" fill="#00cc96" filter="url(#glow-r)">
    <animateMotion dur="6s" begin="0s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;0.566;0.933;1" calcMode="linear"><mpath href="#p5"/></animateMotion>
    <animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="0;0.566;0.583;0.916;0.933;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>
  <circle r="4" fill="#00cc96" filter="url(#glow-r)">
    <animateMotion dur="6s" begin="0s" repeatCount="indefinite" keyPoints="0;0;1;1" keyTimes="0;0.616;0.983;1" calcMode="linear"><mpath href="#p5"/></animateMotion>
    <animate attributeName="opacity" values="0;0;0.45;0.45;0;0" keyTimes="0;0.616;0.633;0.966;0.983;1" dur="6s" begin="0s" repeatCount="indefinite"/>
  </circle>
  <!-- Stable OUT - trail -->
  <circle r="4" fill="#00cc96" filter="url(#glow-r)">
    <animateMotion dur="1.4s" begin="2.0s" repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1" calcMode="linear"><mpath href="#p5"/></animateMotion>
    <animate attributeName="opacity" values="0;0.5;0.5;0" keyTimes="0;0.05;0.9;1" dur="1.4s" begin="2.0s" repeatCount="indefinite"/>
  </circle>
</svg>

<!-- INPUT NODES (fixed positions, no float animation conflict) -->
<div style="position:absolute; top:135px; left:40px; display:flex; flex-direction:column; align-items:center; gap:6px; z-index:15;">
  <div style="width:56px; height:56px; border-radius:14px; background:rgba(0,240,255,0.08); border:2px solid #00F0FF; display:flex; align-items:center; justify-content:center; font-size:24px; box-shadow:0 0 15px rgba(0,240,255,0.3); position:relative;">📋
    <span style="position:absolute;inset:-5px;border-radius:18px;border:1px solid #00F0FF;opacity:0.3;animation:inputPulse 6s infinite;"></span>
  </div>
  <div style="color:#00F0FF; font-size:10px; font-weight:800; letter-spacing:2px; text-align:center;">CLINICAL<br>NOTES</div>
</div>

<div style="position:absolute; top:270px; left:40px; display:flex; flex-direction:column; align-items:center; gap:6px; z-index:15;">
  <div style="width:56px; height:56px; border-radius:14px; background:rgba(168,85,247,0.08); border:2px solid #a855f7; display:flex; align-items:center; justify-content:center; font-size:24px; box-shadow:0 0 15px rgba(168,85,247,0.3); position:relative;">🧪
    <span style="position:absolute;inset:-5px;border-radius:18px;border:1px solid #a855f7;opacity:0.3;animation:inputPulse 6s infinite;"></span>
  </div>
  <div style="color:#a855f7; font-size:10px; font-weight:800; letter-spacing:2px; text-align:center;">LAB<br>RESULTS</div>
</div>

<div style="position:absolute; top:405px; left:40px; display:flex; flex-direction:column; align-items:center; gap:6px; z-index:15;">
  <div style="width:56px; height:56px; border-radius:14px; background:rgba(0,204,150,0.08); border:2px solid #00cc96; display:flex; align-items:center; justify-content:center; font-size:24px; box-shadow:0 0 15px rgba(0,204,150,0.3); position:relative;">🩻
    <span style="position:absolute;inset:-5px;border-radius:18px;border:1px solid #00cc96;opacity:0.3;animation:inputPulse 6s infinite;"></span>
  </div>
  <div style="color:#00cc96; font-size:10px; font-weight:800; letter-spacing:2px; text-align:center;">MEDICAL<br>IMAGING</div>
</div>

<!-- AI CORE -->
<div class="ai-core-wrap" style="top:50%;left:50%;">
  <div class="ring-outer"></div>
  <div class="ring-mid"></div>
  <div class="ring-inner"></div>
  <!-- Neuron nodes on outer ring -->
  <div class="neuron" style="top:-5px;left:50%;margin-left:-5px;background:#00F0FF;color:#00F0FF;box-shadow:0 0 10px #00F0FF;"></div>
  <div class="neuron" style="bottom:-5px;left:50%;margin-left:-5px;background:#a855f7;color:#a855f7;box-shadow:0 0 10px #a855f7;animation-delay:0.5s;"></div>
  <div class="neuron" style="left:-5px;top:50%;margin-top:-5px;background:#00cc96;color:#00cc96;box-shadow:0 0 10px #00cc96;animation-delay:1s;"></div>
  <div class="neuron" style="right:-5px;top:50%;margin-top:-5px;background:#ef553b;color:#ef553b;box-shadow:0 0 10px #ef553b;animation-delay:1.5s;"></div>
  <div class="core-ball">
    <div class="core-inner-text">AI<br>CORE</div>
  </div>
  <div class="core-label">● ANALYZING ●</div>
</div>

<!-- OUTPUT CARDS -->
<div class="output-card" style="top:140px;right:55px;animation:icuAlarm 6s infinite;">
  <div class="out-box" style="background:rgba(239,85,59,0.12);border-color:#ef553b;box-shadow:0 0 30px rgba(239,85,59,0.4);">🚑</div>
  <div class="out-label" style="color:#ef553b;">⚠ HIGH RISK<br>→ ADMIT ICU</div>
</div>
<div class="output-card" style="bottom:140px;right:55px;animation:stableGlow 6s infinite;">
  <div class="out-box" style="background:rgba(0,204,150,0.12);border-color:#00cc96;box-shadow:0 0 20px rgba(0,204,150,0.3);">👨‍⚕️</div>
  <div class="out-label" style="color:#00cc96;">✓ STABLE<br>→ ROUTINE</div>
</div>

<!-- ECG Bottom Banner -->
<div class="ecg-bar">
  <svg viewBox="0 0 900 60" preserveAspectRatio="none">
    <polyline points="0,40 60,40 80,40 90,10 100,55 110,5 120,50 130,40 200,40 260,40 280,40 290,10 300,55 310,5 320,50 330,40 400,40 460,40 480,40 490,10 500,55 510,5 520,50 530,40 600,40 660,40 680,40 690,10 700,55 710,5 720,50 730,40 800,40 860,40 880,40 890,10 900,55"
      fill="none" stroke="#00F0FF" stroke-width="2" opacity="0.5"
      stroke-dasharray="600" stroke-dashoffset="600"
      style="animation:ecgMove 3s linear infinite;"/>
    <polyline points="0,40 60,40 80,40 90,10 100,55 110,5 120,50 130,40 200,40 260,40 280,40 290,10 300,55 310,5 320,50 330,40 400,40 460,40 480,40 490,10 500,55 510,5 520,50 530,40 600,40 660,40 680,40 690,10 700,55 710,5 720,50 730,40 800,40 860,40 880,40 890,10 900,55"
      fill="none" stroke="#00F0FF" stroke-width="1.5" opacity="0.15"/>
  </svg>
</div>
</div>""", unsafe_allow_html=True)

    # --- START NEW AI ANALYSIS CTA ---
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
<style>
.cta-button {
display: block; width: 100%; text-align: center; background: linear-gradient(90deg, #0055FF, #00F0FF);
color: #000; font-size: 20px; font-weight: 900; letter-spacing: 2px; padding: 20px;
border-radius: 16px; text-decoration: none; box-shadow: 0 10px 30px rgba(0,240,255,0.4);
transition: all 0.3s ease; border: 2px solid rgba(255,255,255,0.2); cursor: pointer;
}
.cta-button:hover { transform: translateY(-5px) scale(1.02); box-shadow: 0 15px 40px rgba(0,240,255,0.6); border: 2px solid #FFFFFF; }
</style>""", unsafe_allow_html=True)
        if st.button("🚀 START NEW AI ANALYSIS", type="primary", use_container_width=True):
            st.info("Please select '➕ New AI Analysis' from the sidebar to begin.")
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # --- HOW OUR AI WORKS PIPELINE ---
    st.markdown("""
<div style="color: #FFFFFF; font-size: 20px; font-weight: 800; letter-spacing: 2px; margin-bottom: 20px; padding-left: 10px; border-left: 4px solid #00F0FF;">HOW OUR MEDICAL SYSTEM WORKS</div>
<div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15,23,42,0.4); padding: 30px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 40px; position: relative;">
<!-- Connecting Line -->
<div style="position: absolute; top: 50%; left: 5%; right: 5%; height: 2px; background: rgba(0,240,255,0.2); z-index: 1;"></div>
<!-- Nodes -->
<div class="pipe-node" style="z-index: 2; display: flex; flex-direction: column; align-items: center; gap: 10px; background: #0f172a; padding: 10px; border-radius: 12px; border: 1px solid rgba(0,240,255,0.4); transition: transform 0.3s;" onmouseover="this.style.transform='translateY(-5px)';" onmouseout="this.style.transform='translateY(0)';">
<div style="color: #00F0FF; font-size: 24px; font-weight: 900;">01</div>
<div style="color: #E2E8F0; font-size: 12px; font-weight: 600;">Patient Data</div>
</div>
<div class="pipe-node" style="z-index: 2; display: flex; flex-direction: column; align-items: center; gap: 10px; background: #0f172a; padding: 10px; border-radius: 12px; border: 1px solid rgba(0,240,255,0.4); transition: transform 0.3s;" onmouseover="this.style.transform='translateY(-5px)';" onmouseout="this.style.transform='translateY(0)';">
<div style="color: #00F0FF; font-size: 24px; font-weight: 900;">02</div>
<div style="color: #E2E8F0; font-size: 12px; font-weight: 600;">Clinical & Lab Features</div>
</div>
<div class="pipe-node" style="z-index: 2; display: flex; flex-direction: column; align-items: center; gap: 10px; background: #0f172a; padding: 10px; border-radius: 12px; border: 1px solid rgba(0,240,255,0.4); transition: transform 0.3s;" onmouseover="this.style.transform='translateY(-5px)';" onmouseout="this.style.transform='translateY(0)';">
<div style="color: #00F0FF; font-size: 24px; font-weight: 900;">03</div>
<div style="color: #E2E8F0; font-size: 12px; font-weight: 600;">Medical Image Features</div>
</div>
<div class="pipe-node" style="z-index: 2; display: flex; flex-direction: column; align-items: center; gap: 10px; background: #0f172a; padding: 10px; border-radius: 12px; border: 1px solid rgba(0,240,255,0.8); box-shadow: 0 0 15px rgba(0,240,255,0.4); transition: transform 0.3s;" onmouseover="this.style.transform='translateY(-5px)';" onmouseout="this.style.transform='translateY(0)';">
<div style="color: #00F0FF; font-size: 24px; font-weight: 900;">04</div>
<div style="color: #E2E8F0; font-size: 12px; font-weight: 600; text-shadow: 0 0 10px #00F0FF;">Multimodal Fusion</div>
</div>
<div class="pipe-node" style="z-index: 2; display: flex; flex-direction: column; align-items: center; gap: 10px; background: #0f172a; padding: 10px; border-radius: 12px; border: 1px solid #00cc96; box-shadow: 0 0 15px rgba(0,204,150,0.4); transition: transform 0.3s;" onmouseover="this.style.transform='translateY(-5px)';" onmouseout="this.style.transform='translateY(0)';">
<div style="color: #00cc96; font-size: 24px; font-weight: 900;">05</div>
<div style="color: #E2E8F0; font-size: 12px; font-weight: 600; text-shadow: 0 0 10px #00cc96;">AI Prediction</div>
</div>
</div>""", unsafe_allow_html=True)
    
    # --- RECENT ANALYSIS (REAL DATA) ---
    st.markdown("<div style='color: #FFFFFF; font-size: 20px; font-weight: 800; letter-spacing: 2px; margin-bottom: 20px; padding-left: 10px; border-left: 4px solid #00F0FF;'>RECENT ANALYSIS</div>", unsafe_allow_html=True)
    
    if len(all_assessments) > 0:
        recent_5 = all_assessments[:5]
        
        # Process data safely using list comprehensions and dict.get()
        # The database column is 'model_confidence', not 'confidence'.
        # We handle any missing keys gracefully to avoid KeyErrors.
        df_display = pd.DataFrame({
            'Patient ID': [f"P-{a.get('patient_id', 0):03d}" if isinstance(a.get('patient_id'), int) else str(a.get('patient_id', 'N/A')) for a in recent_5],
            'Date': [str(a.get('assessment_date', 'N/A')).split('.')[0] for a in recent_5],
            'Prediction': [a.get('diagnosis') or a.get('risk_category') or "Pending" for a in recent_5],
            'Confidence': [f"{float(a.get('model_confidence'))*100:.1f}%" if a.get('model_confidence') is not None else "N/A" for a in recent_5],
            'Status': [a.get('severity_level') or 'N/A' for a in recent_5]
        })
        
        st.dataframe(
            df_display, 
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No recent AI analyses found. The system is ready for the first patient.")

elif mode == "➕ New AI Analysis":
    st.title("New Multimodal AI Analysis")
    
    # Wizard State Management
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
        
    st.progress((st.session_state.wizard_step - 1) / 6.0)
    
    cols = st.columns(7)
    cols[0].markdown("**1. Patient**" if st.session_state.wizard_step == 1 else "1. Patient")
    cols[1].markdown("**2. Scan**" if st.session_state.wizard_step == 2 else "2. Scan")
    cols[2].markdown("**3. Vitals**" if st.session_state.wizard_step == 3 else "3. Vitals")
    cols[3].markdown("**4. Labs**" if st.session_state.wizard_step == 4 else "4. Labs")
    cols[4].markdown("**5. Vision**" if st.session_state.wizard_step == 5 else "5. Vision")
    cols[5].markdown("**6. Process**" if st.session_state.wizard_step == 6 else "6. Process")
    cols[6].markdown("**7. Result**" if st.session_state.wizard_step == 7 else "7. Result")
    
    st.markdown("---")
    
    if st.session_state.wizard_step == 1:
        st.header("Step 1: Patient Selection")
        
        with st.expander("➕ Register New Patient (Quick Add)"):
            st.markdown("Fill this out if the patient is not in the system yet.")
            q_name = st.text_input("Full Name", key="quick_name")
            q_dob = st.text_input("Date of Birth (YYYY-MM-DD or DD/MM/YYYY)", placeholder="Optional", key="quick_dob")
            
            default_age = 45
            if q_dob:
                import datetime
                dob_date = None
                try:
                    import pandas as pd
                    dob_date = pd.to_datetime(q_dob).to_pydatetime()
                except Exception:
                    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]
                    for fmt in formats:
                        try:
                            dob_date = datetime.datetime.strptime(q_dob, fmt)
                            break
                        except ValueError:
                            continue
                
                if dob_date:
                    today = datetime.datetime.today()
                    calculated_age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                    if calculated_age > 0:
                        default_age = calculated_age
                        
            q_age = st.number_input("Age (Calculated/Manual)", 1, 150, default_age, key="quick_age")
            q_gender = st.selectbox("Gender", ["Male", "Female"], key="quick_gender")
            q_contact = st.text_input("Contact Number", placeholder="Optional", key="quick_contact")
            q_email = st.text_input("Email", placeholder="Optional", key="quick_email")
            q_address = st.text_area("Address", placeholder="Optional", key="quick_address")
                
            if st.button("Save & Select Patient"):
                if q_name:
                    import datetime
                    mrn = f"MRN-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                    normalized_dob = q_dob if q_dob else "N/A"
                    db.add_patient(
                        name=q_name, 
                        age=q_age, 
                        gender=q_gender, 
                        mrn=mrn,
                        dob=normalized_dob,
                        contact=q_contact if q_contact else "N/A",
                        email=q_email if q_email else "N/A",
                        address=q_address if q_address else "N/A"
                    )
                    st.success(f"Patient {q_name} registered successfully!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Name is required.")
                    
        st.markdown("---")
        
        patients = db.get_all_patients()
        if not patients:
            st.warning("No patients registered yet. Please register one above.")
        else:
            # Reverse list so newest patients appear first
            patients = list(reversed(patients))
            patient_names = [f"{p['name']} (ID: {p['patient_id']})" for p in patients]
            selected = st.selectbox("Select Existing Patient for Analysis:", patient_names)
            st.session_state.selected_patient_id = int(selected.split("(ID: ")[-1].rstrip(")"))
            st.success("Patient selected.")
            
            if st.button("Next ➡️", type="primary"):
                st.session_state.wizard_step = 2
                st.rerun()

    elif st.session_state.wizard_step == 2:
        st.header("Step 2: Document Scanner (Optional)")
        st.info("Upload or scan a document to autofill clinical parameters, or skip to enter manually.")
        
        uploaded_doc = st.file_uploader("Upload report image or text file", type=["png", "jpg", "jpeg", "txt"])
        
        if uploaded_doc is not None:
            if st.session_state.get("last_uploaded_doc_name") != uploaded_doc.name:
                try:
                    if uploaded_doc.type == "text/plain":
                        extracted = uploaded_doc.read().decode("utf-8")
                    else:
                        extracted = image_to_text(uploaded_doc.getvalue())
                    st.session_state.scan_text_input = extracted
                    st.session_state.last_uploaded_doc_name = uploaded_doc.name
                except Exception as e:
                    if str(e) == "no_ocr_engine":
                        st.error("OCR engine is not available. Please enter manually.")
                    else:
                        st.error(f"Error reading document: {e}")

        scan_text = st.text_area("Recognized document text", height=150, key="scan_text_input")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Extract & Autofill", type="secondary"):
                extracted_values = parse_medical_report_text(scan_text)
                if len(extracted_values) > 0:
                    apply_scan_values(extracted_values)
                    st.success("Autofill complete!")
                else:
                    st.warning("No data found.")
        
        with col2:
            if st.button("Next ➡️", type="primary"):
                st.session_state.wizard_step = 3
                st.rerun()
        if st.button("⬅️ Back"):
            st.session_state.wizard_step = 1
            st.rerun()

    elif st.session_state.wizard_step == 3:
        st.header("Step 3: Patient Vitals & Demographics")
        patient = db.get_patient(st.session_state.selected_patient_id)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Vitals")
            st.session_state.scan_sbp = st.number_input("Systolic BP (mmHg)", 50, 220, int(st.session_state.get("scan_sbp", 120)))
            st.session_state.scan_dbp = st.number_input("Diastolic BP (mmHg)", 30, 130, int(st.session_state.get("scan_dbp", 80)))
            st.session_state.scan_hr = st.number_input("Heart Rate (bpm)", 40, 180, int(st.session_state.get("scan_hr", 75)))
            st.session_state.scan_rr = st.number_input("Respiratory Rate (pm)", 8, 40, int(st.session_state.get("scan_rr", 16)))
            st.session_state.scan_spo2 = st.slider("Oxygen Saturation (SpO2 %)", 70, 100, int(st.session_state.get("scan_spo2", 98)))
            st.session_state.scan_temp = st.number_input("Body Temperature (°C)", 35.0, 42.0, float(st.session_state.get("scan_temp", 36.8)))
        with col2:
            st.subheader("Demographics")
            st.session_state.scan_age = st.number_input("Age (Years)", 1, 110, int(patient['age']))
            st.session_state.scan_bmi = st.number_input("BMI", 10.0, 50.0, float(st.session_state.get("scan_bmi", 24.5)))
            st.session_state.scan_charlson = st.slider("Charlson Comorbidity Index", 0, 12, int(st.session_state.get("scan_charlson", 1)))
            st.session_state.scan_gender = patient['gender']
        
        c1, c2 = st.columns(2)
        if c1.button("⬅️ Back"): st.session_state.wizard_step = 2; st.rerun()
        if c2.button("Next ➡️", type="primary"): st.session_state.wizard_step = 4; st.rerun()

    elif st.session_state.wizard_step == 4:
        st.header("Step 4: Laboratory Biomarkers")
        
        st.subheader("Core Lab Biomarkers (Required)")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.scan_wbc = st.number_input("WBC", 1.0, 30.0, float(st.session_state.get("scan_wbc", 7.5)))
            st.session_state.scan_scr = st.number_input("Serum Creatinine", 0.2, 10.0, float(st.session_state.get("scan_scr", 0.9)))
        with col2:
            st.session_state.scan_crp = st.number_input("CRP", 0.0, 200.0, float(st.session_state.get("scan_crp", 4.2)))
            st.session_state.scan_trop = st.number_input("Troponin-I", 0.0, 50.0, float(st.session_state.get("scan_trop", 0.02)))
        st.markdown("---")
        st.subheader("Optional Comprehensive Labs")
        
        # We exclude core tests that are already rendered above
        core_tests = {"wbc", "creatinine", "crp", "trop"}
        
        tabs = st.tabs(list(CLINICAL_REFERENCE_RANGES.keys()))
        for idx, category in enumerate(CLINICAL_REFERENCE_RANGES.keys()):
            with tabs[idx]:
                tests = CLINICAL_REFERENCE_RANGES[category]
                cols = st.columns(3)
                col_idx = 0
                for test_key, test_data in tests.items():
                    if test_key in core_tests:
                        continue
                        
                    test_name = test_data.get('name', test_key)
                    with cols[col_idx % 3]:
                        if 'normal' in test_data and isinstance(test_data['normal'], str):
                            current_val = st.session_state.get(f"scan_{test_key}", "Negative")
                            st.session_state[f"scan_{test_key}"] = st.selectbox(
                                test_name, 
                                ["Negative", "Positive", "Equivocal"], 
                                index=["Negative", "Positive", "Equivocal"].index(current_val) if current_val in ["Negative", "Positive", "Equivocal"] else 0,
                                key=f"ui_{test_key}"
                            )
                        else:
                            default_val = 0.0
                            if 'normal_range' in test_data:
                                default_val = round(float(sum(test_data['normal_range']) / 2), 2)
                            elif 'male_range' in test_data:
                                default_val = round(float(sum(test_data['male_range']) / 2), 2)
                                
                            current_val = st.session_state.get(f"scan_{test_key}")
                            if current_val is None or (current_val == 0.0 and f"ui_{test_key}" not in st.session_state):
                                current_val = default_val
                                
                            st.session_state[f"scan_{test_key}"] = st.number_input(
                                test_name, 
                                value=float(current_val),
                                key=f"ui_{test_key}"
                            )
                    col_idx += 1
                
        c1, c2 = st.columns(2)
        if c1.button("⬅️ Back"): st.session_state.wizard_step = 3; st.rerun()
        if c2.button("Next ➡️", type="primary"): st.session_state.wizard_step = 5; st.rerun()

    elif st.session_state.wizard_step == 5:
        st.header("Step 5: Medical Vision & Texts")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Chest X-Ray")
            dicom_file = st.file_uploader("Upload DICOM Image", type=["dcm"])
            st.session_state.simulated_image = np.zeros((224, 224), dtype=np.float32)
            
            st.session_state.scan_infiltrates = st.checkbox("Infiltrates", value=st.session_state.get("scan_infiltrates", False))
            st.session_state.scan_cardiomegaly = st.checkbox("Cardiomegaly", value=st.session_state.get("scan_cardiomegaly", False))
            st.session_state.scan_pleural_effusion = st.checkbox("Pleural Effusion", value=st.session_state.get("scan_pleural_effusion", False))
        
        with col2:
            st.subheader("Clinical Notes")
            st.session_state.scan_notes = st.text_area("Notes", value=st.session_state.get("scan_notes", "Patient is stable."))
            
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        if c1.button("⬅️ Back"): st.session_state.wizard_step = 4; st.rerun()
        if c2.button("🚀 EXECUTE AI ANALYSIS", type="primary"):
            st.session_state.pop("ai_recs", None) # Clear previous recommendations
            st.session_state.wizard_step = 6
            st.rerun()

    elif st.session_state.wizard_step == 6:
        st.header("Step 6: Multimodal Fusion & Processing")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate AI Fusion Process
        time.sleep(0.5)
        status_text.text("Extracting Clinical Tabular Features...")
        progress_bar.progress(25)
        
        time.sleep(1)
        status_text.text("Analyzing Medical Vision Data...")
        progress_bar.progress(50)
        
        time.sleep(1)
        status_text.text("Executing NLP on Clinical Notes...")
        progress_bar.progress(75)
        
        time.sleep(1)
        status_text.text("Running Neural Multimodal Fusion...")
        progress_bar.progress(100)
        
        # Run real prediction
        gender_binary = 1.0 if st.session_state.get("scan_gender") == "Male" else 0.0
        tabular_payload = [
            float(st.session_state.scan_sbp), float(st.session_state.scan_dbp), float(st.session_state.scan_hr),
            float(st.session_state.scan_rr), float(st.session_state.scan_spo2), float(st.session_state.scan_temp),
            float(st.session_state.scan_wbc), float(st.session_state.scan_scr), float(st.session_state.scan_crp),
            float(st.session_state.scan_trop), float(st.session_state.scan_age), gender_binary,
            float(st.session_state.scan_bmi), float(st.session_state.scan_charlson)
        ]
        
        optional_labs = {}
        for category, tests in CLINICAL_REFERENCE_RANGES.items():
            for test_key in tests.keys():
                if test_key not in {"wbc", "creatinine", "crp", "trop"}:
                    val = st.session_state.get(f"scan_{test_key}")
                    if val != 0.0 and val != "Negative" and val is not None:
                        optional_labs[test_key] = val

        payload = {
            "patient_info": {
                "name": db.get_patient(st.session_state.selected_patient_id)['name'],
                "age": st.session_state.scan_age,
                "gender": st.session_state.scan_gender,
                "optional_labs": optional_labs,
                "infiltrates": st.session_state.scan_infiltrates,
                "cardiomegaly": st.session_state.scan_cardiomegaly,
                "pleural_effusion": st.session_state.scan_pleural_effusion,
                "consolidation": False,
                "lung_nodule": False
            },
            "tabular": tabular_payload,
            "image": st.session_state.simulated_image.tolist(),
            "text": st.session_state.scan_notes
        }
        
        res = run_prediction(payload)
        st.session_state.prediction_result = res
        
        st.success("Analysis Complete!")
        time.sleep(0.5)
        st.session_state.wizard_step = 7
        st.rerun()

    elif st.session_state.wizard_step == 7:
        st.header("Step 7: AI Prediction Result")
        res = st.session_state.prediction_result
        
        k1, k2 = st.columns(2)
        k1.metric("DETERMINED PATIENT STATUS", res.get("diagnostic_label", "Unknown"))
        k2.metric("CRITICAL EMERGENCY RISK CONFIDENCE", f"{res.get('confidence_scores', {}).get('class_1_critical', 0)*100:.2f}%")
        
        if res.get("status") == "error":
            st.error(f"🚨 Prediction Error: {res.get('message')}")
        
        st.markdown("---")
        
        if "feature_importance" in res:
            st.subheader("🧠 SHAP-Style Feature Impact")
            df_fi = pd.DataFrame(res["feature_importance"])
            df_fi["Effect"] = df_fi["Importance"].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
            
            import plotly.express as px
            fig = px.bar(
                df_fi, x="Importance", y="Feature", color="Effect",
                color_discrete_map={"Increases Risk": "#ef553b", "Decreases Risk": "#00cc96"},
                orientation='h',
                height=500
            )
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'}, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#E2E8F0'),
                margin=dict(l=120, r=20, t=20, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # --- AI RECOMMENDATIONS TRIGGER ---
        risk_confidence = res.get('confidence_scores', {}).get('class_1_critical', 0)
        if risk_confidence > 0.60:
            st.markdown("---")
            st.markdown("<h3 style='color: #ff4b4b;'>⚠️ High Risk Detected (>60%) - AI Recommendations</h3>", unsafe_allow_html=True)
            
            if "ai_recs" not in st.session_state:
                with st.spinner("Analyzing patient data for treatment recommendations..."):
                    context_str = f"Age: {st.session_state.scan_age}\n"
                    context_str += f"Vitals: BP {st.session_state.scan_sbp}/{st.session_state.scan_dbp}, HR {st.session_state.scan_hr}, SpO2 {st.session_state.scan_spo2}%, Temp {st.session_state.scan_temp}\n"
                    context_str += f"Labs: WBC {st.session_state.scan_wbc}, Creatinine {st.session_state.scan_scr}, CRP {st.session_state.scan_crp}\n"
                    context_str += f"Risk Score: {risk_confidence*100:.2f}%\n"
                    
                    lang_pref = st.session_state.get('chat_lang', 'Gujarati 🕉️')
                    if not lang_pref: lang_pref = 'Gujarati 🕉️'
                    st.session_state.ai_recs = get_ai_recommendations(context_str, language=lang_pref)
            
            st.warning(st.session_state.ai_recs)
            
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        if c1.button("💾 Save Assessment to Database", type="primary"):
            assessment_id = db.add_assessment(
                patient_id=st.session_state.selected_patient_id,
                vitals={'systolic_bp': st.session_state.scan_sbp, 'heart_rate': st.session_state.scan_hr, 'spo2': st.session_state.scan_spo2},
                labs={'wbc': st.session_state.scan_wbc, 'serum_creatinine': st.session_state.scan_scr},
                clinical_notes=st.session_state.scan_notes,
                diagnosis=res.get('diagnostic_label', 'Unknown')
            )
            db.update_assessment_predictions(
                assessment_id=assessment_id,
                risk_score=res.get('risk_score', 0.0),
                risk_category=res.get('diagnostic_label', 'Unknown'),
                severity_level=res.get('severity', 'Unknown'),
                confidence=res.get('confidence_scores', {}).get('class_1_critical', 0.0)
            )
            st.success("Saved successfully!")
            
        if c2.button("🔄 Start New Analysis"):
            st.session_state.wizard_step = 1
            st.rerun()

        with c3:
            with st.expander("📧 Email PDF Report"):
                doc_email = st.text_input("Doctor/Patient Email")
                if st.button("Send Report"):
                    if not doc_email:
                        st.error("Please enter an email address.")
                    else:
                        with st.spinner("Generating PDF and sending email..."):
                            try:
                                from utils.pdf_generator import generate_pdf_report
                                from utils.email_service import send_medical_report_email
                                from api.chat_service import get_ai_recommendations
                                
                                # Get patient name from DB
                                patient_db = db.get_patient(st.session_state.selected_patient_id)
                                patient_name = patient_db['name'] if patient_db else 'Unknown'
                                gender_label = "Male" if st.session_state.get('scan_gender') == "Male" else "Female"
                                
                                # Reconstruct full payload for PDF
                                pdf_payload = {
                                    "patient_info": {
                                        "Name": patient_name,
                                        "Age": f"{st.session_state.scan_age} Years",
                                        "Gender": gender_label,
                                        "Report Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                                    },
                                    "tabular": [
                                        st.session_state.scan_sbp, st.session_state.scan_dbp, st.session_state.scan_hr,
                                        st.session_state.scan_rr, st.session_state.scan_spo2, st.session_state.scan_temp,
                                        st.session_state.scan_wbc, st.session_state.scan_scr, st.session_state.scan_crp,
                                        0.0, st.session_state.scan_age, 0, 24.5, 1
                                    ]
                                }
                                
                                # Get English AI recommendations for PDF for all patients
                                risk_conf = res.get('confidence_scores', {}).get('class_1_critical', 0)
                                context_str = f"Name: {patient_name}, Age: {st.session_state.scan_age}, Gender: {gender_label}\n"
                                context_str += f"BP: {st.session_state.scan_sbp}/{st.session_state.scan_dbp}, HR: {st.session_state.scan_hr}, SpO2: {st.session_state.scan_spo2}%, Temp: {st.session_state.scan_temp}\n"
                                context_str += f"WBC: {st.session_state.scan_wbc}, Creatinine: {st.session_state.scan_scr}, CRP: {st.session_state.scan_crp}\n"
                                context_str += f"Risk Score: {risk_conf*100:.1f}%, Status: {res.get('diagnostic_label', 'Unknown')}"
                                ai_english_recs = get_ai_recommendations(context_str, language="English")
                                
                                pdf_bytes = generate_pdf_report(pdf_payload, res, ai_gujarati_recs=ai_english_recs)
                                success, err_msg = send_medical_report_email(doc_email, patient_name, pdf_bytes)
                                if success:
                                    st.success(f"Email sent to {doc_email}!")
                                else:
                                    st.error(f"Failed to send email: {err_msg}")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

elif mode == "👥 Patient Management":
    st.title("Patient Management")

    # ── TABS: Manual | Bulk CSV Import ──────────────────────────────────────
    tab_manual, tab_csv = st.tabs(["✍️ Manual Registration", "📂 Bulk Import from CSV"])

    # ── TAB 1: Manual Registration (unchanged) ───────────────────────────────
    with tab_manual:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Register New Patient")
            p_name = st.text_input("Name")
            p_dob = st.text_input("Date of Birth (YYYY-MM-DD or DD/MM/YYYY)", placeholder="Optional")

            default_age = 45
            if p_dob:
                import datetime
                dob_date = None
                try:
                    import pandas as pd
                    dob_date = pd.to_datetime(p_dob).to_pydatetime()
                except Exception:
                    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]
                    for fmt in formats:
                        try:
                            dob_date = datetime.datetime.strptime(p_dob, fmt)
                            break
                        except ValueError:
                            continue
                if dob_date:
                    today = datetime.datetime.today()
                    calculated_age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                    if calculated_age > 0:
                        default_age = calculated_age

            p_age    = st.number_input("Age (Calculated/Manual)", 1, 150, default_age)
            p_gender = st.selectbox("Gender", ["Male", "Female"])
            p_contact = st.text_input("Contact Number", placeholder="Optional")
            p_email   = st.text_input("Email", placeholder="Optional")
            p_address = st.text_area("Address", placeholder="Optional")

            if st.button("Register Patient", key="btn_register_manual"):
                if p_name:
                    patients_existing = db.get_all_patients()
                    is_duplicate = False
                    normalized_dob = p_dob if p_dob else "N/A"
                    for pat in patients_existing:
                        if str(pat.get('name', '')).strip().lower() == p_name.strip().lower() and \
                           str(pat.get('date_of_birth', '')) == normalized_dob:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        st.error(f"⚠️ Patient '{p_name}' with DOB '{normalized_dob}' is already registered!")
                    else:
                        import datetime
                        mrn = f"MRN-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                        db.add_patient(
                            name=p_name, age=p_age, gender=p_gender, mrn=mrn,
                            dob=normalized_dob,
                            contact=p_contact if p_contact else "N/A",
                            email=p_email   if p_email   else "N/A",
                            address=p_address if p_address else "N/A"
                        )
                        st.success("✅ Patient Registered!")
                        st.rerun()
                else:
                    st.error("Name is required.")

        with c2:
            st.subheader("Registered Patients")
            patients = db.get_all_patients()
            if patients:
                # Show newest first
                patients_sorted = sorted(patients, key=lambda x: x.get('created_at',''), reverse=True)
                st.dataframe(
                    pd.DataFrame(patients_sorted)[['patient_id','name','age','gender','date_of_birth','contact_number','email','created_at']],
                    use_container_width=True, hide_index=True
                )
                st.markdown("---")
                st.markdown("### 🗑️ Delete Patients")
                del_options_m = [f"{p['name']}  (ID: {p['patient_id']})" for p in patients_sorted]
                # Select All toggle
                select_all_m = st.toggle("☑️ Select All Patients", key="select_all_manual")
                default_sel_m = del_options_m if select_all_m else []
                del_multi = st.multiselect(
                    "Select one or more patients to delete:",
                    options=del_options_m,
                    default=default_sel_m,
                    key="del_multi_manual"
                )
                if del_multi:
                    st.warning(f"⚠️ **{len(del_multi)} patient(s)** selected for deletion. This will remove ALL their data (assessments, predictions, records).")
                    confirm_del = st.checkbox(f"✅ Yes, permanently delete {len(del_multi)} patient(s)", key="confirm_del_manual")
                    if st.button(f"🗑️ Delete {len(del_multi)} Patient(s)", type="secondary", key="btn_del_manual"):
                        if confirm_del:
                            for opt in del_multi:
                                pid = int(opt.split("(ID: ")[-1].rstrip(")"))
                                db.delete_patient(pid)
                            st.success(f"✅ {len(del_multi)} patient(s) deleted successfully!")
                            time.sleep(0.6)
                            st.rerun()
                        else:
                            st.warning("⚠️ Please check the confirmation checkbox first.")
            else:
                st.info("No patients registered yet.")

    # ── TAB 2: Bulk CSV Import ───────────────────────────────────────────────
    with tab_csv:
        st.markdown("""
<div style="background:rgba(0,240,255,0.05);border:1px solid rgba(0,240,255,0.2);border-radius:12px;padding:18px;margin-bottom:18px;">
<div style="color:#00F0FF;font-size:16px;font-weight:800;margin-bottom:8px;">📂 Bulk Patient Import via CSV / Excel</div>
<div style="color:#94A3B8;font-size:13px;">Upload a <b>CSV</b> or <b>Excel (.xlsx)</b> file — all rows will be auto-registered as patients.<br>
Required column: <code>name</code> &nbsp;|&nbsp; Optional: <code>age, gender, date_of_birth, contact_number, email, address</code></div>
</div>""", unsafe_allow_html=True)

        # Download template
        template_csv = pd.DataFrame({
            "name":          ["Ravi Shah",    "Priya Patel"],
            "age":           [34,             28],
            "gender":        ["Male",         "Female"],
            "date_of_birth": ["1990-05-12",   "1996-08-24"],
            "contact_number":["9876543210",   "9123456780"],
            "email":         ["ravi@mail.com","priya@mail.com"],
            "address":       ["Surat, Gujarat","Ahmedabad, Gujarat"],
        })
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=template_csv.to_csv(index=False).encode("utf-8"),
            file_name="patient_import_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader(
            "Upload Patient File (CSV or Excel)",
            type=["csv", "xlsx"],
            key="bulk_patient_upload"
        )

        if uploaded is not None:
            try:
                if uploaded.name.endswith(".xlsx"):
                    import_df = pd.read_excel(uploaded)
                else:
                    import_df = pd.read_csv(uploaded)

                import_df.columns = [c.strip().lower().replace(" ", "_") for c in import_df.columns]

                if "name" not in import_df.columns:
                    st.error("❌ File must have a 'name' column. Please fix and re-upload.")
                else:
                    st.markdown(f"**Preview** — {len(import_df)} patients found:")
                    st.dataframe(import_df.head(10), use_container_width=True)

                    st.markdown("---")
                    col_btn1, col_btn2 = st.columns([1, 3])
                    with col_btn1:
                        run_import = st.button("🚀 Register All Patients", type="primary", key="btn_bulk_import")

                    if run_import:
                        existing_patients = db.get_all_patients()
                        existing_keys = set(
                            (str(p.get("name","")).strip().lower(), str(p.get("date_of_birth","")))
                            for p in existing_patients
                        )

                        added     = 0
                        skipped   = 0
                        errors    = 0
                        log_lines = []

                        prog_bar  = st.progress(0, text="Starting import…")
                        status_ph = st.empty()
                        import datetime

                        for i, row in import_df.iterrows():
                            pct  = int(((i + 1) / len(import_df)) * 100)
                            name = str(row.get("name", "")).strip()

                            prog_bar.progress(pct, text=f"Processing {i+1}/{len(import_df)} — {name}")

                            if not name:
                                skipped += 1
                                log_lines.append(f"⚠️ Row {i+1}: Empty name — skipped")
                                continue

                            # Age
                            try:
                                age = int(float(row["age"])) if "age" in row and pd.notna(row["age"]) else 30
                                age = max(1, min(150, age))
                            except Exception:
                                age = 30

                            # Gender
                            raw_gender = str(row.get("gender", "Male")).strip().capitalize()
                            gender = raw_gender if raw_gender in ["Male", "Female"] else "Male"

                            # DOB
                            dob_val = str(row.get("date_of_birth", "")).strip()
                            if not dob_val or dob_val.lower() in ["nan", "none", ""]:
                                dob_val = "N/A"

                            # Duplicate check
                            key = (name.lower(), dob_val)
                            if key in existing_keys:
                                skipped += 1
                                log_lines.append(f"⏭️ Row {i+1}: '{name}' already exists — skipped")
                                continue

                            # Contact / Email / Address
                            contact = str(row.get("contact_number", "N/A")).strip()
                            email_v = str(row.get("email", "N/A")).strip()
                            address = str(row.get("address",  "N/A")).strip()
                            for f in [contact, email_v, address]:
                                if f.lower() in ["nan", "none", ""]:
                                    f = "N/A"
                            contact = contact if contact not in ["nan","none",""] else "N/A"
                            email_v = email_v if email_v not in ["nan","none",""] else "N/A"
                            address = address if address not in ["nan","none",""] else "N/A"

                            try:
                                mrn = f"MRN-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{i}"
                                db.add_patient(
                                    name=name, age=age, gender=gender, mrn=mrn,
                                    dob=dob_val, contact=contact,
                                    email=email_v, address=address
                                )
                                existing_keys.add(key)
                                added += 1
                                log_lines.append(f"✅ Row {i+1}: '{name}' registered (Age:{age}, {gender})")
                            except Exception as ex:
                                errors += 1
                                log_lines.append(f"❌ Row {i+1}: '{name}' failed — {ex}")

                        prog_bar.progress(100, text="Import complete!")

                        # Summary cards
                        st.markdown(f"""
<div style="display:flex;gap:16px;margin:16px 0;">
  <div style="flex:1;background:rgba(0,204,150,0.1);border:1px solid #00cc96;border-radius:10px;padding:16px;text-align:center;">
    <div style="color:#00cc96;font-size:28px;font-weight:900;">{added}</div>
    <div style="color:#94A3B8;font-size:12px;font-weight:700;">REGISTERED</div>
  </div>
  <div style="flex:1;background:rgba(251,191,36,0.1);border:1px solid #f59e0b;border-radius:10px;padding:16px;text-align:center;">
    <div style="color:#f59e0b;font-size:28px;font-weight:900;">{skipped}</div>
    <div style="color:#94A3B8;font-size:12px;font-weight:700;">SKIPPED (DUPLICATE)</div>
  </div>
  <div style="flex:1;background:rgba(239,85,59,0.1);border:1px solid #ef553b;border-radius:10px;padding:16px;text-align:center;">
    <div style="color:#ef553b;font-size:28px;font-weight:900;">{errors}</div>
    <div style="color:#94A3B8;font-size:12px;font-weight:700;">ERRORS</div>
  </div>
</div>""", unsafe_allow_html=True)

                        with st.expander("📋 Import Log", expanded=(errors > 0)):
                            for line in log_lines:
                                st.markdown(line)

                        if added > 0:
                            st.balloons()
                            st.success(f"🎉 {added} patient(s) successfully registered!")
                            time.sleep(1)
                            st.rerun()

            except Exception as parse_err:
                st.error(f"❌ Could not read file: {parse_err}")

        # Always show registered patients table below
        st.markdown("---")
        st.subheader("📋 All Registered Patients")
        all_pts = db.get_all_patients()
        if all_pts:
            all_pts_sorted = sorted(all_pts, key=lambda x: x.get('created_at',''), reverse=True)
            st.dataframe(
                pd.DataFrame(all_pts_sorted)[['patient_id','name','age','gender','date_of_birth','contact_number','email','created_at']],
                use_container_width=True, hide_index=True
            )
            st.markdown("---")
            st.markdown("### 🗑️ Delete Patients")
            del_opts2 = [f"{p['name']}  (ID: {p['patient_id']})" for p in all_pts_sorted]
            # Select All toggle
            select_all_2 = st.toggle("☑️ Select All Patients", key="select_all_csv")
            default_sel_2 = del_opts2 if select_all_2 else []
            del_multi2 = st.multiselect(
                "Select one or more patients to delete:",
                options=del_opts2,
                default=default_sel_2,
                key="del_multi_csv"
            )
            if del_multi2:
                st.warning(f"⚠️ **{len(del_multi2)} patient(s)** selected for deletion. This will remove ALL their data.")
                confirm2 = st.checkbox(f"✅ Yes, permanently delete {len(del_multi2)} patient(s)", key="confirm_del_csv")
                if st.button(f"🗑️ Delete {len(del_multi2)} Patient(s)", type="secondary", key="btn_del_csv"):
                    if confirm2:
                        for opt2 in del_multi2:
                            pid2 = int(opt2.split("(ID: ")[-1].rstrip(")"))
                            db.delete_patient(pid2)
                        st.success(f"✅ {len(del_multi2)} patient(s) deleted successfully!")
                        time.sleep(0.6)
                        st.rerun()
                    else:
                        st.warning("⚠️ Please check the confirmation checkbox first.")
        else:
            st.info("No patients registered yet.")

elif mode == "📜 Analysis History":
    st.title("Analysis History")
    assessments = []
    for p in db.get_all_patients():
        assessments.extend(db.get_patient_assessments(p['patient_id']))
    if assessments:
        st.dataframe(pd.DataFrame(assessments), use_container_width=True)
    else:
        st.info("No assessments found.")
        
elif mode == "📚 Clinical Reference":
    st.title("Clinical Lab Reference")
    ref_category = st.selectbox("View Reference:", list(CLINICAL_REFERENCE_RANGES.keys()))
    with st.expander(f"{ref_category} Normal Ranges", expanded=True):
        for test_name, test_data in CLINICAL_REFERENCE_RANGES[ref_category].items():
            name = test_data.get('name', test_name.replace('_', ' ').title())
            st.caption(f"**{name}**")
            if 'normal_range' in test_data:
                st.caption(f"Normal: {test_data['normal_range'][0]} - {test_data['normal_range'][1]} {test_data.get('unit', '')}")
            elif 'male_range' in test_data:
                st.caption(f"M: {test_data['male_range'][0]}-{test_data['male_range'][1]} | F: {test_data['female_range'][0]}-{test_data['female_range'][1]} {test_data.get('unit', '')}")
            st.divider()

elif mode == "📂 Batch Processing (CSV)":
    st.title("Batch Processing (CSV)")
    st.write("Upload a CSV file containing patient data to run bulk predictions. The CSV should contain columns matching the patient info and lab results.")
    
    # Download template button
    template_data = {
        'name': ['John Doe'], 'age': [45], 'gender': ['Male'],
        'systolic_bp': [120.0], 'diastolic_bp': [80.0], 'heart_rate': [75.0], 'respiratory_rate': [18.0],
        'spo2': [98.0], 'body_temperature': [98.6], 'wbc': [7.0], 'serum_creatinine': [0.9],
        'hemoglobin': [14.5], 'hematocrit': [45.0], 'platelets': [250.0], 'glucose': [100.0],
        'bun': [15.0], 'sodium': [140.0], 'clinical_notes': ['Patient is stable.']
    }
    template_df = pd.DataFrame(template_data)
    st.download_button(
        label="📥 Download CSV Template",
        data=template_df.to_csv(index=False).encode('utf-8'),
        file_name='batch_processing_template.csv',
        mime='text/csv',
    )
    
    uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} records. Preview:")
        st.dataframe(df.head())
        
        if st.button("▶️ Run Batch Prediction", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            for i, row in df.iterrows():
                status_text.text(f"Processing record {i+1} of {len(df)}...")
                
                # Handle temperature conversion if F is provided
                temp_val = float(row.get('body_temperature', 37.0))
                if temp_val > 45.0:
                    temp_val = (temp_val - 32) * 5.0/9.0
                    
                gender_str = str(row.get('gender', 'Male')).strip().lower()
                gender_bin = 1.0 if gender_str in ['male', 'm'] else 0.0

                # Format tabular features strictly matching config order
                tabular = [
                    float(row.get('systolic_bp', 120.0)),
                    float(row.get('diastolic_bp', 80.0)),
                    float(row.get('heart_rate', 75.0)),
                    float(row.get('respiratory_rate', 16.0)),
                    float(row.get('spo2', 98.0)),
                    temp_val,
                    float(row.get('wbc', 7.5)),
                    float(row.get('serum_creatinine', 0.9)),
                    float(row.get('crp', 4.0)),
                    float(row.get('troponin', 0.02)),
                    float(row.get('age', 45.0)),
                    gender_bin,
                    float(row.get('bmi', 24.5)),
                    float(row.get('charlson_index', 1.0))
                ]
                
                # Extract optional labs for additional risk checking
                optional_labs = {}
                for lab_key in ['hemoglobin', 'hematocrit', 'platelets', 'glucose', 'bun', 'sodium']:
                    if lab_key in row and not pd.isna(row[lab_key]):
                        optional_labs[lab_key] = float(row[lab_key])
                        
                payload = {
                    "patient_info": {
                        "name": str(row.get('name', f"Patient {i+1}")),
                        "age": int(row.get('age', 40)),
                        "gender": str(row.get('gender', "Unknown")),
                        "optional_labs": optional_labs
                    },
                    "tabular": tabular,
                    "image": [[0.0]*224]*224, # Dummy image
                    "text": str(row.get('clinical_notes', "No notes provided."))
                }
                
                try:
                    res = run_prediction(payload)
                    if res.get("status") == "error":
                        row['risk_score'] = None
                        row['risk_category'] = 'Error'
                        row['severity_level'] = res.get("message", "Unknown error")
                    else:
                        row['risk_score'] = round(res.get('confidence_scores', {}).get('class_1_critical', 0), 4)
                        row['risk_category'] = 'High Risk' if res.get('predicted_class') == 1 else 'Low Risk'
                        row['severity_level'] = res.get('diagnostic_label', 'Unknown')
                except Exception as e:
                    row['risk_score'] = None
                    row['risk_category'] = 'Error'
                    row['severity_level'] = str(e)
                    
                results.append(row)
                progress_bar.progress((i + 1) / len(df))
                
            status_text.text("Batch processing complete!")
            results_df = pd.DataFrame(results)
            st.success(f"Successfully processed {len(results_df)} records.")
            st.dataframe(results_df)
            
            st.download_button(
                label="📥 Download Results",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name='batch_predictions_results.csv',
                mime='text/csv',
            )

elif mode == "🗄️ Database Inspector":
    st.title("Database Inspector")
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "👥 Patients", "📋 Audit Logs"])
    
    with tab1:
        st.subheader("System Overview")
        col1, col2 = st.columns(2)
        try:
            with col1:
                st.metric("Total Patients", db.get_patient_count())
            with col2:
                st.metric("Total Assessments", db.get_assessment_count())
                
            st.subheader("Risk Statistics")
            risk_stats = db.get_risk_statistics()
            if risk_stats:
                stats_df = pd.DataFrame(list(risk_stats.items()), columns=['Risk Category', 'Count'])
                
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 Interactive Bar", "🥧 Pie Chart", "🍩 Donut Chart"])
                
                with chart_tab1:
                    fig_bar = px.bar(
                        stats_df, 
                        x='Risk Category', 
                        y='Count', 
                        color='Risk Category',
                        title="Risk Category Distribution",
                        color_discrete_map={
                            "Low Risk": "#28a745",
                            "STABLE / ROUTINE WARD MONITORING": "#28a745",
                            "High Risk": "#dc3545",
                            "High": "#dc3545",
                            "Low": "#17a2b8",
                            "CRITICAL / IMMEDIATE ICU ATTENTION": "#8b0000"
                        }
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                with chart_tab2:
                    fig_pie = px.pie(
                        stats_df, 
                        names='Risk Category', 
                        values='Count',
                        title="Risk Category Proportions",
                        color='Risk Category',
                        color_discrete_map={
                            "Low Risk": "#28a745",
                            "STABLE / ROUTINE WARD MONITORING": "#28a745",
                            "High Risk": "#dc3545",
                            "High": "#dc3545",
                            "Low": "#17a2b8",
                            "CRITICAL / IMMEDIATE ICU ATTENTION": "#8b0000"
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with chart_tab3:
                    fig_donut = px.pie(
                        stats_df, 
                        names='Risk Category', 
                        values='Count',
                        hole=0.4,
                        title="Risk Category Proportions (Donut)",
                        color='Risk Category',
                        color_discrete_map={
                            "Low Risk": "#28a745",
                            "STABLE / ROUTINE WARD MONITORING": "#28a745",
                            "High Risk": "#dc3545",
                            "High": "#dc3545",
                            "Low": "#17a2b8",
                            "CRITICAL / IMMEDIATE ICU ATTENTION": "#8b0000"
                        }
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("No risk statistics available yet.")
        except Exception as e:
            st.error(f"Could not load overview statistics: {e}")
            
    with tab2:
        st.subheader("Patient Registry")
        try:
            patients = db.get_all_patients()
            if patients:
                patients_df = pd.DataFrame(patients)
                # Format datetime if exists
                if 'created_at' in patients_df.columns:
                    patients_df['created_at'] = pd.to_datetime(patients_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(patients_df, use_container_width=True)
            else:
                st.info("No patients found in database.")
        except Exception as e:
            st.error(f"Error loading patients: {e}")
            
    with tab3:
        st.subheader("System Audit Logs")
        try:
            logs = db.get_audit_log(limit=50)
            if logs:
                logs_df = pd.DataFrame(logs)
                if 'timestamp' in logs_df.columns:
                    logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(logs_df, use_container_width=True)
            else:
                st.info("No audit logs available.")
        except Exception as e:
            st.error(f"Error loading audit logs: {e}")


