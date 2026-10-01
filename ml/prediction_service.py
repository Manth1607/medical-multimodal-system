import os
import json
import numpy as np
import torch
from datetime import datetime, timezone
from pathlib import Path
import logging

from core.config import Config
from ml.models import ClinicalMultimodalFusionClassifier

# Setup simple file logger for inference / OOD events
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger('inference_logger')
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(log_dir / 'inference.log')
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(fh)

# Global cached model
_model = None

def _load_model():
    """Load the model once and cache it."""
    global _model
    if _model is None:
        model = ClinicalMultimodalFusionClassifier()
        checkpoint_path = Config.MODEL_PATH
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.eval()
        _model = model
    return _model

def run_prediction(payload: dict) -> dict:
    """Run the full inference pipeline on a given payload, matching the original API response schema."""
    try:
        model = _load_model()

        # Log basic metadata (no PHI)
        logger.info(json.dumps({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation': 'run_prediction',
            'status': 'started'
        }))

        # ---------------------------------------------------------------------
        # 1️⃣ Verify that all expected clinical fields are present
        # ---------------------------------------------------------------------
        required_len = len(Config.TAB_MEAN)
        if len(payload["tabular"]) != required_len:
            raise ValueError(f"Expected {required_len} tabular features, got {len(payload['tabular'])}")
        
        # Extract raw clinical vitals for safety checks before scaling
        spO2_raw = payload["tabular"][4]
        temp_raw = payload["tabular"][5]
        wbc_raw = payload["tabular"][6]
        troponin_raw = payload["tabular"][9] if len(payload["tabular"]) > 9 else None
        charlson_raw = payload["tabular"][13] if len(payload["tabular"]) > 13 else None
        
        # Extract chest X-ray boolean flags from patient_info (default False)
        patient_info = payload.get('patient_info', {})
        cardiomegaly = bool(patient_info.get('cardiomegaly', False))
        pleural_effusion = bool(patient_info.get('pleural_effusion', False))
        consolidation = bool(patient_info.get('consolidation', False))
        lung_nodule = bool(patient_info.get('lung_nodule', False))
        infiltrates = bool(patient_info.get('infiltrates', False))

        abnormal_xray_count = sum([cardiomegaly, pleural_effusion, consolidation, lung_nodule, infiltrates])
        optional_labs = patient_info.get('optional_labs', {}) or {}


        # ---------------------------------------------------------------------
        # 2️⃣ Preprocess tabular data (scaling)
        # ---------------------------------------------------------------------
        tab_np = np.asarray(payload["tabular"], dtype=np.float32)
        if hasattr(Config, 'CLINICAL_TAB_MEAN') and hasattr(Config, 'CLINICAL_TAB_STD'):
            tab_mean = np.asarray(Config.CLINICAL_TAB_MEAN, dtype=np.float32)
            tab_std = np.asarray(Config.CLINICAL_TAB_STD, dtype=np.float32)
        else:
            tab_mean = np.asarray(Config.TAB_MEAN, dtype=np.float32)
            tab_std = np.asarray(Config.TAB_STD, dtype=np.float32)
        tab_scaled = (tab_np - tab_mean) / (tab_std + 1e-9)

        # ---------------------------------------------------------------------
        # 3️⃣ OOD detection on tabular data
        # ---------------------------------------------------------------------
        z_threshold = getattr(Config, 'OOD_Z_THRESHOLD', 3.0)
        z_scores = np.abs(tab_scaled)
        ood_reasons = []
        if np.any(z_scores > z_threshold):
            ood_reasons.append('tabular_z_score_exceeded')
        if (z_scores > z_threshold).sum() / float(len(z_scores)) > 0.25:
            ood_reasons.append('tabular_many_features_out_of_range')

        # ---------------------------------------------------------------------
        # 4️⃣ Image handling – plus abnormal X-ray heuristic
        # ---------------------------------------------------------------------
        img_arr = np.array(payload["image"], dtype=np.float32)
        if img_arr.ndim == 2:
            img_arr = img_arr.reshape((1, *img_arr.shape))
        elif img_arr.ndim == 3 and img_arr.shape[0] != 1:
            img_arr = img_arr[0:1]
        t_img = torch.tensor([img_arr], dtype=torch.float32)

        # Image size validation
        expected_h, expected_w = Config.IMAGE_SIZE[1], Config.IMAGE_SIZE[2]
        img_h, img_w = img_arr.shape[-2], img_arr.shape[-1]
        if (img_h, img_w) != (expected_h, expected_w):
            ood_reasons.append('image_size_mismatch')
        
        # Heuristic: mean intensity > 0.1 indicates abnormal patch
        abnormal_xray_image = torch.mean(t_img).item() > 0.1
        if abnormal_xray_image:
            ood_reasons.append('heuristic_xray_intensity_abnormal')

        # Optional labs clinical rules
        optional_risk_reasons = []
        def optional_value(name, default=None):
            try:
                value = optional_labs.get(name, default)
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return default

        optional_checks = [
            ('hemoglobin_critical', optional_value('hemoglobin'), lambda v: v < 8.0 or v > 18.0),
            ('wbc_optional_critical', optional_value('wbc_adv'), lambda v: v < 3.0 or v > 15.0),
            ('platelets_critical', optional_value('platelets'), lambda v: v < 50.0 or v > 450.0),
            ('esr_high', optional_value('esr'), lambda v: v > 60.0),
            ('pt_inr_high', optional_value('pt_inr'), lambda v: v > 2.0),
            ('aptt_high', optional_value('aptt'), lambda v: v > 45.0),
            ('glucose_critical', optional_value('glucose'), lambda v: v < 60.0 or v > 300.0),
            ('hba1c_high', optional_value('hba1c'), lambda v: v > 9.0),
            ('egfr_low', optional_value('egfr'), lambda v: v < 30.0),
            ('microalbumin_high', optional_value('microalbumin'), lambda v: v >= 300.0),
            ('alt_critical', optional_value('alt'), lambda v: v > 200.0),
            ('ast_critical', optional_value('ast'), lambda v: v > 200.0),
            ('albumin_low', optional_value('albumin'), lambda v: v < 2.5),
            ('bilirubin_high', optional_value('bilirubin'), lambda v: v > 3.0),
            ('calcium_critical', optional_value('calcium'), lambda v: v < 7.0 or v > 11.5),
        ]
        for reason, value, is_abnormal in optional_checks:
            if value is not None and is_abnormal(value):
                optional_risk_reasons.append(reason)
        optional_risk_score = min(0.25, 0.05 * len(optional_risk_reasons))
        optional_critical = len(optional_risk_reasons) >= 2

        # ---------------------------------------------------------------------
        # 5️⃣ Text tokenisation
        # ---------------------------------------------------------------------
        tokenizer = None
        tokenizer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer')
        if os.path.exists(tokenizer_dir):
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            except Exception:
                tokenizer = None

        if isinstance(payload["text"], str):
            if tokenizer is not None:
                enc = tokenizer(payload["text"], padding='max_length', truncation=True, max_length=Config.MAX_TEXT_LEN)
                tokens = [int(x) % Config.VOCAB_SIZE for x in enc['input_ids']]
                if len(payload["text"].split()) > Config.MAX_TEXT_LEN:
                    ood_reasons.append('text_length_exceeded')
            else:
                words = payload["text"].split()
                if len(words) > Config.MAX_TEXT_LEN:
                    ood_reasons.append('text_length_exceeded')
                tokens = [abs(hash(w)) % Config.VOCAB_SIZE for w in words]
                if len(tokens) < Config.MAX_TEXT_LEN:
                    tokens = tokens + [0] * (Config.MAX_TEXT_LEN - len(tokens))
                else:
                    tokens = tokens[:Config.MAX_TEXT_LEN]
        else:
            tokens = [int(x) for x in payload["text"]]
            if len(tokens) < Config.MAX_TEXT_LEN:
                tokens = tokens + [0] * (Config.MAX_TEXT_LEN - len(tokens))
            else:
                tokens = tokens[:Config.MAX_TEXT_LEN]
        t_txt = torch.tensor([tokens], dtype=torch.long)
        t_tab = torch.tensor([tab_scaled], dtype=torch.float32)

        # Log preprocessing snapshot (no PHI)
        logger.info(json.dumps({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operation': 'run_prediction',
            'status': 'preprocessed',
            'abnormal_xray_image': abnormal_xray_image,
            'abnormal_xray_flags_count': abnormal_xray_count,
            'optional_lab_risk_reasons': optional_risk_reasons,
            'ood_reasons': ood_reasons
        }))

        # ---------------------------------------------------------------------
        # 6️⃣ Model forward pass (with gradient for tabular explainability)
        # ---------------------------------------------------------------------
        # Run heavy encoders without gradients to save memory and prevent hanging
        with torch.no_grad():
            txt_embed = model.txt_enc(t_txt)
            
        t_tab.requires_grad = True
        t_img.requires_grad = True
        with torch.enable_grad():
            img_embed = model.img_enc(t_img)
            tab_embed = model.tab_enc(t_tab)
            fused = torch.cat((tab_embed, img_embed, txt_embed), dim=1)
            raw_logits = model.classifier(fused)
            
            # Get logits for the critical class (index 1)
            critical_logit = raw_logits[0, 1]
            
            # Efficiently compute ONLY t_tab and t_img gradients
            grads = torch.autograd.grad(outputs=critical_logit, inputs=(t_tab, t_img))
            
            # Tabular: Input * Gradient feature attribution (proxy for SHAP)
            tab_gradients = grads[0][0].detach().numpy()
            tab_inputs = t_tab[0].detach().numpy()
            attributions = tab_gradients * tab_inputs
            
            # Image: Input * Gradient saliency map (proxy for Grad-CAM)
            img_gradients = grads[1][0].detach().numpy() # Shape: (1, 224, 224)
            img_inputs = t_img[0].detach().numpy()
            img_saliency = np.abs(img_gradients * img_inputs).squeeze() # Shape: (224, 224)
            if np.max(img_saliency) > 0:
                img_saliency = img_saliency / np.max(img_saliency)
            img_saliency_list = np.round(img_saliency, 3).tolist()
            
        with torch.no_grad():
            probs = torch.softmax(raw_logits, dim=1)
            predicted_class = int(torch.argmax(raw_logits, dim=1).item())
            confidences = probs.tolist()[0]

        # Map feature attributions to readable names
        feature_names = [
            "Systolic BP", "Diastolic BP", "Heart Rate", "Respiratory Rate", 
            "SpO2", "Temperature", "WBC", "Creatinine", "CRP", "Troponin-I", 
            "Age", "Gender", "BMI", "Charlson Index"
        ]
        
        feature_importance = []
        for name, attr in zip(feature_names, attributions):
            feature_importance.append({
                "Feature": name,
                "Importance": float(attr)
            })


        # ---------------------------------------------------------------------
        # 7️⃣ Clinical safety override
        # ---------------------------------------------------------------------
        # Modified for real-world relevance: only critical direct indicators trigger immediate ICU override.
        # Temp, Charlson, and generic X-ray abnormalities are handled via doctor recommendations.
        severe = (
            (spO2_raw < Config.SEVERE_SPO2_THRESHOLD) or
            (temp_raw > Config.SEVERE_TEMP_THRESHOLD) or
            (troponin_raw is not None and troponin_raw > Config.SEVERE_TROPONIN_I_THRESHOLD) or
            optional_critical
        )
        if optional_risk_reasons or abnormal_xray_count > 0:
            xray_boost = 0.10 * abnormal_xray_count
            total_boost = optional_risk_score + xray_boost if optional_risk_reasons else xray_boost
            confidences[1] = min(0.99, max(confidences[1], confidences[1] + total_boost))
            confidences[0] = 1 - confidences[1]
            if optional_risk_reasons:
                ood_reasons.extend(optional_risk_reasons)
            if abnormal_xray_count > 0 and 'abnormal_xray_detected' not in ood_reasons:
                ood_reasons.append('abnormal_xray_detected')
        if severe:
            predicted_class = 1
            confidences[1] = max(confidences[1], 0.9)
            confidences[0] = 1 - confidences[1]
            if 'severity_override' not in ood_reasons:
                ood_reasons.append('severity_override')

        # ---------------------------------------------------------------------
        # 8️⃣ Reason labeling & recommendations
        # ---------------------------------------------------------------------
        reason_labels = {
            'tabular_z_score_exceeded': 'One or more core clinical values are far outside the expected range.',
            'tabular_many_features_out_of_range': 'Multiple core clinical values are outside the expected range.',
            'abnormal_xray_detected': 'Chest X-ray simulation contains abnormal image findings.',
            'heuristic_xray_intensity_abnormal': 'Note: Image intensity heuristic triggered. Professional clinical review of imaging is required to confirm abnormalities.',
            'severity_override': 'Clinical safety rules escalated this case to critical risk.',
            'hemoglobin_critical': 'Hemoglobin is in a critical range.',
            'wbc_optional_critical': 'White blood cell count is in a critical range.',
            'platelets_critical': 'Platelet count is in a critical range.',
            'esr_high': 'ESR is markedly elevated.',
            'pt_inr_high': 'PT/INR is elevated, suggesting bleeding or anticoagulation risk.',
            'aptt_high': 'aPTT is elevated, suggesting coagulation abnormality.',
            'glucose_critical': 'Glucose is in a critical range.',
            'hba1c_high': 'HbA1c is very high, suggesting poor glycemic control.',
            'egfr_low': 'eGFR is low, suggesting severe kidney impairment.',
            'microalbumin_high': 'Microalbumin is markedly elevated.',
            'alt_critical': 'ALT is critically elevated.',
            'ast_critical': 'AST is critically elevated.',
            'albumin_low': 'Albumin is low, suggesting poor nutrition, inflammation, or liver/kidney disease.',
            'bilirubin_high': 'Bilirubin is elevated, suggesting possible liver or biliary dysfunction.',
            'calcium_critical': 'Calcium is in a critical range.'
        }
        clinical_risk_reasons = [
            reason_labels.get(reason, reason.replace('_', ' ').title())
            for reason in ood_reasons
        ]

        doctor_recommendations = []
        if confidences[1] > 0.5:
            doctor_recommendations.append('Review patient immediately and correlate with bedside assessment.')
            if spO2_raw < Config.SEVERE_SPO2_THRESHOLD:
                doctor_recommendations.append('Assess airway and breathing; consider oxygen support and urgent ABG if clinically indicated.')
            if temp_raw > Config.SEVERE_TEMP_THRESHOLD:
                doctor_recommendations.append('Evaluate for sepsis or severe infection; consider cultures, lactate, and empiric therapy per protocol.')
            if troponin_raw is not None and troponin_raw > Config.SEVERE_TROPONIN_I_THRESHOLD:
                doctor_recommendations.append('Obtain or review ECG and repeat troponin; consider urgent cardiology evaluation.')
            if charlson_raw is not None and charlson_raw > Config.SEVERE_CHARLSON_THRESHOLD:
                doctor_recommendations.append('Account for high comorbidity burden when deciding monitoring level or admission.')
            if abnormal_xray_image or abnormal_xray_count >= Config.SEVERE_XRAY_ABNORMAL_COUNT:
                doctor_recommendations.append('Review chest imaging and consider respiratory or infectious differential based on findings.')
            if 'wbc_optional_critical' in optional_risk_reasons or 'esr_high' in optional_risk_reasons:
                doctor_recommendations.append('Check for active infection or inflammation; trend CBC and inflammatory markers.')
            if 'pt_inr_high' in optional_risk_reasons or 'aptt_high' in optional_risk_reasons or 'platelets_critical' in optional_risk_reasons:
                doctor_recommendations.append('Assess bleeding risk and medication history; consider coagulation workup or hematology input.')
            if 'glucose_critical' in optional_risk_reasons or 'hba1c_high' in optional_risk_reasons:
                doctor_recommendations.append('Manage dysglycemia per protocol and check ketones or osmolality if symptoms suggest crisis.')
            if 'egfr_low' in optional_risk_reasons or 'microalbumin_high' in optional_risk_reasons:
                doctor_recommendations.append('Assess renal function trend, hydration status, nephrotoxic medications, and urine output.')
            if 'alt_critical' in optional_risk_reasons or 'ast_critical' in optional_risk_reasons or 'bilirubin_high' in optional_risk_reasons or 'albumin_low' in optional_risk_reasons:
                doctor_recommendations.append('Review liver panel trend, medication or toxin exposure, hepatitis risk, and abdominal symptoms.')
            if 'calcium_critical' in optional_risk_reasons:
                doctor_recommendations.append('Repeat calcium with albumin correction or ionized calcium and evaluate ECG/symptoms.')

        risk_status = "CRITICAL / IMMEDIATE ICU ATTENTION" if predicted_class == 1 else "STABLE / ROUTINE WARD MONITORING"
        
        return {
            "status": "success",
            "predicted_class": predicted_class,
            "diagnostic_label": risk_status,
            "raw_logits": raw_logits.detach().numpy().tolist()[0],
            "confidence_scores": {
                "class_0_stable": confidences[0],
                "class_1_critical": confidences[1]
            },
            "ood": len(ood_reasons) > 0,
            "ood_reasons": ood_reasons,
            "clinical_risk_reasons": clinical_risk_reasons,
            "doctor_recommendations": doctor_recommendations,
            "feature_importance": feature_importance,
            "image_heatmap": img_saliency_list
        }
    except Exception as error:
        return {"status": "error", "message": str(error)}
