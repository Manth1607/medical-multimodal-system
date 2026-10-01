import re
from datetime import datetime
from fpdf import FPDF


def clean_for_pdf(text: str) -> str:
    """Strip markdown, emojis and non-latin1 characters so FPDF can render the text."""
    if not text:
        return ""
    # Normalize Windows/Mac line endings first
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove markdown bold/italic (handles multiline too)
    text = re.sub(r'\*{1,3}([^*]+?)\*{1,3}', r'\1', text, flags=re.DOTALL)
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Replace markdown bullets with plain dash
    text = re.sub(r'^\s*[-*+]\s+', '- ', text, flags=re.MULTILINE)
    # Remove emojis and non-latin1 characters
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class ClinicalReportGenerator(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def _eff_w(self):
        return self.w - self.l_margin - self.r_margin

    def header(self):
        self.set_fill_color(15, 30, 60)
        self.rect(0, 0, self.w, 18, 'F')
        self.set_font('Arial', 'B', 13)
        self.set_text_color(255, 255, 255)
        self.set_y(4)
        self.cell(self._eff_w(), 10, 'MEDICAL MULTIMODAL DIAGNOSTICS REPORT', 0, 0, 'C')
        self.set_text_color(0, 0, 0)
        self.set_y(20)

    def footer(self):
        self.set_y(-14)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(120, 120, 120)
        half = self._eff_w() / 2
        self.cell(half, 8, f'Generated: {datetime.now().strftime("%d/%m/%Y %H:%M")}  |  Page {self.page_no()}', 0, 0, 'L')
        self.cell(half, 8, 'CONFIDENTIAL MEDICAL RECORD', 0, 0, 'R')
        self.set_text_color(0, 0, 0)

    def section_title(self, title: str):
        self.ln(4)
        self.set_fill_color(230, 236, 255)
        self.set_font('Arial', 'B', 11)
        self.cell(self._eff_w(), 8, f'  {title}', 0, 1, 'L', fill=True)
        self.ln(2)

    def add_patient_info(self, info: dict):
        self.section_title('PATIENT INFORMATION')
        self.set_font('Arial', '', 10)
        label_w = 55
        val_w = self._eff_w() - label_w
        for key, value in info.items():
            self.set_font('Arial', 'B', 10)
            self.cell(label_w, 7, f"{str(key).replace('_', ' ').title()}:", 0, 0)
            self.set_font('Arial', '', 10)
            self.cell(val_w, 7, clean_for_pdf(str(value)), 0, 1)
        self.ln(3)

    def add_vitals(self, tabular: list):
        self.section_title('VITALS & LABORATORY RESULTS')
        labels = [
            "Systolic BP (mmHg)", "Diastolic BP (mmHg)", "Heart Rate (bpm)",
            "Respiratory Rate (/min)", "SpO2 (%)", "Temperature (C)",
            "WBC (x10^9/L)", "Creatinine (mg/dL)", "CRP (mg/L)",
            "Troponin-I (ng/mL)", "Age (Years)", "Gender (0=F, 1=M)",
            "BMI", "Charlson Comorbidity Index"
        ]
        col1 = self._eff_w() * 0.68
        col2 = self._eff_w() * 0.32

        self.set_fill_color(50, 80, 150)
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 9)
        self.cell(col1, 7, '  Parameter', 1, 0, 'L', fill=True)
        self.cell(col2, 7, 'Value', 1, 1, 'C', fill=True)
        self.set_text_color(0, 0, 0)

        for idx, (label, val) in enumerate(zip(labels, tabular)):
            fill = idx % 2 == 0
            self.set_fill_color(245, 248, 255) if fill else self.set_fill_color(255, 255, 255)
            self.set_font('Arial', '', 9)
            self.cell(col1, 6, f'  {label}', 0, 0, 'L', fill=True)
            self.cell(col2, 6, str(val), 0, 1, 'C', fill=True)
        self.ln(4)

    def add_diagnostic_summary(self, result: dict):
        self.section_title('AI DIAGNOSTIC ASSESSMENT')
        risk_label = result.get('diagnostic_label', 'Unknown')
        is_critical = 'CRITICAL' in risk_label.upper()

        # Risk Badge
        if is_critical:
            self.set_fill_color(220, 53, 69)
        else:
            self.set_fill_color(40, 167, 69)
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 12)
        self.cell(self._eff_w(), 10, f'  STATUS: {clean_for_pdf(risk_label)}', 0, 1, 'L', fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

        # Confidence
        confs = result.get('confidence_scores', {})
        c_stable = confs.get('class_0_stable', 0) * 100
        c_crit = confs.get('class_1_critical', 0) * 100
        col = self._eff_w() / 2
        self.set_font('Arial', 'B', 10)
        self.cell(col, 7, f'Stable Probability: {c_stable:.1f}%', 1, 0, 'C')
        self.set_text_color(220, 53, 69) if c_crit > 60 else self.set_text_color(40, 167, 69)
        self.cell(col, 7, f'Critical Probability: {c_crit:.1f}%', 1, 1, 'C')
        self.set_text_color(0, 0, 0)
        self.ln(3)

        # OOD / Alert Flags
        ood = result.get('ood_reasons', [])
        if ood:
            self.set_font('Arial', 'B', 10)
            self.cell(self._eff_w(), 6, 'Alert Flags Detected:', 0, 1)
            self.set_font('Arial', 'I', 9)
            self.set_text_color(200, 50, 0)
            for flag in ood:
                self.cell(self._eff_w(), 5, f'  - {flag}', 0, 1)
            self.set_text_color(0, 0, 0)
            self.ln(2)

        # Clinical Risk Reasons
        clin_risks = result.get('clinical_risk_reasons', [])
        if clin_risks:
            self.set_font('Arial', 'B', 10)
            self.cell(self._eff_w(), 6, 'Clinical Risk Factors Identified:', 0, 1)
            self.set_font('Arial', '', 9)
            for reason in clin_risks:
                self.multi_cell(self._eff_w(), 5, f'  - {clean_for_pdf(str(reason))}', new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

        # Feature Importance
        fi = result.get('feature_importance', [])
        if fi:
            self.section_title('TOP RISK FEATURE CONTRIBUTIONS (SHAP-Style)')
            col1 = self._eff_w() * 0.60
            col2 = self._eff_w() * 0.22
            col3 = self._eff_w() * 0.18

            self.set_fill_color(50, 80, 150)
            self.set_text_color(255, 255, 255)
            self.set_font('Arial', 'B', 9)
            self.cell(col1, 6, '  Feature', 1, 0, 'L', fill=True)
            self.cell(col2, 6, 'Impact Score', 1, 0, 'C', fill=True)
            self.cell(col3, 6, 'Effect', 1, 1, 'C', fill=True)
            self.set_text_color(0, 0, 0)

            # Sort by abs importance, show top 10
            sorted_fi = sorted(fi, key=lambda x: abs(x.get('Importance', 0)), reverse=True)[:10]
            for idx, item in enumerate(sorted_fi):
                imp = item.get('Importance', 0)
                effect = 'Increases Risk' if imp > 0 else 'Decreases Risk'
                fill = idx % 2 == 0
                self.set_fill_color(245, 248, 255) if fill else self.set_fill_color(255, 255, 255)
                self.set_font('Arial', '', 9)
                self.cell(col1, 6, f"  {clean_for_pdf(str(item.get('Feature', '')))}", 0, 0, 'L', fill=True)
                self.set_text_color(200, 40, 40) if imp > 0 else self.set_text_color(40, 150, 70)
                self.cell(col2, 6, f'{imp:+.4f}', 0, 0, 'C', fill=True)
                self.cell(col3, 6, effect, 0, 1, 'C', fill=True)
                self.set_text_color(0, 0, 0)
            self.ln(4)

        # Doctor Recommendations
        recs = result.get('doctor_recommendations', [])
        if recs:
            self.section_title('CLINICAL RECOMMENDATIONS')
            self.set_font('Arial', '', 10)
            for i, rec in enumerate(recs, 1):
                self.multi_cell(self._eff_w(), 6, clean_for_pdf(f'{i}. {rec}'), new_x="LMARGIN", new_y="NEXT")
            self.ln(3)

    def add_ai_recs(self, ai_text: str):
        cleaned = clean_for_pdf(ai_text)
        if not cleaned:
            return
        self.section_title('AI-GENERATED DETAILED RECOMMENDATIONS (Gemini)')
        self.set_font('Arial', '', 10)
        w = self._eff_w()
        # Write paragraph by paragraph so page breaks work correctly
        paragraphs = cleaned.split('\n')
        for para in paragraphs:
            para = para.strip()
            if para:
                self.multi_cell(w, 6, para, new_x="LMARGIN", new_y="NEXT")
            else:
                self.ln(3)  # blank line between sections
        self.ln(3)


def generate_pdf_report(payload: dict, result: dict, ai_gujarati_recs: str = None) -> bytes:
    """Generate a comprehensive PDF report and return it as bytes."""
    pdf = ClinicalReportGenerator()
    pdf.add_page()

    # Page 1 — Patient Info + Vitals
    if 'patient_info' in payload:
        pdf.add_patient_info(payload['patient_info'])

    if 'tabular' in payload:
        pdf.add_vitals(payload['tabular'])

    # Page 2 — Diagnostic Results
    pdf.add_page()
    pdf.add_diagnostic_summary(result)

    # AI Recommendations (always show if provided)
    if ai_gujarati_recs:
        pdf.add_ai_recs(ai_gujarati_recs)

    pdf_out = pdf.output()
    if isinstance(pdf_out, str):
        return pdf_out.encode('latin1')
    return bytes(pdf_out)
