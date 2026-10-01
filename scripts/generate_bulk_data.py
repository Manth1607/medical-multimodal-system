import sys
import os
import random
import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

db_path = os.path.join(PROJECT_DIR, "database", "clinical_data.db")
if "--reset" in sys.argv and os.path.exists(db_path):
    os.remove(db_path)
    print("Old database removed (--reset flag provided).")

from database.database import DatabaseManager
from ml.prediction_service import run_prediction

def generate_bulk_data():
    db = DatabaseManager()
    
    first_names = ["Amit", "Rahul", "Sita", "Priya", "John", "Jane", "Ramesh", "Suresh", "Geeta", "Anjali", "Vikram", "Neha", "Pooja", "Raj", "Karan"]
    last_names = ["Patel", "Sharma", "Desai", "Verma", "Singh", "Shah", "Mehta", "Joshi", "Chauhan", "Rao"]
    cities = ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Mumbai", "Pune", "Delhi", "Bengaluru"]

    healthy_notes = [
        "Patient came for a routine checkup. No major complaints. Blood pressure is normal.",
        "Annual physical exam. Patient is feeling well, vitals are stable.",
        "Follow-up for mild headache, currently asymptomatic.",
        "Patient reports good health. No respiratory distress.",
        "Routine blood work follow-up. All parameters are within normal range."
    ]

    critical_notes = [
        "Patient brought to ER with severe chest pain radiating to left arm. High suspicion of acute myocardial infarction.",
        "Severe shortness of breath, SpO2 dropping to 75% on room air. Cyanosis observed.",
        "Patient presents with high grade fever (103F), chills, and altered mental status. Suspected sepsis.",
        "Acute respiratory distress. Patient is gasping for air, requiring immediate intubation.",
        "Hypotensive shock. Blood pressure critically low. Immediate ICU transfer recommended."
    ]

    def generate_patient():
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        age = random.randint(30, 80)
        gender = random.choice(["Male", "Female"])
        mrn = f"MRN-{random.randint(100000, 999999)}"
        dob = (datetime.date.today() - datetime.timedelta(days=age*365)).strftime("%Y-%m-%d")
        contact = f"9{random.randint(100000000, 999999999)}"
        return {
            "name": name, "age": age, "gender": gender, "mrn": mrn, "dob": dob,
            "contact": contact, "email": f"{name.split()[0].lower()}@example.com",
            "address": f"{random.choice(cities)}, India"
        }

    def generate_assessment(age, gender, status="healthy"):
        if status == "healthy":
            vitals = {
                "systolic_bp": random.randint(110, 130),
                "diastolic_bp": random.randint(70, 85),
                "heart_rate": random.randint(60, 90),
                "respiratory_rate": random.randint(12, 18),
                "spo2": random.randint(95, 100),
                "body_temperature": round(random.uniform(36.5, 37.2), 1)
            }
            labs = {
                "wbc": round(random.uniform(4.0, 10.0), 1),
                "serum_creatinine": round(random.uniform(0.6, 1.2), 1),
                "hemoglobin": round(random.uniform(12.0, 16.0), 1),
                "hematocrit": round(random.uniform(36.0, 48.0), 1),
                "platelets": random.randint(150, 400),
                "glucose": random.randint(80, 120),
                "bun": round(random.uniform(7.0, 20.0), 1),
                "sodium": random.randint(135, 145)
            }
            notes = random.choice(healthy_notes)
            diag = "Healthy / Routine"
        else:
            # Extreme values to trigger override and high ML logits
            vitals = {
                "systolic_bp": random.choice([random.randint(180, 220), random.randint(60, 80)]),
                "diastolic_bp": random.choice([random.randint(110, 130), random.randint(40, 50)]),
                "heart_rate": random.randint(120, 150),
                "respiratory_rate": random.randint(28, 40),
                "spo2": random.randint(70, 84), # Below 85 triggers override!
                "body_temperature": round(random.uniform(39.5, 41.0), 1)
            }
            labs = {
                "wbc": round(random.uniform(20.0, 35.0), 1),
                "serum_creatinine": round(random.uniform(2.5, 5.0), 1),
                "hemoglobin": round(random.uniform(6.0, 9.0), 1),
                "hematocrit": round(random.uniform(20.0, 28.0), 1),
                "platelets": random.randint(20, 80),
                "glucose": random.randint(300, 500),
                "bun": round(random.uniform(40.0, 80.0), 1),
                "sodium": random.choice([random.randint(115, 125), random.randint(155, 165)])
            }
            notes = random.choice(critical_notes)
            diag = "Critical / Immediate ICU Required"

        return vitals, labs, notes, diag

    simulated_image = [[0.0]*224 for _ in range(224)]
    
    total_patients = 50
    print(f"Generating {total_patients} patients...")

    for i in range(total_patients):
        p = generate_patient()
        pid = db.add_patient(**p)
        
        num_assessments = 2 if random.random() < 0.5 else 1
        
        for a_idx in range(num_assessments):
            status = "healthy" if a_idx == 0 else "critical"
            vitals, labs, notes, diag = generate_assessment(p["age"], p["gender"], status)
            aid = db.add_assessment(pid, vitals, labs, notes, diag)
            
            gender_val = 1.0 if p["gender"] == "Male" else 0.0
            crp = 5.0 if status == "healthy" else random.uniform(80.0, 150.0)
            troponin = 0.01 if status == "healthy" else random.uniform(0.6, 2.5) # >0.5 triggers override
            
            tabular = [
                vitals["systolic_bp"], vitals["diastolic_bp"], vitals["heart_rate"], vitals["respiratory_rate"],
                vitals["spo2"], vitals["body_temperature"], labs["wbc"], labs["serum_creatinine"],
                crp, troponin, float(p["age"]), gender_val, 25.0, 1.0
            ]
            
            payload = {
                "patient_info": {"cardiomegaly": status=="critical", "pleural_effusion": status=="critical", "consolidation": status=="critical", "lung_nodule": False},
                "tabular": tabular,
                "image": simulated_image,
                "text": notes
            }
            
            pred_res = run_prediction(payload)
            
            if pred_res.get("status") == "success":
                risk_score = pred_res["confidence_scores"]["class_1_critical"] * 100
                risk_cat = "High" if risk_score >= 50 else "Low"
                severity = pred_res["diagnostic_label"]
                conf = max(pred_res["confidence_scores"]["class_1_critical"], pred_res["confidence_scores"]["class_0_stable"]) * 100
                db.update_assessment_predictions(aid, risk_score, risk_cat, severity, conf)

    print("Database recreated and 50 patients added with varied clinical notes and extreme critical scores (90%+).")

if __name__ == '__main__':
    generate_bulk_data()
  