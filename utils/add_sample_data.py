#!/usr/bin/env python3
"""
Sample Data Entry Script
Adds sample patients and assessments to the database system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import DatabaseManager, init_database
from datetime import datetime, timedelta

# Force UTF-8 output on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

print("="*80)
print("DATABASE SYSTEM - SAMPLE DATA ENTRY")
print("="*80)

# Initialize database
print("\n✓ Initializing database...")
init_database()
db = DatabaseManager()
print("✓ Database initialized successfully")

# Sample Patient Data
sample_patients = [
    {
        "name": "Rajesh Kumar",
        "age": 52,
        "gender": "Male",
        "mrn": "MRN-2026-001",
        "dob": "1974-03-20",
        "contact": "+91-9876543210",
        "email": "rajesh.kumar@hospital.com",
        "address": "456 Apollo Hospital, Delhi-110001, India"
    },
    {
        "name": "Priya Sharma",
        "age": 38,
        "gender": "Female",
        "mrn": "MRN-2026-002",
        "dob": "1988-07-15",
        "contact": "+91-8765432109",
        "email": "priya.sharma@hospital.com",
        "address": "789 Max Hospital, Mumbai-400015, India"
    },
    {
        "name": "David Wilson",
        "age": 65,
        "gender": "Male",
        "mrn": "MRN-2026-003",
        "dob": "1961-11-08",
        "contact": "+1-555-9999",
        "email": "david.wilson@hospital.com",
        "address": "321 St. Mary's Hospital, New York-10001, USA"
    }
]

# Add Patients
print("\n" + "="*80)
print("ADDING SAMPLE PATIENTS TO DATABASE")
print("="*80)

patient_ids = []
existing_patients = {p.get('medical_record_number'): p['patient_id'] for p in db.get_all_patients() if p.get('medical_record_number')}

for i, patient in enumerate(sample_patients, 1):
    try:
        if patient["mrn"] in existing_patients:
            patient_id = existing_patients[patient["mrn"]]
            print(f"\n✓ Patient {i} Already Exists in Database (ID: {patient_id})")
            print(f"  Name: {patient['name']}")
            print(f"  Age: {patient['age']} years")
            print(f"  Gender: {patient['gender']}")
            print(f"  MRN: {patient['mrn']}")
            print(f"  Email: {patient['email']}")
        else:
            patient_id = db.add_patient(
                name=patient["name"],
                age=patient["age"],
                gender=patient["gender"],
                mrn=patient["mrn"],
                dob=patient["dob"],
                contact=patient["contact"],
                email=patient["email"],
                address=patient["address"]
            )
            print(f"\n✓ Patient {i} Added Successfully")
            print(f"  ID: {patient_id}")
            print(f"  Name: {patient['name']}")
            print(f"  Age: {patient['age']} years")
            print(f"  Gender: {patient['gender']}")
            print(f"  MRN: {patient['mrn']}")
            print(f"  Email: {patient['email']}")
        patient_ids.append((patient_id, patient["mrn"]))
    except Exception as e:
        print(f"✗ Error adding patient {i}: {str(e)}")

# Sample Assessment Data
assessments_data = [
    {
        "patient_id": 1,
        "sbp": 130,
        "dbp": 85,
        "hr": 78,
        "rr": 18,
        "spo2": 96,
        "temp": 37.2,
        "wbc": 8.5,
        "scr": 1.1,
        "crp": 8.5,
        "trop": 0.03,
        "bmi": 26.5,
        "charlson": 2,
        "risk_score": 0.42,
        "risk_category": "Moderate Risk",
        "severity": "Stable",
        "notes": "Patient presents with elevated BP. On antihypertensive therapy. Routine follow-up."
    },
    {
        "patient_id": 2,
        "sbp": 118,
        "dbp": 76,
        "hr": 72,
        "rr": 16,
        "spo2": 98,
        "temp": 36.8,
        "wbc": 7.2,
        "scr": 0.8,
        "crp": 2.1,
        "trop": 0.01,
        "bmi": 23.2,
        "charlson": 1,
        "risk_score": 0.15,
        "risk_category": "Low Risk",
        "severity": "Stable",
        "notes": "Healthy female patient. Routine health checkup. All vitals normal."
    },
    {
        "patient_id": 3,
        "sbp": 145,
        "dbp": 92,
        "hr": 88,
        "rr": 20,
        "spo2": 94,
        "temp": 37.5,
        "wbc": 10.2,
        "scr": 1.4,
        "crp": 12.5,
        "trop": 0.05,
        "bmi": 28.5,
        "charlson": 4,
        "risk_score": 0.68,
        "risk_category": "High Risk",
        "severity": "Requires Monitoring",
        "notes": "Elderly male with multiple comorbidities. Hypertension and elevated inflammatory markers. ICU monitoring recommended."
    }
]

# Add Assessments
print("\n" + "="*80)
print("ADDING CLINICAL ASSESSMENTS")
print("="*80)

mrn_map = {p.get('medical_record_number'): p['patient_id'] for p in db.get_all_patients() if p.get('medical_record_number')}
assessment_ids = []
for i, assessment in enumerate(assessments_data, 1):
    try:
        # Resolve real patient ID by matching MRN
        mrn = f"MRN-2026-00{i}"
        actual_patient_id = mrn_map.get(mrn, assessment["patient_id"])
        
        vitals = {
            "systolic_bp": assessment["sbp"],
            "diastolic_bp": assessment["dbp"],
            "heart_rate": assessment["hr"],
            "respiratory_rate": assessment["rr"],
            "spo2": assessment["spo2"],
            "body_temperature": assessment["temp"]
        }
        labs = {
            "wbc": assessment["wbc"],
            "serum_creatinine": assessment["scr"],
            "glucose": 100.0,
            "bun": 15.0,
            "sodium": 138.0,
            "hemoglobin": 14.0,
            "hematocrit": 42.0,
            "platelets": 250.0,
            "crp": assessment["crp"],
            "trop": assessment["trop"],
            "bmi": assessment["bmi"],
            "charlson": assessment["charlson"]
        }
        
        assessment_id = db.add_assessment(
            patient_id=actual_patient_id,
            vitals=vitals,
            labs=labs,
            clinical_notes=assessment["notes"],
            diagnosis=assessment["risk_category"]
        )
        assessment_ids.append(assessment_id)
        
        # Update predictions
        db.update_assessment_predictions(
            assessment_id=assessment_id,
            risk_score=assessment["risk_score"],
            risk_category=assessment["risk_category"],
            severity_level=assessment["severity"],
            confidence=0.92 if assessment["risk_category"] == "High Risk" else 0.85
        )
        
        # Add clinical note
        db.add_text_note(
            assessment_id=assessment_id,
            content=assessment["notes"],
            note_type="Clinical Assessment"
        )
        
        patient = db.get_patient(actual_patient_id)
        patient_name = patient['name'] if patient else f"Patient {actual_patient_id}"
        print(f"\n✓ Assessment {i} Added Successfully")
        print(f"  Assessment ID: {assessment_id}")
        print(f"  Patient: {patient_name} (ID: {actual_patient_id})")
        print(f"  Vitals: BP {assessment['sbp']}/{assessment['dbp']} | HR {assessment['hr']} | SpO2 {assessment['spo2']}%")
        print(f"  Risk Score: {assessment['risk_score']:.2f}")
        print(f"  Risk Category: {assessment['risk_category']}")
        print(f"  Severity: {assessment['severity']}")
        print(f"  Notes: {assessment['notes'][:50]}...")
    except Exception as e:
        print(f"✗ Error adding assessment {i}: {str(e)}")

# Display Summary
print("\n" + "="*80)
print("DATABASE SUMMARY")
print("="*80)

# Get all patients
all_patients = db.get_all_patients()
print(f"\n✓ Total Patients: {len(all_patients)}")
for patient in all_patients:
    assessments = db.get_patient_assessments(patient['patient_id'])
    print(f"\n  {patient['patient_id']}. {patient['name']}")
    print(f"     Age: {patient['age']} | Gender: {patient['gender']}")
    print(f"     MRN: {patient.get('medical_record_number', 'N/A')}")
    print(f"     Email: {patient.get('email', 'N/A')}")
    print(f"     Assessments: {len(assessments)}")
    for assessment in assessments:
        score = assessment['risk_score'] if assessment.get('risk_score') is not None else 0.0
        print(f"       - Risk: {assessment.get('risk_category', 'N/A')} | Score: {score:.2f}")

# Get statistics
print("\n" + "="*80)
print("STATISTICS")
print("="*80)

print(f"\n✓ Total Patients: {db.get_patient_count()}")
print(f"✓ Total Assessments: {db.get_assessment_count()}")

risk_stats = db.get_risk_statistics()
print(f"\n✓ Risk Distribution:")
for category, count in risk_stats.items():
    print(f"   {category}: {count}")

# Get audit log
audit_log = db.get_audit_log()
print(f"\n✓ Audit Log Entries: {len(audit_log)}")
print("\nRecent Audit Log:")
for i, entry in enumerate(audit_log[-5:], 1):
    print(f"  {i}. {entry.get('action', 'N/A')} | {entry.get('table_name', 'N/A')} | {entry.get('user_action', 'N/A')}")

print("\n" + "="*80)
print("✓ ALL DATA SUCCESSFULLY SAVED TO DATABASE!")
print("="*80)
print("\nDatabase File: ./clinical_data.db")
print("Ready to use in Streamlit Dashboard: http://localhost:8501")
print("\n" + "="*80)
