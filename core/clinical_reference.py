"""
Clinical Laboratory Reference Ranges Database - CORRECTED
Comprehensive normal values for diagnosis support
"""


CLINICAL_REFERENCE_RANGES = {
    "CBC": {
        "hemoglobin": {"name": "Hemoglobin", "unit": "g/dL", "male_range": (13.0, 16.0), "female_range": (11.5, 15.5), "critical_low": 7.0},
        "wbc": {"name": "WBC", "unit": "/μL", "normal_range": (4000, 11000), "critical_low": 2000},
        "platelets": {"name": "Platelets", "unit": "/μL", "normal_range": (150000, 450000), "critical_low": 50000},
        "esr": {"name": "ESR", "unit": "mm/hr", "male_range": (0, 15), "female_range": (0, 20)},
        "crp": {"name": "CRP", "unit": "mg/dL", "normal_range": (0.0, 1.0)},
    },
    "Coagulation": {
        "pt_inr": {"name": "PT/INR", "unit": "ratio", "normal_range": (0.8, 1.2)},
        "appt": {"name": "aPTT", "unit": "seconds", "normal_range": (25, 35)},
    },
    "Endocrine": {
        "fasting_glucose": {"name": "Glucose", "unit": "mg/dL", "normal_range": (70, 100), "critical_low": 40},
        "hba1c": {"name": "HbA1c", "unit": "%", "normal_range": (0.0, 5.7)},
        "tsh": {"name": "TSH", "unit": "mIU/L", "normal_range": (0.4, 4.0)},
        "free_t4": {"name": "Free T4", "unit": "ng/dL", "normal_range": (0.8, 2.8)},
        "serum_calcium": {"name": "Calcium", "unit": "mg/dL", "normal_range": (8.2, 10.2), "critical_low": 6.5},
        "vitamin_d": {"name": "Vitamin D", "unit": "ng/mL", "normal_range": (30, 100)},
        "serum_cortisol_am": {"name": "Cortisol AM", "unit": "μg/dL", "normal_range": (5, 23)},
        "serum_cortisol_pm": {"name": "Cortisol PM", "unit": "μg/dL", "normal_range": (3, 16)},
        "total_t3": {"name": "Total T3", "unit": "ng/dL", "normal_range": (80, 200)},
    },
    "Kidney": {
        "creatinine": {"name": "Creatinine", "unit": "mg/dL", "male_range": (0.7, 1.3), "female_range": (0.6, 1.1), "critical_high": 10.0},
        "egfr": {"name": "eGFR", "unit": "mL/min", "normal_range": (90, 200)},
        "microalbumin": {"name": "Microalbumin", "unit": "mg/g", "normal_range": (0, 30)},
    },
    "Liver": {
        "alt": {"name": "ALT", "unit": "U/L", "normal_range": (10, 40), "critical_high": 500},
        "ast": {"name": "AST", "unit": "U/L", "normal_range": (10, 40), "critical_high": 500},
        "alp": {"name": "ALP", "unit": "U/L", "normal_range": (44, 147)},
        "ggt": {"name": "GGT", "unit": "U/L", "normal_range": (5, 40)},
        "albumin": {"name": "Albumin", "unit": "g/dL", "normal_range": (3.5, 5.0), "critical_low": 2.0},
        "bilirubin_total": {"name": "Bilirubin", "unit": "mg/dL", "normal_range": (0.1, 1.2)},
    },
    "Infectious": {
        "widal": {"name": "Widal", "normal": "< 1:80"},
        "dengue": {"name": "Dengue", "normal": "Negative"},
        "malaria": {"name": "Malaria", "normal": "Negative"},
        "hiv": {"name": "HIV", "normal": "Negative"},
        "hbsag": {"name": "HBsAg", "normal": "Negative"},
        "anti_hcv": {"name": "Anti-HCV", "normal": "Negative"},
        "covid": {"name": "COVID PCR", "normal": "Negative"},
        "widal_o": {"name": "Widal O", "normal": "< 1:80"},
        "dengue_ns1": {"name": "Dengue NS1", "normal": "Negative"},
    },
    "Tumor": {
        "psa": {"name": "PSA", "unit": "ng/mL", "normal_range": (0, 4.0)},
        "ca125": {"name": "CA-125", "unit": "U/mL", "normal_range": (0, 35)},
        "afp": {"name": "AFP", "unit": "ng/mL", "normal_range": (0, 5.4)},
        "cea": {"name": "CEA", "unit": "ng/mL", "normal_range": (0, 2.5)},
    },
    "Autoimmune": {
        "ana": {"name": "ANA", "normal": "Negative"},
        "ra_factor": {"name": "RA Factor", "unit": "IU/mL", "normal_range": (0, 14)},
        "anti_ccp": {"name": "Anti-CCP", "unit": "U/mL", "normal_range": (0, 20)},
        "anti_dsdna": {"name": "Anti-dsDNA", "unit": "IU/mL", "normal_range": (0, 10)},
    },
}

def check_value_status(category: str, test: str, value: float, gender: str = None) -> dict:
    """Check if a lab value is normal, abnormal, or critical"""
    try:
        if category not in CLINICAL_REFERENCE_RANGES or test not in CLINICAL_REFERENCE_RANGES[category]:
            return {'status': 'unknown', 'range': 'N/A', 'message': f'Test not found'}
        
        info = CLINICAL_REFERENCE_RANGES[category][test]
        
        # Check critical values first
        if 'critical_low' in info and value < info['critical_low']:
            return {'status': 'critical', 'range': str(value), 'message': f'CRITICAL LOW'}
        if 'critical_high' in info and value > info['critical_high']:
            return {'status': 'critical', 'range': str(value), 'message': f'CRITICAL HIGH'}
        
        # Gender-specific ranges
        if 'male_range' in info and 'female_range' in info:
            if gender and gender.lower() == 'male':
                lo, hi = info['male_range']
                rng = f"{lo}-{hi}"
                return {'status': 'normal' if lo <= value <= hi else 'abnormal', 'range': rng, 'message': 'In range' if lo <= value <= hi else 'Out of range'}
            elif gender and gender.lower() == 'female':
                lo, hi = info['female_range']
                rng = f"{lo}-{hi}"
                return {'status': 'normal' if lo <= value <= hi else 'abnormal', 'range': rng, 'message': 'In range' if lo <= value <= hi else 'Out of range'}
        
        # Standard range
        if 'normal_range' in info:
            lo, hi = info['normal_range']
            rng = f"{lo}-{hi}"
            return {'status': 'normal' if lo <= value <= hi else 'abnormal', 'range': rng, 'message': 'Normal' if lo <= value <= hi else 'Abnormal'}
        
        return {'status': 'info', 'range': str(value), 'message': 'Value recorded'}
    except Exception as e:
        return {'status': 'error', 'range': 'N/A', 'message': 'Error checking value: Invalid input type or structure'}

def get_all_categories() -> list:
    """Get list of all test categories"""
    return list(CLINICAL_REFERENCE_RANGES.keys())

def get_tests_in_category(category: str) -> dict:
    """Get all tests in a category"""
    return CLINICAL_REFERENCE_RANGES.get(category, {})
