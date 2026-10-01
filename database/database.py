import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import bcrypt

DB_PATH = os.getenv(
    'DATABASE_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clinical_data.db'),
)

def init_database():
    """Initialize the SQLite database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            medical_record_number TEXT UNIQUE,
            date_of_birth TEXT,
            contact_number TEXT,
            email TEXT,
            address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Clinical assessments table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clinical_assessments (
            assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Vitals
            systolic_bp REAL,
            diastolic_bp REAL,
            heart_rate REAL,
            respiratory_rate REAL,
            spo2 REAL,
            body_temperature REAL,
            
            -- Core Lab Biomarkers
            wbc REAL,
            serum_creatinine REAL,
            hemoglobin REAL,
            hematocrit REAL,
            platelets REAL,
            glucose REAL,
            bun REAL,
            sodium REAL,
            
            -- Additional Clinical Data
            clinical_notes TEXT,
            diagnosis TEXT,
            
            -- Model Predictions
            risk_score REAL,
            risk_category TEXT,
            severity_level TEXT,
            model_confidence REAL,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    ''')
    
    # Image data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_data (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL,
            image_filename TEXT,
            image_path TEXT,
            image_type TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES clinical_assessments(assessment_id)
        )
    ''')
    
    # Text notes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL,
            note_type TEXT,
            note_content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES clinical_assessments(assessment_id)
        )
    ''')
    
    # Model predictions history
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_id INTEGER NOT NULL,
            model_version TEXT,
            tabular_input TEXT,
            prediction_output TEXT,
            execution_time_ms REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assessment_id) REFERENCES clinical_assessments(assessment_id)
        )
    ''')
    
    # Audit log
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            table_name TEXT,
            record_id INTEGER,
            user_action TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_clinical_assessments_patient_id ON clinical_assessments(patient_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_data_assessment_id ON image_data(assessment_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_text_notes_assessment_id ON text_notes(assessment_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_history_assessment_id ON prediction_history(assessment_id)')
    
    # Add default admin if users table is empty
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        admin_username = os.getenv('INITIAL_ADMIN_USERNAME')
        admin_password = os.getenv('INITIAL_ADMIN_PASSWORD')
        if admin_username and admin_password:
            hashed_pw = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            cursor.execute(
                'INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)',
                (admin_username, hashed_pw, 'admin')
            )
            print(f"[OK] Default admin user created (username: {admin_username})")
        else:
            print("[WARNING] INITIAL_ADMIN_USERNAME and/or INITIAL_ADMIN_PASSWORD not set. Skipping default admin creation.")
    
    conn.commit()
    conn.close()
    print(f"[OK] Database initialized at {DB_PATH}")


class DatabaseManager:
    """Manage all database operations for the clinical system."""
    
    def __init__(self):
        self.db_path = DB_PATH
        if not os.path.exists(self.db_path):
            init_database()
    
    def get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # ===== USER AUTH OPERATIONS =====
    def add_user(self, username: str, password: str, role: str = 'user') -> int:
        """Add a new user for authentication."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute(
            'INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)',
            (username, hashed_password, role)
        )
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.log_audit('INSERT', 'users', user_id, f'Added new user: {username}')
        return user_id
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Fetch a user by their username."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
        
    # ===== PATIENT OPERATIONS =====
    def add_patient(self, name: str, age: int, gender: str, mrn: str, 
                   dob: str = None, contact: str = None, email: str = None, 
                   address: str = None) -> int:
        """Add a new patient to the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patients 
            (name, age, gender, medical_record_number, date_of_birth, contact_number, email, address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, age, gender, mrn, dob, contact, email, address))
        
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.log_audit('INSERT', 'patients', patient_id, 'Added new patient')
        return patient_id
    
    def get_patient(self, patient_id: int) -> Optional[Dict]:
        """Retrieve patient information."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_all_patients(self) -> List[Dict]:
        """Get all patients."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def update_patient(self, patient_id: int, **kwargs) -> bool:
        """Update patient information."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        allowed_fields = ['name', 'age', 'gender', 'contact_number', 'email', 'address']
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [patient_id]
        
        cursor.execute(
            f'UPDATE patients SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE patient_id = ?',
            values
        )
        
        conn.commit()
        conn.close()
        
        self.log_audit('UPDATE', 'patients', patient_id, f'Updated fields: {list(updates.keys())}')
        return cursor.rowcount > 0
    
    def delete_patient(self, patient_id: int) -> bool:
        """Delete a patient and all their associated records."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get all assessments for this patient to delete related records
        cursor.execute('SELECT assessment_id FROM clinical_assessments WHERE patient_id = ?', (patient_id,))
        assessments = cursor.fetchall()
        
        for assessment in assessments:
            aid = assessment['assessment_id']
            cursor.execute('DELETE FROM image_data WHERE assessment_id = ?', (aid,))
            cursor.execute('DELETE FROM text_notes WHERE assessment_id = ?', (aid,))
            cursor.execute('DELETE FROM prediction_history WHERE assessment_id = ?', (aid,))
            
        cursor.execute('DELETE FROM clinical_assessments WHERE patient_id = ?', (patient_id,))
        cursor.execute('DELETE FROM patients WHERE patient_id = ?', (patient_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if deleted:
            self.log_audit('DELETE', 'patients', patient_id, 'Deleted patient and all associated records')
        return deleted
    
    # ===== CLINICAL ASSESSMENT OPERATIONS =====
    def add_assessment(self, patient_id: int, vitals: Dict, labs: Dict, 
                      clinical_notes: str = None, diagnosis: str = None) -> int:
        """Add a clinical assessment for a patient."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO clinical_assessments (
                patient_id, systolic_bp, diastolic_bp, heart_rate, respiratory_rate,
                spo2, body_temperature, wbc, serum_creatinine, hemoglobin, hematocrit,
                platelets, glucose, bun, sodium, clinical_notes, diagnosis
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id,
            vitals.get('systolic_bp'),
            vitals.get('diastolic_bp'),
            vitals.get('heart_rate'),
            vitals.get('respiratory_rate'),
            vitals.get('spo2'),
            vitals.get('body_temperature'),
            labs.get('wbc'),
            labs.get('serum_creatinine'),
            labs.get('hemoglobin'),
            labs.get('hematocrit'),
            labs.get('platelets'),
            labs.get('glucose'),
            labs.get('bun'),
            labs.get('sodium'),
            clinical_notes,
            diagnosis
        ))
        
        assessment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.log_audit('INSERT', 'clinical_assessments', assessment_id, 'Added new assessment')
        return assessment_id
    
    def get_assessment(self, assessment_id: int) -> Optional[Dict]:
        """Retrieve a specific assessment."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM clinical_assessments WHERE assessment_id = ?', (assessment_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_patient_assessments(self, patient_id: int) -> List[Dict]:
        """Get all assessments for a patient."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM clinical_assessments WHERE patient_id = ? ORDER BY assessment_date DESC',
            (patient_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def update_assessment_predictions(self, assessment_id: int, risk_score: float,
                                     risk_category: str, severity_level: str,
                                     confidence: float) -> bool:
        """Update assessment with model predictions."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE clinical_assessments 
            SET risk_score = ?, risk_category = ?, severity_level = ?, model_confidence = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE assessment_id = ?
        ''', (risk_score, risk_category, severity_level, confidence, assessment_id))
        
        conn.commit()
        conn.close()
        
        self.log_audit('UPDATE', 'clinical_assessments', assessment_id, 'Updated predictions')
        return cursor.rowcount > 0
    
    # ===== IMAGE OPERATIONS =====
    def add_image(self, assessment_id: int, filename: str, filepath: str, 
                 image_type: str = 'xray') -> int:
        """Add an image to an assessment."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO image_data (assessment_id, image_filename, image_path, image_type)
            VALUES (?, ?, ?, ?)
        ''', (assessment_id, filename, filepath, image_type))
        
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.log_audit('INSERT', 'image_data', image_id, f'Added image: {filename}')
        return image_id
    
    def get_assessment_images(self, assessment_id: int) -> List[Dict]:
        """Get all images for an assessment."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM image_data WHERE assessment_id = ? ORDER BY upload_date DESC',
            (assessment_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    # ===== TEXT NOTES OPERATIONS =====
    def add_text_note(self, assessment_id: int, note_type: str, content: str) -> int:
        """Add a text note to an assessment."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO text_notes (assessment_id, note_type, note_content)
            VALUES (?, ?, ?)
        ''', (assessment_id, note_type, content))
        
        note_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.log_audit('INSERT', 'text_notes', note_id, f'Added note: {note_type}')
        return note_id
    
    def get_assessment_notes(self, assessment_id: int) -> List[Dict]:
        """Get all notes for an assessment."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM text_notes WHERE assessment_id = ? ORDER BY created_at DESC',
            (assessment_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    # ===== PREDICTION HISTORY =====
    def add_prediction_record(self, assessment_id: int, model_version: str,
                             tabular_input: str, prediction_output: str,
                             execution_time_ms: float) -> int:
        """Record a model prediction."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO prediction_history 
            (assessment_id, model_version, tabular_input, prediction_output, execution_time_ms)
            VALUES (?, ?, ?, ?, ?)
        ''', (assessment_id, model_version, tabular_input, prediction_output, execution_time_ms))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def get_assessment_predictions(self, assessment_id: int) -> List[Dict]:
        """Get prediction history for an assessment."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM prediction_history WHERE assessment_id = ? ORDER BY created_at DESC',
            (assessment_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    # ===== AUDIT LOG =====
    def log_audit(self, action: str, table_name: str, record_id: int, 
                 user_action: str) -> None:
        """Log database operations for audit trail."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log (action, table_name, record_id, user_action)
            VALUES (?, ?, ?, ?)
        ''', (action, table_name, record_id, user_action))
        
        conn.commit()
        conn.close()
    
    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """Get audit log entries."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?',
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    # ===== STATISTICS & REPORTING =====
    def get_patient_count(self) -> int:
        """Get total number of patients."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as count FROM patients')
        count = cursor.fetchone()['count']
        conn.close()
        
        return count
    
    def get_assessment_count(self) -> int:
        """Get total number of assessments."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as count FROM clinical_assessments')
        count = cursor.fetchone()['count']
        conn.close()
        
        return count
    
    def get_risk_statistics(self) -> Dict:
        """Get statistics on risk categories."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT risk_category, COUNT(*) as count
            FROM clinical_assessments
            WHERE risk_category IS NOT NULL
            GROUP BY risk_category
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        return {row['risk_category']: row['count'] for row in rows}


# Initialize database on module import
if __name__ == '__main__':
    init_database()
    db = DatabaseManager()
    print(f"[OK] Database connection successful")
    print(f"[OK] Total patients: {db.get_patient_count()}")
    print(f"[OK] Total assessments: {db.get_assessment_count()}")
