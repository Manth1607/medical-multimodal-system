import sqlite3
from datetime import datetime, timedelta
import requests  # Added for live city location lookup


def get_city():
    """Return approximate city based on public IP via ipinfo.io.
    Returns 'Unknown' if the request fails or the city field is missing.
    """
    try:
        resp = requests.get('https://ipinfo.io/json', timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('city', 'Unknown')
    except Exception:
        pass
    return 'Unknown'

import os
db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'clinical_data.db')
db = sqlite3.connect(db_path)
db.row_factory = sqlite3.Row
c = db.cursor()

print('='*80)
print('PATIENT DATA ANALYSIS - LAST 100 YEARS TO TODAY')
print('='*80)

today = datetime.now()
hundred_yrs_ago = today - timedelta(days=365*100)
# Determine user's approximate city based on IP
city = get_city()

print(f'\nDate Range Analysis:')
print(f'  Approximate City: {city}')
print(f'  Start Date (100 years ago): {hundred_yrs_ago.strftime("%Y-%m-%d")}')
print(f'  End Date (Today):           {today.strftime("%Y-%m-%d")}')
print(f'  Analysis Period:            {hundred_yrs_ago.strftime("%Y")} to {today.strftime("%Y")} ({100} years)')

# Patient count
c.execute('SELECT COUNT(*) as cnt FROM patients')
patient_count = c.fetchone()['cnt']
print(f'\n{"-"*80}')
print(f'PATIENT SUMMARY')
print(f'{"-"*80}')
print(f'Total Patients: {patient_count}')

# Show patients
c.execute('SELECT patient_id, name, age, gender, date_of_birth, created_at FROM patients')
patients = c.fetchall()

if patients:
    print(f'\n{"ID":<5} {"Name":<20} {"Age":<6} {"Gender":<10} {"DOB":<15} {"Registered":<20}')
    print('-'*80)
    for p in patients:
        dob = p['date_of_birth'][:10] if p['date_of_birth'] else 'N/A'
        reg = p['created_at'][:10] if p['created_at'] else 'N/A'
        print(f'{p[0]:<5} {p[1]:<20} {str(p[2]):<6} {p[3]:<10} {dob:<15} {reg:<20}')

# Gender ratio
print(f'\n{"-"*80}')
print(f'GENDER DISTRIBUTION')
print(f'{"-"*80}')

c.execute('SELECT gender, COUNT(*) as cnt FROM patients GROUP BY gender')
genders = c.fetchall()

total_gender = sum(g['cnt'] for g in genders)
for g in genders:
    pct = (g['cnt']/total_gender*100) if total_gender > 0 else 0
    print(f'  {g[0]:<15}: {g[1]} patients ({pct:.1f}%)')

if len(genders) == 2:
    male = next((g['cnt'] for g in genders if g[0] == 'Male'), 0)
    female = next((g['cnt'] for g in genders if g[0] == 'Female'), 0)
    if male > 0 and female > 0:
        ratio = male / female
        print(f'\n  Male to Female Ratio: {ratio:.2f}:1')

# Age stats
print(f'\n{"-"*80}')
print(f'AGE STATISTICS')
print(f'{"-"*80}')

c.execute('SELECT AVG(age) as avg_age, MIN(age) as min_age, MAX(age) as max_age FROM patients WHERE age IS NOT NULL')
stats = c.fetchone()

if stats['avg_age']:
    print(f'  Average Age:      {float(stats["avg_age"]):.1f} years')
    print(f'  Minimum Age:      {stats["min_age"]} years')
    print(f'  Maximum Age:          {stats["max_age"]} years')
    print(f'  Age Range:        {stats["max_age"] - stats["min_age"]} years')

# Assessments
print(f'\n{"-"*80}')
print(f'ASSESSMENT SUMMARY')
print(f'{"-"*80}')

c.execute('SELECT COUNT(*) as cnt FROM clinical_assessments')
assess_count = c.fetchone()['cnt']
print(f'Total Assessments: {assess_count}')

if patient_count > 0:
    avg_assessments = assess_count / patient_count
    print(f'Avg Assessments per Patient: {avg_assessments:.1f}')

# Risk distribution
print(f'\n{"-"*80}')
print(f'RISK CATEGORY DISTRIBUTION')
print(f'{"-"*80}')

c.execute('SELECT risk_category, COUNT(*) as cnt FROM clinical_assessments WHERE risk_category IS NOT NULL GROUP BY risk_category')
risks = c.fetchall()

if risks:
    total_risks = sum(r['cnt'] for r in risks)
    for r in risks:
        pct = (r['cnt']/total_risks*100) if total_risks > 0 else 0
        print(f'  {r[0]:<20}: {r[1]} assessments ({pct:.1f}%)')

# Severity distribution
print(f'\n{"-"*80}')
print(f'SEVERITY LEVEL DISTRIBUTION')
print(f'{"-"*80}')

c.execute('SELECT severity_level, COUNT(*) as cnt FROM clinical_assessments WHERE severity_level IS NOT NULL GROUP BY severity_level')
severities = c.fetchall()

if severities:
    total_sev = sum(s['cnt'] for s in severities)
    for s in severities:
        pct = (s['cnt']/total_sev*100) if total_sev > 0 else 0
        print(f'  {s[0]:<20}: {s[1]} assessments ({pct:.1f}%)')

db.close()
print('\n' + '='*80)
print('END OF PATIENT DATA ANALYSIS REPORT')
print('='*80 + '\n')
