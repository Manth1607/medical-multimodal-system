"""
FastAPI server for the Medical Multimodal System.
Provides:
- GET /              -> serves dashboard HTML (templates/dashboard.html)
- POST /predict      -> runs inference via prediction_service.run_prediction
Static assets (if any) are served under /static.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import uvicorn
import jwt
from datetime import datetime, timedelta, timezone
import bcrypt
import os

# Updated Imports for the new folder structure
from ml.prediction_service import run_prediction
from database.database import DatabaseManager
from core.config import Config

app = FastAPI()
db = DatabaseManager()
templates = Jinja2Templates(directory="templates")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic Models for request validation
class PatientInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    optional_labs: Optional[Dict[str, float]] = None
    infiltrates: bool = False
    cardiomegaly: bool = False
    pleural_effusion: bool = False
    consolidation: bool = False
    lung_nodule: bool = False

class PredictionRequest(BaseModel):
    patient_info: Optional[PatientInfo] = None
    tabular: List[float]
    image: List[List[float]]
    text: str

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = db.get_user_by_username(username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db.get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Serve static files (if present) under /static
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the dashboard HTML at the root URL
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")

# Prediction endpoint
@app.post("/predict")
def predict(payload: PredictionRequest, current_user: dict = Depends(get_current_user)):
    result = run_prediction(payload.dict())
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message", "Inference error"))
    return result

class EmailRequest(BaseModel):
    to_email: str
    patient_name: str
    payload: dict
    result: dict

@app.post("/email-report")
def email_report(req: EmailRequest, current_user: dict = Depends(get_current_user)):
    from utils.pdf_generator import generate_pdf_report
    from utils.email_service import send_medical_report_email
    
    pdf_bytes = generate_pdf_report(req.payload, req.result)
    success, err_msg = send_medical_report_email(req.to_email, req.patient_name, pdf_bytes)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {err_msg}")
    return {"status": "success", "message": "Email sent successfully"}

@app.get("/verify-token")
def verify_token(current_user: dict = Depends(get_current_user)):
    return {"status": "valid", "username": current_user["username"]}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=os.getenv("RELOAD", "false").lower() == "true")
