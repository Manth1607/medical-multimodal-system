# Medical Multimodal Diagnostics & ICU Triage System

## Overview
This system is an AI-driven multimodal clinical triage tool designed to assist healthcare professionals in predicting patient risk and ICU admission requirements. It combines tabular patient data (vitals, labs), medical imaging (e.g., chest X-rays), and unstructured clinical text (notes) into a unified predictive model.

## Features
- **Multimodal AI**: Leverages PyTorch to process Tabular (MLP), Image (CNN), and Text (BiLSTM) modalities.
- **FastAPI Backend**: A robust backend providing a secure `/predict` endpoint for inference.
- **Streamlit Dashboard**: An interactive UI for data entry, OCR (via Tesseract), patient history viewing, and live AI predictions.
- **Clinical Safety Overrides**: Hardcoded clinical rules to catch critical conditions (e.g., severe SpO2 drop, high troponin) that override AI predictions for safety.
- **Explainability**: Saliency maps for images and feature importance for tabular inputs.
- **Audit & History**: SQLite database logging patient info, assessments, and model predictions.

## Architecture & Project Structure
- `api/`: FastAPI server for REST endpoints.
- `core/`: Configuration values and clinical reference ranges.
- `ml/`: PyTorch models, data pipeline, and prediction service.
- `database/`: SQLite database setup and operations manager.
- `dashboard/`: Streamlit interactive web interface.
- `utils/`: Helpers for OCR and patient data analysis.
- `tests/`: Testing scripts for API, DB, and JWT auth.
- `scripts/`: unified runners like `run_all.py` and `run_unified_app.py`.

## Quickstart Guide

### Prerequisites
1. Python 3.9+
2. Tesseract OCR installed on your system (Required for Streamlit Dashboard text extraction).

### Installation
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Update `core/config.py` if necessary (e.g. `SECRET_KEY`).

### Execution
Run the end-to-end setup and validation script:
```bash
python scripts/run_all.py
```
This script will:
- Verify configurations and imports.
- Train the multimodal PyTorch model.
- Initialize the SQLite database and run DB tests.
- Spin up the FastAPI server in the background.
- Test the `/predict` endpoint.

To run the live dashboard:
```bash
streamlit run dashboard/app.py
```
Or start both FastAPI and Streamlit together:
```bash
python scripts/run_unified_app.py
```

### Docker Deployment
1. Install Docker Engine/Desktop with the Docker Compose plugin.
2. Create a `.env` file from `.env.example` and set a unique `SECRET_KEY` and strong `INITIAL_ADMIN_PASSWORD`. Add API/email credentials only when those features are needed.
3. Build and start the API and dashboard:
   ```bash
   docker compose up --build -d
   ```
4. Open the dashboard at `http://localhost:8501`; the API is available at `http://localhost:8000`.

The SQLite database is stored in the persistent `clinical_data` Docker volume and shared by both services. Stop the services with `docker compose down`; the database volume is retained. To remove the database as well, use `docker compose down -v`.

## Security
- Authentication is handled via JWT (JSON Web Tokens).
- Passwords are encrypted using `bcrypt`.
- SQL queries use parameterized queries to prevent SQL injection.

## License
Confidential & Proprietary. Do not distribute.
