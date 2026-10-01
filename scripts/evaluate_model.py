import os
import sys

# Set project root in path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from core.config import Config
from ml.models import ClinicalMultimodalFusionClassifier
from ml.data_pipeline import get_data_loaders

def evaluate_model():
    print("=" * 60)
    print("       MEDICAL MULTIMODAL MODEL - ACCURACY EVALUATION")
    print("=" * 60)
    
    # 1. Check if model weights exist
    if not os.path.exists(Config.MODEL_PATH):
        print(f"[ERROR] Model file not found at: {Config.MODEL_PATH}")
        print("Please train the model first by running: python -m ml.train")
        return

    # 2. Load model
    print(f"Loading trained weights from: {Config.MODEL_PATH}")
    model = ClinicalMultimodalFusionClassifier()
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # 3. Load Validation / Test Dataset
    _, val_loader = get_data_loaders()
    print(f"Evaluating on Validation Dataset ({len(val_loader.dataset)} samples)...\n")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            tabular = batch['tabular']
            image = batch['image']
            text = batch['text']
            labels = batch['label']

            outputs = model(tabular, image, text)
            probs = torch.softmax(outputs, dim=1)
            # Increase threshold from 0.5 to 0.65 to reduce False Positives (predicting Critical wrongly)
            preds = (probs[:, 1] > 0.65).long()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 4. Calculate Metrics
    acc = accuracy_score(all_targets, all_preds) * 100
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100
    cm = confusion_matrix(all_targets, all_preds)

    correct_count = (all_preds == all_targets).sum()
    total_count = len(all_targets)

    # 5. Display Results
    print("+" + "-" * 58 + "+")
    print(f"|  Overall Accuracy   : {acc:6.2f}% ({correct_count}/{total_count} correct)            |")
    print(f"|  Precision (Weighted): {prec:6.2f}%                                |")
    print(f"|  Recall (Weighted)   : {rec:6.2f}%                                |")
    print(f"|  F1-Score (Weighted) : {f1:6.2f}%                                |")
    print("+" + "-" * 58 + "+")
    print()
    print("Confusion Matrix:")
    print(f"                Predicted Stable (0)    Predicted ICU/Critical (1)")
    print(f"Actual Stable (0)        {cm[0][0]:<20}    {cm[0][1]:<20}")
    if len(cm) > 1:
        print(f"Actual ICU (1)           {cm[1][0]:<20}    {cm[1][1]:<20}")
    print()
    print("Detailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=["Class 0 (Stable)", "Class 1 (Critical)"], zero_division=0))

if __name__ == "__main__":
    evaluate_model()
