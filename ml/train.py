import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from core.config import Config
from ml.data_pipeline import get_data_loaders
from ml.models import ClinicalMultimodalFusionClassifier
import os

# Download and save tokenizer for consistent text preprocessing
tokenizer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer')
if not os.path.exists(tokenizer_dir):
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        tk.save_pretrained(tokenizer_dir)
        print(f"Saved tokenizer to {tokenizer_dir}")
    except Exception as e:
        print('Warning: unable to download/save tokenizer:', e)

def train_system():
    train_loader, val_loader = get_data_loaders()
    model = ClinicalMultimodalFusionClassifier()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    print("Beginning 17-Week Model Validation Iterations...")

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss, correct = 0, 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch['tabular'], batch['image'], batch['text'])
            loss = criterion(outputs, batch['label'])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch['label']).sum().item()
            
        train_acc = (correct / len(train_loader.dataset)) * 100
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['tabular'], batch['image'], batch['text'])
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == batch['label']).sum().item()
                val_total += batch['label'].size(0)

        val_acc = (val_correct / val_total) * 100
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch+1}/{Config.EPOCHS}] - Loss: {total_loss/len(train_loader):.4f} "
            f"- Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}% - LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best validation accuracy. Saved model to: {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"  No improvement on validation accuracy (patience {patience_counter}/{Config.PATIENCE}).")
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered. Stopping training.")
                break
        
    if best_val_acc == 0.0:
        torch.save(model.state_dict(), Config.MODEL_PATH)
        print(f"Model weights saved to: {Config.MODEL_PATH}")
    else:
        print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_system()
