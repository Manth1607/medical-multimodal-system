import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from core.config import Config

class ExpandedMedicalDataset(Dataset):
    def __init__(self, num_samples=100, split='train'):
        self.num_samples = num_samples
        self.split = split
        rng = np.random.RandomState(Config.SEED + (0 if split == 'train' else 1))

        # Simulating 14 explicit clinical tabular vectors
        self.tabular_data = rng.randn(num_samples, Config.NUM_TABULAR_FEATURES).astype(np.float32)

        # Medically accurate synthetic signal using all 14 features:
        tabular_signal = (
            0.15 * self.tabular_data[:, 0]   # SBP
            + 0.10 * self.tabular_data[:, 1] # DBP
            + 0.20 * self.tabular_data[:, 2] # HR
            + 0.20 * self.tabular_data[:, 3] # RR
            - 0.40 * self.tabular_data[:, 4] # SpO2 (low is bad)
            + 0.15 * self.tabular_data[:, 5] # Temp
            + 0.20 * self.tabular_data[:, 6] # WBC
            + 0.25 * self.tabular_data[:, 7] # Creatinine
            + 0.20 * self.tabular_data[:, 8] # CRP
            + 0.30 * self.tabular_data[:, 9] # Troponin
            + 0.15 * self.tabular_data[:, 10] # Age
            + 0.10 * self.tabular_data[:, 12] # BMI
            + 0.25 * self.tabular_data[:, 13] # Charlson
        )

        # Simulating 224x224 medical images (e.g., Chest X-Rays) with class-specific patch patterns
        self.image_data = rng.randn(num_samples, *Config.IMAGE_SIZE).astype(np.float32)
        
        # Simulating unstructured clinical text: create simple synthetic sentence strings
        vocab_words = [f"word{i}" for i in range(20000)]
        self.text_str = []
        for i in range(num_samples):
            # create a variable-length sentence of words
            length = rng.randint(5, Config.MAX_TEXT_LEN)
            words = rng.choice(vocab_words, size=length)
            self.text_str.append(' '.join(words))

        # If a tokenizer was saved to disk, load it and encode text to token ids
        tokenizer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer')
        if os.path.exists(tokenizer_dir):
            try:
                from transformers import AutoTokenizer
                tk = AutoTokenizer.from_pretrained(tokenizer_dir)
                enc = tk(self.text_str, padding='max_length', truncation=True, max_length=Config.MAX_TEXT_LEN)
                # Map tokenizer ids into the model's vocab range to keep compatibility
                mapped = [[int(x) % Config.VOCAB_SIZE for x in seq] for seq in enc['input_ids']]
                self.text_data = np.array(mapped, dtype=np.int64)
            except Exception:
                # fallback to random ids if tokenizer load fails
                self.text_data = rng.randint(0, Config.VOCAB_SIZE, size=(num_samples, Config.MAX_TEXT_LEN)).astype(np.int64)
        else:
            self.text_data = rng.randint(0, Config.VOCAB_SIZE, size=(num_samples, Config.MAX_TEXT_LEN)).astype(np.int64)

        # Realistic clinical multimodal fusion with natural biological variation
        # 1. Subtle imaging variation (opacities, texture variance instead of artificial blocks)
        image_signal = rng.randn(num_samples).astype(np.float32) * 0.3
        for idx in range(num_samples):
            # Subtle anatomical texture with realistic clinical variability
            if tabular_signal[idx] + rng.normal(0, 0.5) > 0:
                patch_mag = 0.5
                self.image_data[idx, 0, 80:144, 80:144] += patch_mag + rng.randn(64, 64).astype(np.float32) * 0.25
                image_signal[idx] += 0.4
            else:
                self.image_data[idx, 0, 80:144, 80:144] += rng.randn(64, 64).astype(np.float32) * 0.25

        # 2. Text tokens correlation with subtle clinical lexicon (no artificial hard-split)
        text_signal = rng.randn(num_samples).astype(np.float32) * 0.3
        for idx in range(num_samples):
            if tabular_signal[idx] + rng.normal(0, 0.5) > 0:
                crit_tokens = rng.choice([1000, 2500, 3200, 4800, 7500], size=5)
                self.text_data[idx, :5] = crit_tokens
                text_signal[idx] += 0.35
            else:
                stable_tokens = rng.choice([150, 420, 890, 1200, 2100], size=5)
                self.text_data[idx, :5] = stable_tokens
                text_signal[idx] -= 0.25

        # 3. Combined clinical risk score with unmeasured medical variance (noise)
        clinical_risk = tabular_signal + image_signal + text_signal + rng.normal(scale=0.9, size=num_samples).astype(np.float32)
        self.labels = (clinical_risk > np.median(clinical_risk)).astype(np.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tab = torch.tensor(self.tabular_data[idx])
        img = torch.tensor(self.image_data[idx])
        txt = torch.tensor(self.text_data[idx])
        lbl = torch.tensor(self.labels[idx])

        # Data Augmentation during training for better generalization
        if self.split == 'train':
            # 50% chance horizontal flip for medical image invariance
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[2])
            # Subtle vital sign measurement jitter
            tab = tab + torch.randn_like(tab) * 0.02

        return {
            'tabular': tab,
            'image': img,
            'text': txt,
            'label': lbl
        }

def get_data_loaders():
    train_set = ExpandedMedicalDataset(num_samples=Config.TRAIN_SAMPLES, split='train')
    val_set = ExpandedMedicalDataset(num_samples=Config.VAL_SAMPLES, split='val')
    return DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True), DataLoader(val_set, batch_size=Config.BATCH_SIZE, shuffle=False)
