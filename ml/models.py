import torch
import torch.nn as nn
from core.config import Config

class TabularDeepEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.NUM_TABULAR_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, Config.EMBED_DIM),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class MedicalCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 224x224
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 112x112
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 56x56
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                # Downsample to fixed 4x4 shape
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, Config.EMBED_DIM),
            nn.ReLU()
        )
    def forward(self, x):
        return self.features(x)

class ClinicalTextBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.VOCAB_SIZE, 64)
        self.lstm = nn.LSTM(64, Config.EMBED_DIM // 2, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(self.embedding(x))
        # Concatenate forward and backward final states to match EMBED_DIM
        return torch.cat((hidden[-2], hidden[-1]), dim=1)

class MultimodalAttentionFusion(nn.Module):
    """
    Dynamically computes importance weights across Tabular, Image, and Text modalities
    per patient, rather than naive fixed concatenation.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, tab, img, txt):
        concat = torch.cat((tab, img, txt), dim=1)
        weights = self.attn_net(concat)
        
        w_tab = weights[:, 0:1]
        w_img = weights[:, 1:2]
        w_txt = weights[:, 2:3]
        
        fused = torch.cat((tab * w_tab, img * w_img, txt * w_txt), dim=1)
        return fused

class ClinicalMultimodalFusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.tab_enc = TabularDeepEncoder()
        self.img_enc = MedicalCNNEncoder()
        self.txt_enc = ClinicalTextBiLSTM()
        self.attention_fusion = MultimodalAttentionFusion(Config.EMBED_DIM)
        
        self.classifier = nn.Sequential(
            nn.Linear(Config.EMBED_DIM * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, Config.NUM_CLASSES)
        )

    def forward(self, tab, img, txt):
        t_feat = self.tab_enc(tab)
        i_feat = self.img_enc(img)
        x_feat = self.txt_enc(txt)
        fused = self.attention_fusion(t_feat, i_feat, x_feat)
        return self.classifier(fused)
