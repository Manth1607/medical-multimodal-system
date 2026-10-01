import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SEED = 42
    
    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY")
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY is not configured")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # 📊 Modality 1: 14 Tabular Clinical Points
    NUM_TABULAR_FEATURES = 14  
    
    # 🩻 Modality 2: Higher Resolution Imaging (224x224 standard medical size)
    IMAGE_SIZE = (1, 224, 224) 
    
    # 📝 Modality 3: Deep Text Clinical Dictionary
    VOCAB_SIZE = 30522         
    MAX_TEXT_LEN = 100         # Allows full length physician admitting notes
    
    # Model Architecture Specs
    EMBED_DIM = 128            # Increased embedding capacity for complex features
    NUM_CLASSES = 2            # 0: Discharge/Stable, 1: ICU Admission Required
    BATCH_SIZE = 16
    EPOCHS = 25
    LEARNING_RATE = 0.0005     # Initial learning rate (adapted via LR scheduler)
    WEIGHT_DECAY = 1e-5        # L2 regularization to discourage overfitting
    PATIENCE = 7               # Extended early stopping patience
    TRAIN_SAMPLES = 2000       # Increased training set size for enhanced feature learning
    VAL_SAMPLES = 1000         # Reliable validation set size

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "multimodal_model.pth"))
    # Tabular feature scaling (computed from synthetic training data)
    TAB_MEAN = [
        -0.025800000876188278,
        0.006599999964237213,
        0.01080000028014183,
        -0.08240000158548355,
        -0.09749999642372131,
        0.051500000059604645,
        0.0052999998442828655,
        0.07100000232458115,
        -0.03060000017285347,
        -0.0032999999821186066,
        0.0689999982714653,
        -0.017000000923871994,
        0.027000000700354576,
        0.04809999838471413,
    ]
    TAB_STD = [
        0.9491999745368958,
        1.0454000234603882,
        0.993399977684021,
        0.9581000208854675,
        0.9330000281333923,
        1.0032000541687012,
        0.9919000267982483,
        1.01419997215271,
        0.9610999822616577,
        0.991599977016449,
        1.035599946975708,
        1.0667999982833862,
        0.9914000034332275,
        0.9555000066757202,
    ]
    # Clinical-unit means/stds for common vitals/biomarkers to normalize real inputs
    CLINICAL_TAB_MEAN = [
        120.0,  # SBP
        80.0,   # DBP
        75.0,   # HR
        16.0,   # RR
        98.0,   # SpO2
        37.0,   # Temp
        7.5,    # WBC
        0.9,    # Serum creatinine
        4.0,    # CRP
        0.02,   # Troponin-I
        45.0,   # Age
        0.5,    # Gender (binary mean)
        24.5,   # BMI
        1.0,    # Charlson
    ]
    CLINICAL_TAB_STD = [
        15.0,  # SBP
        10.0,  # DBP
        12.0,  # HR
        4.0,   # RR
        1.5,   # SpO2
        0.7,   # Temp
        3.0,   # WBC
        0.3,   # Serum creatinine
        6.0,   # CRP
        0.05,  # Troponin-I
        18.0,  # Age
        0.5,   # Gender binary
        4.5,   # BMI
        2.0,   # Charlson
    ]
    # Clinical safety thresholds
    SEVERE_SPO2_THRESHOLD = 85
    SEVERE_TEMP_THRESHOLD = 39.5
    SEVERE_TROPONIN_I_THRESHOLD = 0.5
    SEVERE_CHARLSON_THRESHOLD = 7
    SEVERE_XRAY_ABNORMAL_COUNT = 2
