from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from scipy.signal import welch
import pickle

app = FastAPI(title="HAR Activity Classifier", version="1.0")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="HAR Activity Classifier", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a portfolio project; a real product would restrict this
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load trained model + feature column order ----------
with open('model/har_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

activity_labels = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
    4: 'SITTING', 5: 'STANDING', 6: 'LAYING'
}

# ---------- Feature functions (must match notebook exactly) ----------
def time_domain_features(window):
    centered = window - window.mean()
    return {
        'mean': np.mean(window),
        'std': np.std(window),
        'rms': np.sqrt(np.mean(window**2)),
        'min': np.min(window),
        'max': np.max(window),
        'zero_crossing_rate': ((centered[:-1] * centered[1:]) < 0).sum(),
    }

def freq_domain_features(window, fs=50):
    freqs, psd = welch(window, fs=fs, nperseg=len(window))
    dominant_freq = freqs[np.argmax(psd)]
    spectral_energy = np.sum(psd**2)
    psd_norm = psd / psd.sum()
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return {
        'dominant_freq': dominant_freq,
        'spectral_energy': spectral_energy,
        'spectral_entropy': spectral_entropy,
    }

def gravity_features(wx, wy, wz):
    return {
        'grav_mean_x': wx.mean(), 'grav_mean_y': wy.mean(), 'grav_mean_z': wz.mean(),
        'grav_std_x': wx.std(), 'grav_std_y': wy.std(), 'grav_std_z': wz.std(),
    }

def safe_corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]

def axis_correlation_features(wx, wy, wz):
    return {
        'corr_xy': safe_corr(wx, wy),
        'corr_xz': safe_corr(wx, wz),
        'corr_yz': safe_corr(wy, wz),
    }

# ---------- Request schema ----------
class SensorWindow(BaseModel):
    body_acc_x: list[float] = Field(..., min_length=128, max_length=128)
    body_acc_y: list[float] = Field(..., min_length=128, max_length=128)
    body_acc_z: list[float] = Field(..., min_length=128, max_length=128)
    body_gyro_x: list[float] = Field(..., min_length=128, max_length=128)
    body_gyro_y: list[float] = Field(..., min_length=128, max_length=128)
    body_gyro_z: list[float] = Field(..., min_length=128, max_length=128)
    total_acc_x: list[float] = Field(..., min_length=128, max_length=128)
    total_acc_y: list[float] = Field(..., min_length=128, max_length=128)
    total_acc_z: list[float] = Field(..., min_length=128, max_length=128)

def build_feature_row(data: SensorWindow) -> dict:
    row = {}
    body_channels = {
        'body_acc_x': np.array(data.body_acc_x), 'body_acc_y': np.array(data.body_acc_y),
        'body_acc_z': np.array(data.body_acc_z), 'body_gyro_x': np.array(data.body_gyro_x),
        'body_gyro_y': np.array(data.body_gyro_y), 'body_gyro_z': np.array(data.body_gyro_z),
    }
    for name, window in body_channels.items():
        for feat_name, value in time_domain_features(window).items():
            row[f'{name}_{feat_name}'] = value
        for feat_name, value in freq_domain_features(window).items():
            row[f'{name}_{feat_name}'] = value

    total_acc_x = np.array(data.total_acc_x)
    total_acc_y = np.array(data.total_acc_y)
    total_acc_z = np.array(data.total_acc_z)
    row.update(gravity_features(total_acc_x, total_acc_y, total_acc_z))
    row.update(axis_correlation_features(body_channels['body_acc_x'], body_channels['body_acc_y'], body_channels['body_acc_z']))
    return row

# ---------- Routes ----------
@app.post("/predict")
def predict(data: SensorWindow):
    try:
        row = build_feature_row(data)
        x = np.array([[row[col] for col in feature_columns]])
        pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0]
        return {
            "activity": activity_labels[int(pred)],
            "confidence": round(float(proba.max()), 4),
            "all_probabilities": {activity_labels[i+1]: round(float(p), 4) for i, p in enumerate(proba)}
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing expected feature: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest", "n_features": len(feature_columns)}