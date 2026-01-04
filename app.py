import os, io, json, numpy as np
from flask import Flask, request, jsonify, render_template
import joblib, librosa

APP_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(APP_DIR, "models")
STATIC_DIR  = os.path.join(APP_DIR, "static")
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(os.path.join(STATIC_DIR,'metrics'), exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=os.path.join(APP_DIR,"templates"))

# Load models
text_bundle = joblib.load(os.path.join(MODELS_DIR, "text_model.joblib"))
text_model = text_bundle["model"]
text_labels = text_bundle["labels"]

audio_bundle = joblib.load(os.path.join(MODELS_DIR, "audio_model.joblib"))
audio_model = audio_bundle["model"]
audio_labels = audio_bundle["labels"]

def proba_text(sent):
    probs = text_model.predict_proba([sent])[0]
    return {text_labels[i]: float(probs[i]) for i in range(len(text_labels))}

def extract_audio_features(buf, target_sr=16000):
    import soundfile as sf
    if not hasattr(np, "complex"): np.complex = np.complex128  # compat
    if not hasattr(np, "float"): np.float = float
    y, sr = sf.read(io.BytesIO(buf))
    if hasattr(y, "ndim") and y.ndim > 1: y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    cent = librosa.feature.spectral_centroid(y, sr=sr).mean()
    roll = librosa.feature.spectral_rolloff(y, sr=sr).mean()
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y, sr=sr).mean(axis=1)
    rms = librosa.feature.rms(y).mean()
    feats = np.hstack([zcr, cent, roll, rms, mfcc, chroma])
    return feats.reshape(1,-1)

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/predict_text")
def predict_text():
    try:
        data = request.get_json(force=True)
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error":"Text is empty"}), 400
        probs = proba_text(text)
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        return jsonify({"label": label, "probabilities": probs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/predict_audio")
def predict_audio():
    try:
        if request.is_json and request.json.get("sample_path"):
            sample_path = request.json["sample_path"]
            fpath = os.path.join(STATIC_DIR, "samples", "audio", os.path.basename(sample_path))
            if not os.path.exists(fpath):
                return jsonify({"error":"Sample not found"}), 404
            with open(fpath, "rb") as f:
                buf = f.read()
        else:
            if "file" not in request.files:
                return jsonify({"error":"No file in form-data as 'file'"}), 400
            file = request.files["file"]
            buf = file.read()

        feats = extract_audio_features(buf)
        probs = audio_model.predict_proba(feats)[0]
        probs = {audio_labels[i]: float(probs[i]) for i in range(len(audio_labels))}
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        return jsonify({"label": label, "probabilities": probs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/api/samples")
def api_samples():
    import pandas as pd
    idx_path = os.path.join(DATA_DIR, "audio_index.csv")
    df = pd.read_csv(idx_path)
    samples = []
    for _, r in df.iterrows():
        fname = r["file"]
        samples.append({"file": fname, "label": r["label"], "url": f"/static/samples/audio/{fname}"})
    return jsonify({"samples": samples})

@app.get("/api/health")
def health():
    return jsonify({"status":"ok"})

@app.get("/openapi.json")
def openapi():
    spec = {
      "openapi":"3.0.0",
      "info":{"title":"EmotionX API","version":"1.0.0"},
      "paths":{
        "/predict_text":{"post":{"requestBody":{"required":True,"content":{"application/json":{"schema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}}},"responses":{"200":{"description":"OK"}}}},
        "/predict_audio":{"post":{"requestBody":{"required":True,"content":{"multipart/form-data":{"schema":{"type":"object","properties":{"file":{"type":"string","format":"binary"}}}},"application/json":{"schema":{"type":"object","properties":{"sample_path":{"type":"string"}}}}}},"responses":{"200":{"description":"OK"}}}},
        "/api/samples":{"get":{"responses":{"200":{"description":"OK"}}}},
        "/api/health":{"get":{"responses":{"200":{"description":"OK"}}}}
      }
    }
    return jsonify(spec)

@app.get("/docs")
def docs():
    return render_template("docs.html")

# -------- Metrics (Confusion Matrices) --------
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def _save_cm_png(y_true, y_pred, labels, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(6,5), dpi=140)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center', color='black', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

@app.get("/metrics")
def metrics():
    # TEXT
    text_csv = os.path.join(DATA_DIR, "text_samples.csv")
    df_t = pd.read_csv(text_csv)
    from sklearn.preprocessing import LabelEncoder
    le_t = LabelEncoder().fit(df_t["label"])
    X = df_t["text"].tolist()
    y = le_t.transform(df_t["label"].tolist())
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred_t = text_model.predict(X_va)
    labels_t = list(le_t.classes_)
    text_cm_path = os.path.join(STATIC_DIR, "metrics", "text_cm.png")
    _save_cm_png(y_va, y_pred_t, labels_t, text_cm_path, "Text Confusion Matrix")
    text_report = classification_report(y_va, y_pred_t, target_names=labels_t)

    # AUDIO
    audio_idx = pd.read_csv(os.path.join(DATA_DIR, "audio_index.csv"))
    feats, labs = [], []
    import soundfile as sf
    for _, r in audio_idx.iterrows():
        fpath = os.path.join(STATIC_DIR, "samples", "audio", r["file"])
        with open(fpath,"rb") as f: buf = f.read()
        v = extract_audio_features(buf)
        feats.append(v[0]); labs.append(r["label"])
    feats = np.vstack(feats)
    from sklearn.preprocessing import LabelEncoder
    le_a = LabelEncoder().fit(labs)
    y_all = le_a.transform(labs)
    X_tr, X_va, y_tr, y_va = train_test_split(feats, y_all, test_size=7, random_state=42, stratify=y_all)
    y_pred_a = audio_model.predict(X_va)
    labels_a = list(le_a.classes_)
    audio_cm_path = os.path.join(STATIC_DIR, "metrics", "audio_cm.png")
    _save_cm_png(y_va, y_pred_a, labels_a, audio_cm_path, "Audio Confusion Matrix")
    audio_report = classification_report(y_va, y_pred_a, target_names=labels_a)

    return render_template("metrics.html", text_report=text_report, audio_report=audio_report)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
