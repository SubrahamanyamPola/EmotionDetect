```md
# EmotionX — Text & Audio Emotion Recognition (Offline Flask Demo)

EmotionX is a lightweight, **fully offline** Flask web app that predicts **7 emotions** from either:

- **Text** (type a sentence)
- **Audio** (`.wav` upload, or select a built-in sample clip)

It is meant as an educational / portfolio-style demonstration with bundled sample data and pre-trained models.

**Emotions supported:** `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`

---

## What’s inside

### Text emotion model
- **Approach:** TF-IDF (1–2 grams) + Logistic Regression
- **Output:** top emotion label + full probability distribution

### Audio emotion model
- **Approach:** handcrafted audio features (Librosa) + Random Forest classifier
- **Features extracted per clip (29 total):**
  - Zero crossing rate (mean)
  - Spectral centroid (mean)
  - Spectral rolloff (mean)
  - RMS energy (mean)
  - 13 MFCC means
  - 12 chroma means

> Note: This is a small demo dataset—accuracy numbers are not representative of production-grade performance.

---

## Project structure

```

EmotionX_Web_Final/
├─ app.py
├─ requirements.txt
├─ start_server.sh / start_server.bat
├─ build_exe.sh / build_exe.bat
├─ models/
│  ├─ text_model.joblib
│  ├─ audio_model.joblib
│  ├─ text_report.txt
│  └─ audio_report.txt
├─ data/
│  ├─ text_samples.csv
│  └─ audio_index.csv
├─ static/
│  ├─ css/styles.css
│  ├─ js/app.js
│  └─ samples/audio/*.wav
└─ templates/
├─ index.html
├─ docs.html
└─ metrics.html

````

---

## Quick start (recommended)

### 1) Prerequisites
- Python **3.10+** (3.11 works well)
- On Linux/macOS: build tools are usually already available
- On Windows: use PowerShell / CMD

### 2) Install & run

#### macOS / Linux
```bash
cd EmotionX_Web_Final
bash start_server.sh
````

#### Windows (CMD)

```bat
cd EmotionX_Web_Final
start_server.bat
```

The app will start at:

* [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Using the web UI

Open:

* **Home UI:** `/`

  * Text prediction: enter a sentence → **Predict Emotion**
  * Audio prediction: upload a `.wav` → **Predict Uploaded Audio**
  * Or click **Predict** next to a built-in sample clip

* **API Docs page:** `/docs`

* **Metrics page:** `/metrics`

---

## REST API

### Health

```bash
curl http://127.0.0.1:5000/api/health
```

### List built-in audio samples

```bash
curl http://127.0.0.1:5000/api/samples
```

### Predict from text

```bash
curl -X POST http://127.0.0.1:5000/predict_text \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I am so happy to see you!\"}"
```

**Response shape**

```json
{
  "label": "joy",
  "probabilities": {
    "joy": 0.81,
    "neutral": 0.12,
    "surprise": 0.04
  }
}
```

### Predict from audio (upload)

```bash
curl -X POST http://127.0.0.1:5000/predict_audio \
  -F file=@path/to/your_clip.wav
```

### Predict from a bundled audio sample (JSON)

```bash
curl -X POST http://127.0.0.1:5000/predict_audio \
  -H "Content-Type: application/json" \
  -d "{ \"sample_path\": \"joy_1.wav\" }"
```

### OpenAPI spec

* `/openapi.json`

---

## Metrics & evaluation page

Route: **`/metrics`**

What happens when you open it:

* Loads the bundled sample CSVs
* Creates a validation split
* Runs the model on the validation set
* Generates confusion matrices:

  * `static/metrics/text_cm.png`
  * `static/metrics/audio_cm.png`
* Renders a `classification_report` for both models

> Because the bundled datasets are small, the metrics can be unstable and vary based on the split.

---

## Important implementation details (how the app works)

### Model loading

At startup, `app.py` loads two Joblib bundles:

* `models/text_model.joblib` → a dict with:

  * `model`: scikit-learn `Pipeline` (`TfidfVectorizer` → `LogisticRegression`)
  * `labels`: list of label strings

* `models/audio_model.joblib` → a dict with:

  * `model`: a scikit-learn classifier with `predict_proba`
  * `labels`: list of label strings

Paths are resolved relative to `app.py` so the project can be run from the folder directly.

### Audio preprocessing

When you call `/predict_audio`:

1. Audio is read using `soundfile` from bytes (no temp files needed).
2. Stereo is converted to mono by averaging channels.
3. Resampled to 16 kHz for consistent features.
4. Librosa features are computed and concatenated.
5. The model returns per-class probabilities.

---

## Troubleshooting

### 1) “Audio model failed to load / incompatible dtype” (scikit-learn pickle mismatch)

If you see an error like:

* `node array from the pickle has an incompatible dtype`

This usually means the audio model (`audio_model.joblib`) was saved with a different scikit-learn version than the one you’re currently using.

Fix options:

* **Option A (recommended): pin versions** in a fresh venv (example)

  ```bash
  pip install "scikit-learn==1.3.2" "numpy==1.26.4" "joblib==1.3.2"
  pip install -r requirements.txt
  ```
* **Option B:** retrain and re-export the audio model using your current environment (see next section).

### 2) Librosa / NumPy compatibility warnings

`app.py` includes small compatibility patches for older code paths:

* `np.complex` and `np.float` fallbacks

If you still get Librosa-related errors:

* ensure you installed from `requirements.txt`
* consider upgrading pip:

  ```bash
  python -m pip install --upgrade pip
  ```

### 3) Upload issues

* Only `.wav` is supported by the UI (and recommended for API).
* Keep clips short (a few seconds) for faster processing.

---

## (Optional) Retraining / re-exporting the models

This repository includes **inference** + UI, but not full training scripts.
Below are minimal examples you can adapt.

### Rebuild the text model from `data/text_samples.csv`

Create `train_text.py` (example):

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/text_samples.csv")
X = df["text"].astype(str).tolist()
y = df["label"].astype(str).tolist()

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=300)),
])

pipe.fit(X, y)

bundle = {"model": pipe, "labels": sorted(df["label"].unique().tolist())}
joblib.dump(bundle, "models/text_model.joblib")
print("Saved models/text_model.joblib")
```

### Rebuild the audio model (outline)

You’d need to:

1. iterate files listed in `data/audio_index.csv`
2. extract the same 29-D feature vector used in `app.py`
3. train a classifier with `predict_proba` (e.g., RandomForestClassifier)
4. `joblib.dump({"model": model, "labels": label_list}, "models/audio_model.joblib")`

You can copy the feature extractor from `app.py` (`extract_audio_features`) to keep it consistent.

---

## Building an executable (PyInstaller)

Scripts included:

* `build_exe.sh`
* `build_exe.bat`

**Important:** a Flask app usually needs to ship **templates**, **static files**, and **models**.
A plain `--onefile app.py` build typically will **NOT** bundle these extra folders unless you add them explicitly.

### Recommended: build as a folder app (easier)

```bash
pyinstaller --noconfirm --onedir --name EmotionX app.py \
  --add-data "templates:templates" \
  --add-data "static:static" \
  --add-data "models:models" \
  --add-data "data:data"
```

Run:

* macOS/Linux: `./dist/EmotionX/EmotionX`
* Windows: `dist\\EmotionX\\EmotionX.exe`

If you really want `--onefile`, you’ll still need `--add-data` and extra setup for runtime paths.

---

## License / usage

This project is provided as a demo/learning artefact. If you plan to publish it, add a LICENSE file and document dataset provenance.

```
```
