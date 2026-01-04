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

