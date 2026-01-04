#!/usr/bin/env bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pyinstaller --noconfirm --onefile --name EmotionX app.py
echo "Binary (dist/EmotionX) built. Run: ./dist/EmotionX"
