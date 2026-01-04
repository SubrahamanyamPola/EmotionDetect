@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
pyinstaller --noconfirm --onefile --name EmotionX app.py
echo Binary built at dist\EmotionX.exe
