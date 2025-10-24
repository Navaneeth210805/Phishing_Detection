@echo off
echo 🚀 Setting up Enhanced PyTorch Phishing Detection Environment
echo ============================================================

echo.
echo 📁 Current Directory:
cd

echo.
echo 🔍 Checking Virtual Environment...
if not exist ".venv" (
    echo ❌ Virtual environment not found!
    echo 💡 Creating new virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created successfully
)

echo.
echo 🌟 Activating Virtual Environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated

echo.
echo 📦 Installing/Updating Dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo 🧪 Testing PyTorch Installation...
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed successfully'); print(f'🔥 CUDA available: {torch.cuda.is_available()}')"

echo.
echo 🎯 Running Enhanced PyTorch Model...
python enhanced_pytorch_model.py

echo.
echo 🎉 Setup and execution complete!
pause
