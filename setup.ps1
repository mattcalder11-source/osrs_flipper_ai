# ===========================
# OSRS Flipper AI Setup Script
# ===========================

Write-Host "🔧 Setting up OSRS Flipper AI environment..." -ForegroundColor Cyan

# --- Step 1: Check for Python ---
Write-Host "Checking Python installation..."
$pythonVersion = & python --version 2>$null
if (-not $pythonVersion) {
    Write-Host "❌ Python not found. Please install it from https://www.python.org/downloads/ and re-run this script." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Python detected: $pythonVersion"

# --- Step 2: Create virtual environment ---
if (-Not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
    if (-Not (Test-Path ".venv")) {
        Write-Host "❌ Failed to create .venv. Check Python installation and try again." -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Virtual environment created."
} else {
    Write-Host "ℹ️ Virtual environment already exists."
}

# --- Step 3: Activate the virtual environment ---
Write-Host "Activating virtual environment..."
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    & $venvPath
    Write-Host "✅ Virtual environment activated."
} else {
    Write-Host "❌ Could not find activation script: $venvPath" -ForegroundColor Red
    exit 1
}

# --- Step 4: Install dependencies ---
if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    Write-Host "✅ Dependencies installed."
} else {
    Write-Host "⚠️ No requirements.txt found. Skipping dependency install."
}

# --- Step 5: Confirm setup ---
Write-Host "`n🎉 Setup complete!"
Write-Host "You can now run: python src/ingest.py" -ForegroundColor Green
