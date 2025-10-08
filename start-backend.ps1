# Start Backend Server
Write-Host "🚀 Starting Document Intelligence Backend..." -ForegroundColor Green

# Activate virtual environment
& "./venv/Scripts/Activate.ps1"

# Install/update dependencies
Write-Host "📦 Installing/updating dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Start the backend server
Write-Host "🌐 Starting FastAPI server on http://localhost:8000" -ForegroundColor Green
Write-Host "📚 API Documentation available at http://localhost:8000/docs" -ForegroundColor Cyan
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
