#!/bin/bash
set -e

echo "🚀 Setting up Document Intelligence System..."

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
echo "📦 Installing backend dependencies..."
pip install -r requirements.txt

# Install frontend dependencies
echo "🎨 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create environment file
if [ ! -f .env ]; then
    echo "📄 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please update .env with your API keys"
fi

# Create necessary directories
mkdir -p uploads processed logs

echo "✅ Setup complete! Next steps:"
echo "1. Update .env file with your API keys"
echo "2. Run: python -m uvicorn backend.main:app --reload"
echo "3. In another terminal: cd frontend && npm run dev"
