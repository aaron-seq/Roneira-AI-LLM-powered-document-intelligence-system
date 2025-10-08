# Start Frontend Development Server
Write-Host "🎨 Starting Document Intelligence Frontend..." -ForegroundColor Green

# Navigate to frontend directory
Set-Location frontend

# Install/update dependencies
Write-Host "📦 Installing/updating dependencies..." -ForegroundColor Yellow
npm install

# Start the development server
Write-Host "🌐 Starting React development server on http://localhost:3000" -ForegroundColor Green
npm run dev

# Return to root directory
Set-Location ..
