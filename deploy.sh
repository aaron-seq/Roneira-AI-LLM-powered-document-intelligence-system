#!/bin/bash
# ==============================================================================
# Free Deployment Script for Roneira AI Document Intelligence
# Supports Railway, Render, Fly.io, and local deployment
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Copy free environment config if .env doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.free" ]; then
            cp .env.free .env
            print_success "Copied .env.free to .env"
        else
            print_warning "No .env file found. Please create one from .env.free"
        fi
    fi
    
    # Create necessary directories
    mkdir -p uploads data logs
    print_success "Created necessary directories"
}

# Function for local deployment
deploy_local() {
    print_status "Starting local deployment..."
    
    # Check requirements
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    if ! command_exists docker; then
        print_error "Docker is required but not installed"
        exit 1
    fi
    
    # Install Python dependencies
    print_status "Installing Python dependencies..."
    pip3 install -r requirements-free.txt
    
    # Start services with Docker Compose
    print_status "Starting services with Docker Compose..."
    docker-compose -f docker-compose.free.yml up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Start the application
    print_status "Starting Roneira AI application..."
    python3 main_free.py &
    APP_PID=$!
    
    print_success "Application started successfully!"
    print_status "Access your application at: http://localhost:8000"
    print_status "Health check: http://localhost:8000/health"
    print_status "API docs: http://localhost:8000/docs"
    
    echo
    print_status "Press Ctrl+C to stop the application"
    
    # Wait for user interrupt
    trap 'kill $APP_PID; docker-compose -f docker-compose.free.yml down; print_success "Application stopped"; exit 0' INT
    wait $APP_PID
}

# Function for Railway deployment
deploy_railway() {
    print_status "Preparing Railway deployment..."
    
    if ! command_exists railway; then
        print_error "Railway CLI is not installed. Install it with: npm install -g @railway/cli"
        exit 1
    fi
    
    # Login check
    if ! railway whoami >/dev/null 2>&1; then
        print_status "Please login to Railway..."
        railway login
    fi
    
    # Deploy
    print_status "Deploying to Railway..."
    railway up
    
    print_success "Deployed to Railway successfully!"
    print_status "Check your Railway dashboard for the deployed URL"
}

# Function for Render deployment
deploy_render() {
    print_status "Preparing Render deployment..."
    
    if [ ! -f "render.yaml" ]; then
        print_error "render.yaml not found. Please ensure it exists."
        exit 1
    fi
    
    print_status "Render deployment configuration ready"
    print_status "Steps to deploy on Render:"
    echo "1. Go to https://render.com/"
    echo "2. Connect your GitHub repository"
    echo "3. Create a new service from this repository"
    echo "4. Render will automatically use the render.yaml configuration"
    echo "5. Set your environment variables (DEEPSEEK_API_KEY, SECRET_KEY) in the Render dashboard"
    
    print_success "Render configuration is ready!"
}

# Function for Fly.io deployment
deploy_fly() {
    print_status "Preparing Fly.io deployment..."
    
    if ! command_exists flyctl; then
        print_error "Fly CLI is not installed. Install it from: https://fly.io/docs/getting-started/installing-flyctl/"
        exit 1
    fi
    
    # Login check
    if ! flyctl auth whoami >/dev/null 2>&1; then
        print_status "Please login to Fly.io..."
        flyctl auth login
    fi
    
    # Set secrets
    print_status "Setting up secrets..."
    read -p "Enter your DeepSeek API key: " deepseek_key
    read -p "Enter your secret key (or press enter for auto-generated): " secret_key
    
    if [ -z "$secret_key" ]; then
        secret_key=$(openssl rand -hex 32)
    fi
    
    flyctl secrets set DEEPSEEK_API_KEY="$deepseek_key" SECRET_KEY="$secret_key"
    
    # Create volume for data persistence
    flyctl volumes create roneira_data --size 1
    
    # Deploy
    print_status "Deploying to Fly.io..."
    flyctl deploy
    
    print_success "Deployed to Fly.io successfully!"
    flyctl status
}

# Function to show help
show_help() {
    echo "Roneira AI Free Deployment Script"
    echo
    echo "Usage: $0 [OPTION]"
    echo
    echo "Options:"
    echo "  local     Deploy locally with Docker Compose"
    echo "  railway   Deploy to Railway (free tier)"
    echo "  render    Prepare for Render deployment"
    echo "  fly       Deploy to Fly.io (free tier)"
    echo "  help      Show this help message"
    echo
    echo "Examples:"
    echo "  $0 local     # Start local development server"
    echo "  $0 railway  # Deploy to Railway"
    echo "  $0 render   # Show Render deployment instructions"
    echo "  $0 fly      # Deploy to Fly.io"
}

# Main execution
main() {
    print_status "Roneira AI Free Deployment Script"
    print_status "=====================================\n"
    
    # Setup environment
    setup_environment
    
    case "${1:-help}" in
        "local")
            deploy_local
            ;;
        "railway")
            deploy_railway
            ;;
        "render")
            deploy_render
            ;;
        "fly")
            deploy_fly
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@"