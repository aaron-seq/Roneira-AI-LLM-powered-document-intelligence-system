# ==============================================================================
# Makefile for the LLM Document Intelligence System
#
# This Makefile provides a convenient set of commands for common development
# and operational tasks. It helps streamline the workflow for developers.
#
# To see a list of all available commands, run: make help
# ==============================================================================

# --- Variables ---
# Defines common variables used in the commands.
PYTHON := python3
PIP := pip
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := llm-document-intelligence

# --- Help Command ---
# The default command, which displays a helpful list of all available commands.
.DEFAULT_GOAL := help
help:
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ------------------------------------------------------------------------------
# Installation and Setup
# ------------------------------------------------------------------------------

install-backend: ## Install backend Python dependencies
	$(PIP) install -r requirements.txt

install-frontend: ## Install frontend Node.js dependencies
	cd frontend && npm install

setup: install-backend install-frontend ## Run the complete setup for the project
	@echo "Project setup complete."

# ------------------------------------------------------------------------------
# Development
# ------------------------------------------------------------------------------

dev: ## Start the full development environment using Docker Compose
	$(DOCKER_COMPOSE) up --build

stop-dev: ## Stop the development environment
	$(DOCKER_COMPOSE) down

logs: ## View the logs from all running services
	$(DOCKER_COMPOSE) logs -f

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------

test-backend: ## Run backend tests
	$(PYTHON) -m pytest tests/

test-frontend: ## Run frontend tests
	cd frontend && npm test

test: test-backend test-frontend ## Run all tests

# ------------------------------------------------------------------------------
# Code Quality
# ------------------------------------------------------------------------------

lint-backend: ## Lint the backend Python code
	flake8 .

format-backend: ## Format the backend Python code
	black .

lint-frontend: ## Lint the frontend TypeScript/React code
	npm run lint

format-frontend: ## Format the frontend TypeScript/React code
	npm run format

quality-check: lint-backend lint-frontend ## Run all code quality checks

# ------------------------------------------------------------------------------
# Docker
# ------------------------------------------------------------------------------

docker-build: ## Build the Docker images for the application
	$(DOCKER_COMPOSE) build

docker-push: ## Push the Docker images to a registry
	$(DOCKER_COMPOSE) push

# ------------------------------------------------------------------------------
# Database
# ------------------------------------------------------------------------------

db-init: ## Initialize the database and create tables
	$(DOCKER_COMPOSE) exec app python -c "import asyncio; from database import initialize_database; asyncio.run(initialize_database())"

db-shell: ## Open a shell to the PostgreSQL database
	$(DOCKER_COMPOSE) exec postgres psql -U postgres -d document_intelligence

# ------------------------------------------------------------------------------
# Quick Start
# ------------------------------------------------------------------------------

quick-start: setup dev ## A quick way to set up and start the project
	@echo "Quick start complete. The application is now running."

.PHONY: help install-backend install-frontend setup dev stop-dev logs test-backend test-frontend test lint-backend format-backend lint-frontend format-frontend quality-check docker-build docker-push db-init db-shell quick-start