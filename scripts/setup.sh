#!/bin/bash

# IT Support Chatbot - Setup Script
# This script sets up the development environment

set -e  # Exit on error

echo "========================================="
echo "IT Support Chatbot - Setup Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo -e "${GREEN}✓ Python 3 found: $(python3 --version)${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.9 or higher.${NC}"
    exit 1
fi

# Check pip
echo "Checking pip..."
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓ pip3 found${NC}"
else
    echo -e "${RED}✗ pip3 not found. Installing...${NC}"
    python3 -m ensurepip --upgrade
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p documents
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"

# Create .env file if it doesn't exist
echo ""
if [ ! -f ".env" ]; then
    if [ -f "config/.env.template" ]; then
        echo "Creating .env file from template..."
        cp config/.env.template .env
        echo -e "${GREEN}✓ .env file created${NC}"
        echo -e "${YELLOW}⚠ Please edit .env and add your ANTHROPIC_API_KEY${NC}"
    else
        echo -e "${YELLOW}⚠ .env.template not found, creating basic .env${NC}"
        echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
        echo -e "${GREEN}✓ Basic .env file created${NC}"
        echo -e "${YELLOW}⚠ Please edit .env and add your ANTHROPIC_API_KEY${NC}"
    fi
else
    echo -e "${YELLOW}⚠ .env file already exists${NC}"
fi

# Summary
echo ""
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Add your Anthropic API key to .env"
echo "2. Add PDF documents to the documents/ folder"
echo "3. Run the application:"
echo "   - Console: python src/it_support_chatbot_claude_api.py console"
echo "   - Web: streamlit run src/app_claude.py"
echo ""
echo "To activate the virtual environment later:"
echo "   source venv/bin/activate"
echo ""
echo "For more information, see README.md"
echo "========================================="
