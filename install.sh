#!/bin/bash
# Installation script for AmbedkarGPT SEMRAG System

echo "========================================="
echo "AmbedkarGPT SEMRAG System Installation"
echo "========================================="

# Check Python version
echo -e "\n[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo -e "\n[2/6] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate || . venv/Scripts/activate

# Upgrade pip
echo -e "\n[3/6] Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo -e "\n[4/6] Installing Python packages..."
pip install -r requirements.txt --quiet

# Download spaCy model
echo -e "\n[5/6] Downloading spaCy NER model..."
python -m spacy download en_core_web_sm --quiet

# Check Ollama
echo -e "\n[6/6] Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    echo "Available models:"
    ollama list
    
    # Check if llama3:8b is available
    if ollama list | grep -q "llama3:8b"; then
        echo "✓ llama3:8b model is ready"
    else
        echo "⚠ llama3:8b not found. Run: ollama pull llama3:8b"
    fi
else
    echo "⚠ Ollama not installed"
    echo "Please install from: https://ollama.com"
fi

echo -e "\n========================================="
echo "Installation Complete!"
echo "========================================="
echo -e "\nNext steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Place Ambedkar_book.pdf in data/ directory"
echo "3. Run demo: python demo.py --pdf data/Ambedkar_book.pdf"
echo "========================================="
