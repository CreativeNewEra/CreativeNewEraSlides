#!/bin/bash
# Activation script for CreativeNewEraSlides

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install .[dev]
else
    source venv/bin/activate
fi

# Run the application
echo "Starting CreativeNewEraSlides..."
python main.py