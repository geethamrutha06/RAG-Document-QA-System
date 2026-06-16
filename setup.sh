#!/bin/bash
echo "====================================="
echo "RAG Document Q&A System Setup"
echo "====================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Please edit .env file and add your OpenAI API key"
    echo "   You can get one from: https://platform.openai.com/api-keys"
fi

echo ""
echo "====================================="
echo "✅ Setup complete!"
echo "====================================="
echo ""
echo "To run the application:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
