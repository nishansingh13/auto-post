#!/bin/bash
# Simple Whisper Setup - Just what we need

echo "🎬 Simple Setup - Whisper Caption Generator"

# Activate your virtual environment
echo "🐍 Activating myenv..."
source ~/myenv/bin/activate

# Install only what we need
echo "📦 Installing Whisper and MoviePy..."
pip install openai-whisper moviepy pillow numpy

echo ""
echo "✅ Done! Ready to use:"
echo "   python whisper_captions.py"
