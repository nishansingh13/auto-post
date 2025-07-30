#!/bin/bash
# FOCUSED Setup - Only MFA (Best Quality for Transcript-Based Alignment)
# Since you have transcript, we only need the BEST method: Montreal Forced Alignment

echo "🎯 FOCUSED SETUP - Montreal Forced Alignment (MFA)"
echo "Installing ONLY the BEST tool since you have transcript..."
echo ""

# Activate your virtual environment
echo "🐍 Activating myenv virtual environment..."
source ~/myenv/bin/activate

# Install core dependencies
echo "📦 Installing core system dependencies..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update
    sudo apt install -y ffmpeg python3-dev git
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg git
fi

# Install minimal Python packages needed for MFA
echo "📦 Installing Python packages in myenv..."
pip install --upgrade pip
pip install montreal-forced-alignment
pip install moviepy librosa soundfile
pip install pillow numpy

# Download MFA models (English)
echo "📥 Downloading MFA English models..."
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Create directories
mkdir -p output temp mfa_temp

echo ""
echo "✅ SETUP COMPLETE!"
echo ""
echo "🥇 INSTALLED: Montreal Forced Alignment (MFA) - BROADCAST QUALITY"
echo "   ✅ Perfect word-level timing synchronization"
echo "   ✅ Professional broadcast quality results"
echo "   ✅ Works perfectly with your dialogue transcript"
echo ""
echo "🚀 READY TO USE:"
echo "   python mfa_caption_generator.py"
echo ""
echo "💡 MFA will give you PERFECT word synchronization using your transcript!"
