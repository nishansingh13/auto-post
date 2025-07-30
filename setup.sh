#!/bin/bash
# Setup script for video caption generation with word highlighting

echo "🎬 Setting up Video Caption Generator with Word Highlighting..."

# Install system dependencies
echo "📦 Installing system dependencies..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Ubuntu/Debian
    sudo apt update
    sudo apt install -y ffmpeg espeak espeak-data libespeak1 libespeak-dev portaudio19-dev python3-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    brew install ffmpeg espeak portaudio
fi

# Activate your virtual environment
echo "🐍 Activating myenv virtual environment..."
source ~/myenv/bin/activate

# Install Python dependencies
echo "� Installing Python dependencies in myenv..."
pip install -r requirements.txt

# Install additional tools for professional alignment
echo "🔧 Installing additional alignment tools..."

# Option 1: Montreal Forced Alignment (MFA) - Professional grade
echo "Installing Montreal Forced Alignment (MFA)..."
conda install -c conda-forge montreal-forced-alignment || echo "⚠️  MFA installation failed - conda required"

# Option 2: Gentle Forced Aligner via Docker
echo "Setting up Gentle Forced Aligner (Docker)..."
if command -v docker &> /dev/null; then
    docker pull lowerquality/gentle
    echo "✅ Gentle aligner ready"
else
    echo "⚠️  Docker not found - Gentle aligner unavailable"
fi

# Create necessary directories
mkdir -p output temp

echo "✅ Setup complete! Here's what you can do now:"
echo ""
echo "🎯 RECOMMENDED APPROACH (You have transcript!):"
echo ""
echo "⭐ FORCED ALIGNMENT - Best Quality for Your Use Case"
echo "   ✅ Perfect word synchronization"
echo "   ✅ Broadcast-quality timing"
echo "   ✅ Works with your existing dialogue"
echo "   ✅ Professional results"
echo ""
echo "📋 ALL AVAILABLE APPROACHES:"
echo ""
echo "1️⃣  FORCED ALIGNMENT (RECOMMENDED - You have transcript!)"
echo "   - Montreal Forced Alignment (MFA) - Professional grade"
echo "   - Wav2Vec2 alignment - Good balance"
echo "   - Gentle aligner - Easy Docker setup"
echo ""
echo "2️⃣  WHISPER (Alternative - No transcript needed)"
echo "   - Automatic transcription + word timing"
echo "   - Good for any video content"
echo "   - Medium accuracy for word sync"
echo ""
echo "3️⃣  AUDIO FEATURES (Fallback - Basic alignment)"
echo "   - Uses energy/spectral analysis"
echo "   - Works without external dependencies"
echo "   - Lower accuracy but always available"
echo ""
echo "🎨 CAPTION STYLES AVAILABLE:"
echo "   • modern    - TikTok/Instagram style with highlighting"
echo "   • karaoke   - Progressive word highlighting"  
echo "   • subtitle  - Traditional subtitle with emphasis"
echo "   • classic   - Simple word-by-word display"
echo ""
echo "🚀 QUICK START (Optimized for your project):"
echo "   python enhance_with_captions.py        # Uses your existing dialogue"
echo "   python scripts/caption_generator.py    # Advanced options"
echo ""
echo "💡 Since you have transcript from Peter/Stewie dialogue:"
echo "   → Use forced alignment for perfect word synchronization!"
echo "   → Try: python optimize_for_transcript.py"
