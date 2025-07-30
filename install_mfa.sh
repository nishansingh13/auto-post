#!/bin/bash
# Fixed MFA Setup - Install via conda

echo "🎯 INSTALLING MFA (Montreal Forced Alignment)"
echo "The BEST method for perfect word synchronization"
echo ""

# Activate your virtual environment first
source ~/myenv/bin/activate

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Installing Miniconda..."
    
    # Download and install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    source $HOME/miniconda/bin/activate
    
    # Initialize conda
    conda init bash
    echo "✅ Miniconda installed. Please restart your terminal and run this script again."
    exit 0
fi

# Install MFA via conda
echo "📦 Installing MFA via conda..."
conda install -c conda-forge montreal-forced-alignment -y

# Download English models
echo "📥 Downloading English models..."
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Install other dependencies in your virtual environment
echo "📦 Installing other dependencies in myenv..."
source ~/myenv/bin/activate
pip install moviepy pillow numpy

echo ""
echo "✅ MFA SETUP COMPLETE!"
echo ""
echo "🥇 Now you have the BEST alignment tool installed!"
echo ""
echo "🚀 Run: python mfa_caption_generator.py"
