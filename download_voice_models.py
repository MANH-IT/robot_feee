"""
download_voice_models.py - Tự động tải Vosk và Stanza models cho voice recognition
"""

import os
import sys
import urllib.request
import zipfile
import subprocess
from pathlib import Path

def download_vosk_model():
    """Tải Vosk model tiếng Việt (khoảng 40MB)"""
    print("\n🔊 [1/2] Downloading Vosk Vietnamese model...")
    
    models_dir = Path("models/vosk_vn")
    if models_dir.exists():
        print(f"   ✅ Model already exists at {models_dir}")
        return
    
    vosk_url = "https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip"
    zip_path = Path("models/vosk_model.zip")
    
    try:
        # Tải file zip
        print(f"   📥 Downloading from {vosk_url}")
        urllib.request.urlretrieve(vosk_url, zip_path)
        print(f"   ✅ Downloaded to {zip_path}")
        
        # Giải nén
        print(f"   📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("models")
        
        # Rename folder
        extracted = Path("models/vosk-model-small-vn-0.4")
        if extracted.exists():
            extracted.rename(models_dir)
        
        # Clean up
        zip_path.unlink()
        
        print(f"   ✅ Vosk model installed at {models_dir}")
        
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        print(f"   💡 Please manually download from: {vosk_url}")

def download_stanza_model():
    """Tải Stanza model tiếng Việt"""
    print("\n📖 [2/2] Downloading Stanza Vietnamese model...")
    
    try:
        import stanza
        # Download Vietnamese model
        print("   Downloading vi model...")
        stanza.download('vi', verbose=True)
        print(f"   ✅ Stanza Vietnamese model downloaded")
        
    except ImportError:
        print(f"   ❌ Stanza not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "stanza"])
        print(f"   ✅ Stanza installed. Please run this script again.")
        
    except Exception as e:
        print(f"   ❌ Download failed: {e}")

def verify_installation():
    """Kiểm tra models đã được cài đặt"""
    print("\n" + "="*50)
    print("🔍 VERIFYING INSTALLATION")
    print("="*50)
    
    # Check Vosk
    vosk_path = Path("models/vosk_vn")
    if vosk_path.exists():
        print(f"✅ Vosk model: {vosk_path}")
    else:
        print(f"❌ Vosk model missing at {vosk_path}")
    
    # Check Stanza
    try:
        import stanza
        print(f"✅ Stanza installed: {stanza.__version__}")
    except ImportError:
        print(f"❌ Stanza not installed")
    
    # Check SNN intent model
    snn_path = Path("models/snn_intent.pth")
    if snn_path.exists():
        print(f"✅ SNN Intent model: {snn_path}")
    else:
        print(f"❌ SNN Intent model missing")

if __name__ == "__main__":
    print("="*50)
    print("🎤 VOICE MODELS DOWNLOADER")
    print("Robot FEEE - Offline Speech Recognition")
    print("="*50)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Download models
    download_vosk_model()
    download_stanza_model()
    
    # Verify
    verify_installation()
    
    print("\n✅ Setup complete! Voice recognition is ready.")
