"""
Download Pre-collected ASL Alphabet Dataset
This will download a professional dataset from Kaggle with hand keypoints already extracted
"""

print("=" * 70)
print("ASL ALPHABET DATASET DOWNLOADER")
print("=" * 70)
print("\nTo download the professional ASL dataset from Kaggle:")
print("\n1. Install Kaggle API:")
print("   pip install kaggle")
print("\n2. Setup Kaggle credentials:")
print("   - Go to https://www.kaggle.com/settings")
print("   - Click 'Create New API Token'")
print("   - Save kaggle.json to: C:\\Users\\User\\.kaggle\\kaggle.json")
print("\n3. Download the dataset:")
print("   kaggle datasets download -d kapillondhe/asl-hand-keypoints")
print("\n4. Extract the dataset")
print("=" * 70)

import os
import subprocess
import sys

def install_kaggle():
    """Install kaggle package"""
    print("\n📦 Installing Kaggle API...")
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
    print("✓ Kaggle installed")

def check_kaggle_auth():
    """Check if Kaggle is authenticated"""
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    kaggle_path_win = os.path.expanduser("~\\.kaggle\\kaggle.json")
    
    if os.path.exists(kaggle_path) or os.path.exists(kaggle_path_win):
        return True
    return False

def download_dataset():
    """Download ASL hand keypoints dataset"""
    print("\n📥 Downloading ASL Hand Keypoints Dataset...")
    
    # Try to download
    try:
        result = subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "kapillondhe/asl-hand-keypoints",
            "-p", ".",
            "--unzip"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Dataset downloaded successfully!")
            return True
        else:
            print(f"⚠️  Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("⚠️  Kaggle command not found. Please install kaggle first.")
        return False

if __name__ == "__main__":
    print("\nStarting download process...\n")
    
    # Step 1: Install kaggle
    try:
        import kaggle
        print("✓ Kaggle API already installed")
    except ImportError:
        install_kaggle()
    
    # Step 2: Check authentication
    if not check_kaggle_auth():
        print("\n" + "=" * 70)
        print("⚠️  KAGGLE AUTHENTICATION REQUIRED")
        print("=" * 70)
        print("\nPlease follow these steps:")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Save the downloaded 'kaggle.json' file to:")
        print(f"   {os.path.expanduser('~')}/.kaggle/kaggle.json")
        print("\nThen run this script again.")
        print("=" * 70)
        sys.exit(1)
    
    print("✓ Kaggle authenticated")
    
    # Step 3: Download dataset
    if download_dataset():
        print("\n" + "=" * 70)
        print("✅ SUCCESS! Dataset ready to use")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ Download failed. Please download manually:")
        print("https://www.kaggle.com/datasets/kapillondhe/asl-hand-keypoints")
        print("=" * 70)
