"""
Complete ASL Dataset Setup - Downloads professional dataset
"""

import os
import sys

print("=" * 70)
print("ASL ALPHABET - PROFESSIONAL DATASET SETUP")
print("=" * 70)

print("\n📋 INSTRUCTIONS TO FIX WRONG PREDICTIONS:")
print("-" * 70)

print("\n✅ STEP 1: Download Professional ASL Dataset")
print("   Go to: https://www.kaggle.com/datasets/kapillondhe/asl-hand-keypoints")
print("   Click 'Download' (create free account if needed)")
print("   You'll get: asl-hand-keypoints.zip")

print("\n✅ STEP 2: Extract and Replace Data")
print("   1. Extract the downloaded ZIP file")
print("   2. You'll see a 'data' folder with A-Z subfolders")
print("   3. Each folder has .npy files (hand keypoints)")

print("\n✅ STEP 3: Replace Your Current Data")
backup_exists = os.path.exists("data_backup")
if not backup_exists and os.path.exists("data"):
    print("   Creating backup of your current data...")
    try:
        os.rename("data", "data_backup")
        print("   ✓ Your data backed up to 'data_backup'")
    except Exception as e:
        print(f"   Note: {e}")

print("\n   Now copy the downloaded 'data' folder to this location:")
print(f"   {os.path.abspath('.')}")

print("\n✅ STEP 4: Verify Data Structure")
print("   Your folder should look like:")
print("   data/")
print("     ├── A/")
print("     │   ├── 0/")
print("     │   │   ├── 0.npy")
print("     │   │   ├── 1.npy")
print("     │   │   └── ... (30 files)")
print("     │   ├── 1/")
print("     │   └── ...")
print("     ├── B/")
print("     ├── C/")
print("     └── ... (through Z)")

print("\n✅ STEP 5: Train Model on Professional Data")
print("   Run: python train_improved.py")

print("\n✅ STEP 6: Test Predictions")
print("   Run: python predict_debug.py")

print("\n" + "=" * 70)
print("🔍 CHECKING CURRENT STATUS...")
print("=" * 70)

if os.path.exists("data"):
    letters = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
    print(f"✓ Data folder exists with {len(letters)} letters")
    
    if len(letters) == 26:
        print("✓ All 26 letters present (A-Z)")
        
        # Check one letter
        test_letter = "A"
        if test_letter in letters:
            seqs = os.listdir(os.path.join("data", test_letter))
            print(f"✓ Letter '{test_letter}' has {len(seqs)} sequences")
            
            if len(seqs) > 0:
                first_seq = seqs[0]
                frames = len([f for f in os.listdir(os.path.join("data", test_letter, first_seq)) if f.endswith('.npy')])
                print(f"✓ Each sequence has {frames} frames")
                
                print("\n✅ DATA IS READY!")
                print("   Now run: python train_improved.py")
            else:
                print("\n⚠️  No sequences found. Please download the professional dataset.")
    else:
        print(f"⚠️  Only {len(letters)} letters found. Expected 26.")
        print("   Please download the complete dataset from Kaggle.")
else:
    print("❌ No 'data' folder found!")
    print("   Please download and extract the Kaggle dataset.")

print("\n" + "=" * 70)
print("📌 REMEMBER: The professional dataset will give MUCH better results!")
print("=" * 70)
