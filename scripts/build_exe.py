"""
Build Script to Create Standalone Executable
This script packages the ASL Recognition application into a single .exe file
"""

import os
import subprocess
import sys
import shutil

print("=" * 70)
print("ASL RECOGNITION - EXECUTABLE BUILDER")
print("=" * 70)

# Check if PyInstaller is installed
try:
    import PyInstaller
    print("✅ PyInstaller is installed")
except ImportError:
    print("📦 Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    print("✅ PyInstaller installed successfully!")

print("\n🔨 Building executable...")
print("-" * 70)

# PyInstaller command to create executable
build_command = [
    "pyinstaller",
    "--onefile",  # Create a single executable file
    "--windowed",  # Don't show console window (remove this if you want console)
    "--name=ASL_Recognition",  # Name of the executable
    "--add-data=action_v2.h5;.",  # Include model file
    "--icon=NONE",  # You can add an icon file here if you have one
    "--hidden-import=pyttsx3.drivers",
    "--hidden-import=pyttsx3.drivers.sapi5",
    "--hidden-import=cv2",
    "--hidden-import=mediapipe",
    "--hidden-import=tensorflow",
    "--collect-all=mediapipe",
    "--collect-all=tensorflow",
    "asl_recognition_app.py"
]

print(f"Command: {' '.join(build_command)}\n")

try:
    result = subprocess.run(build_command, check=True, capture_output=True, text=True)
    print(result.stdout)
    
    print("\n" + "=" * 70)
    print("✅ BUILD SUCCESSFUL!")
    print("=" * 70)
    print("\n📁 Executable Location:")
    print(f"   {os.path.abspath('dist/ASL_Recognition.exe')}")
    print("\n📋 To run the application:")
    print("   1. Navigate to the 'dist' folder")
    print("   2. Double-click 'ASL_Recognition.exe'")
    print("   3. Make sure 'action_v2.h5' is in the same folder as the .exe")
    print("=" * 70)
    
except subprocess.CalledProcessError as e:
    print("\n❌ BUILD FAILED!")
    print(e.stderr)
    sys.exit(1)