#!/usr/bin/env python3
"""
Setup script to fix import issues and make the model easily importable.
"""

import sys
import os
from pathlib import Path

def setup_python_path():
    """Setup Python path for proper imports."""
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    
    # Add to Python path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    # Set environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_paths = [str(src_dir), str(current_dir)]
    
    for path in new_paths:
        if path not in current_pythonpath:
            if current_pythonpath:
                current_pythonpath = f"{path}{os.pathsep}{current_pythonpath}"
            else:
                current_pythonpath = path
    
    os.environ['PYTHONPATH'] = current_pythonpath
    
    print(f"✓ Python path setup complete")
    print(f"  Added: {src_dir}")
    print(f"  Added: {current_dir}")

def test_imports():
    """Test that all imports work correctly."""
    try:
        from src.models.paraformer import ParaformerASR
        print("✓ ParaformerASR import successful")
        
        import torch
        model = ParaformerASR(input_dim=80, vocab_size=100)
        print("✓ Model creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Setting up imports for Paraformer ASR...")
    setup_python_path()
    
    if test_imports():
        print("\n🎉 Setup complete! You can now import the model:")
        print("   from src.models.paraformer import ParaformerASR")
    else:
        print("\n❌ Setup failed. Check the error messages above.")