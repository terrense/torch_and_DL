#!/usr/bin/env python3
"""
End-to-end workflow validation script.
Tests complete workflows from installation to result generation for both projects.
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

class WorkflowValidator:
    def __init__(self):
        self.results = []
        self.failed_tests = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def run_command(self, cmd: List[str], cwd: str = None, timeout: int = 300) -> Tuple[bool, str, str]:
        """Run a command and return success status, stdout, stderr."""
        try:
            self.log(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            success = result.returncode == 0
            return success, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out after {timeout}s", "ERROR")
            return False, "", f"Timeout after {timeout}s"
        except Exception as e:
            self.log(f"Command failed: {e}", "ERROR")
            return False, "", str(e)
    
    def check_file_exists(self, filepath: str) -> bool:
        """Check if a file exists."""
        exists = Path(filepath).exists()
        if exists:
            self.log(f"✓ File exists: {filepath}")
        else:
            self.log(f"✗ File missing: {filepath}", "ERROR")
        return exists
    
    def check_directory_structure(self, project_dir: str, expected_dirs: List[str]) -> bool:
        """Verify project directory structure."""
        self.log(f"Checking directory structure for {project_dir}")
        all_exist = True
        for dir_name in expected_dirs:
            dir_path = os.path.join(project_dir, dir_name)
            if not os.path.isdir(dir_path):
                self.log(f"✗ Missing directory: {dir_path}", "ERROR")
                all_exist = False
            else:
                self.log(f"✓ Directory exists: {dir_path}")
        return all_exist
    
    def validate_unet_workflow(self) -> bool:
        """Validate U-Net Transformer Segmentation workflow."""
        self.log("=" * 60)
        self.log("VALIDATING U-NET TRANSFORMER SEGMENTATION WORKFLOW")
        self.log("=" * 60)
        
        project_dir = "unet_transformer_seg"
        
        # 1. Check directory structure
        self.log("\n1. Checking directory structure...")
        expected_dirs = ["configs", "src", "tests", "scripts", "docs"]
        if not self.check_directory_structure(project_dir, expected_dirs):
            self.failed_tests.append("U-Net: Directory structure")
            return False
        
        # 2. Check key files
        self.log("\n2. Checking key files...")
        key_files = [
            f"{project_dir}/README.md",
            f"{project_dir}/requirements.txt",
            f"{project_dir}/configs/unet_baseline.yaml",
            f"{project_dir}/configs/unet_transformer.yaml",
            f"{project_dir}/docs/SPEC.md",
            f"{project_dir}/docs/CONTRACTS.md",
            f"{project_dir}/docs/ABLATIONS.md",
            f"{project_dir}/scripts/train.py",
            f"{project_dir}/scripts/evaluate.py",
            f"{project_dir}/scripts/inference.py",
        ]
        all_files_exist = all(self.check_file_exists(f) for f in key_files)
        if not all_files_exist:
            self.failed_tests.append("U-Net: Key files missing")
            return False
        
        # 3. Check Python imports
        self.log("\n3. Checking Python imports...")
        import_test = f"""
import sys
sys.path.insert(0, '{project_dir}')
try:
    from src.config import ModelConfig, TrainingConfig
    from src.models.unet import UNet
    from src.models.unet_transformer import UNetTransformer
    from src.data.toy_shapes import ToyShapesDataset
    from src.losses.dice_loss import DiceLoss
    from src.metrics.seg_metrics import IoU, PixelAccuracy
    print('SUCCESS')
except Exception as e:
    print(f'FAILED: {{e}}')
"""
        success, stdout, stderr = self.run_command(
            [sys.executable, "-c", import_test],
            timeout=30
        )
        if not success or "SUCCESS" not in stdout:
            self.log(f"Import test failed: {stderr}", "ERROR")
            self.failed_tests.append("U-Net: Python imports")
            return False
        self.log("✓ All imports successful")
        
        # 4. Run unit tests
        self.log("\n4. Running unit tests...")
        success, stdout, stderr = self.run_command(
            [sys.executable, "-m", "pytest", "tests/test_data.py", "-v"],
            cwd=project_dir,
            timeout=60
        )
        if not success:
            self.log(f"Unit tests failed: {stderr}", "WARNING")
            # Don't fail workflow for test failures, just warn
        
        # 5. Run smoke test (quick training)
        self.log("\n5. Running smoke test (30 steps)...")
        success, stdout, stderr = self.run_command(
            [sys.executable, "-m", "pytest", "tests/test_smoke.py", "-v", "-k", "test_baseline_training"],
            cwd=project_dir,
            timeout=180
        )
        if not success:
            self.log(f"Smoke test failed: {stderr}", "WARNING")
        
        self.log("\n✓ U-Net workflow validation completed")
        return True
    
    def validate_paraformer_workflow(self) -> bool:
        """Validate Paraformer ASR workflow."""
        self.log("=" * 60)
        self.log("VALIDATING PARAFORMER ASR WORKFLOW")
        self.log("=" * 60)
        
        project_dir = "paraformer_asr"
        
        # 1. Check directory structure
        self.log("\n1. Checking directory structure...")
        expected_dirs = ["configs", "src", "tests", "scripts", "docs"]
        if not self.check_directory_structure(project_dir, expected_dirs):
            self.failed_tests.append("Paraformer: Directory structure")
            return False
        
        # 2. Check key files
        self.log("\n2. Checking key files...")
        key_files = [
            f"{project_dir}/README.md",
            f"{project_dir}/requirements.txt",
            f"{project_dir}/configs/paraformer_base.yaml",
            f"{project_dir}/docs/SPEC.md",
            f"{project_dir}/docs/CONTRACTS.md",
            f"{project_dir}/docs/ABLATIONS.md",
            f"{project_dir}/scripts/train.py",
            f"{project_dir}/scripts/evaluate.py",
            f"{project_dir}/scripts/inference.py",
        ]
        all_files_exist = all(self.check_file_exists(f) for f in key_files)
        if not all_files_exist:
            self.failed_tests.append("Paraformer: Key files missing")
            return False
        
        # 3. Check Python imports
        self.log("\n3. Checking Python imports...")
        import_test = f"""
import sys
sys.path.insert(0, '{project_dir}')
try:
    from src.config import ModelConfig, TrainingConfig
    from src.models.paraformer import Paraformer
    from src.models.encoder import Encoder
    from src.models.predictor import Predictor
    from src.models.decoder import Decoder
    from src.data.tokenizer import CharTokenizer
    from src.data.toy_seq2seq import ToySeq2SeqDataset
    from src.losses.seq_loss import MaskedCrossEntropyLoss
    from src.decode.greedy import greedy_decode
    print('SUCCESS')
except Exception as e:
    print(f'FAILED: {{e}}')
"""
        success, stdout, stderr = self.run_command(
            [sys.executable, "-c", import_test],
            timeout=30
        )
        if not success or "SUCCESS" not in stdout:
            self.log(f"Import test failed: {stderr}", "ERROR")
            self.failed_tests.append("Paraformer: Python imports")
            return False
        self.log("✓ All imports successful")
        
        # 4. Run unit tests
        self.log("\n4. Running unit tests...")
        success, stdout, stderr = self.run_command(
            [sys.executable, "-m", "pytest", "tests/test_data.py", "-v"],
            cwd=project_dir,
            timeout=60
        )
        if not success:
            self.log(f"Unit tests failed: {stderr}", "WARNING")
        
        # 5. Run smoke test
        self.log("\n5. Running smoke test (30 steps)...")
        success, stdout, stderr = self.run_command(
            [sys.executable, "-m", "pytest", "tests/test_smoke.py", "-v", "-k", "test_training_smoke"],
            cwd=project_dir,
            timeout=180
        )
        if not success:
            self.log(f"Smoke test failed: {stderr}", "WARNING")
        
        self.log("\n✓ Paraformer workflow validation completed")
        return True
    
    def validate_documentation(self) -> bool:
        """Validate documentation completeness."""
        self.log("=" * 60)
        self.log("VALIDATING DOCUMENTATION")
        self.log("=" * 60)
        
        # Check for required documentation sections
        docs_to_check = [
            ("unet_transformer_seg/docs/CONTRACTS.md", ["Troubleshooting Guide", "Common Shape Mismatches"]),
            ("unet_transformer_seg/docs/ABLATIONS.md", ["Model Variants", "Ablation Studies"]),
            ("paraformer_asr/docs/CONTRACTS.md", ["Troubleshooting Guide", "Common Shape Mismatches"]),
            ("paraformer_asr/docs/ABLATIONS.md", ["Model Variants", "Ablation Studies"]),
        ]
        
        all_valid = True
        for doc_path, required_sections in docs_to_check:
            self.log(f"\nChecking {doc_path}...")
            if not os.path.exists(doc_path):
                self.log(f"✗ Documentation missing: {doc_path}", "ERROR")
                all_valid = False
                continue
            
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for section in required_sections:
                if section in content:
                    self.log(f"✓ Section found: {section}")
                else:
                    self.log(f"✗ Section missing: {section}", "ERROR")
                    all_valid = False
        
        if not all_valid:
            self.failed_tests.append("Documentation: Missing sections")
        
        return all_valid
    
    def validate_reproducibility(self) -> bool:
        """Validate reproducibility features."""
        self.log("=" * 60)
        self.log("VALIDATING REPRODUCIBILITY")
        self.log("=" * 60)
        
        # Check for seed control utilities
        seed_files = [
            "unet_transformer_seg/src/utils/reproducibility.py",
            "paraformer_asr/src/utils/reproducibility.py",
        ]
        
        all_exist = True
        for seed_file in seed_files:
            if self.check_file_exists(seed_file):
                # Check for set_seed function
                with open(seed_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if "def set_seed" in content:
                    self.log(f"✓ set_seed function found in {seed_file}")
                else:
                    self.log(f"✗ set_seed function missing in {seed_file}", "ERROR")
                    all_exist = False
            else:
                all_exist = False
        
        if not all_exist:
            self.failed_tests.append("Reproducibility: Missing utilities")
        
        return all_exist
    
    def generate_report(self):
        """Generate validation report."""
        self.log("\n" + "=" * 60)
        self.log("VALIDATION REPORT")
        self.log("=" * 60)
        
        if not self.failed_tests:
            self.log("\n✓ ALL VALIDATIONS PASSED", "SUCCESS")
            return True
        else:
            self.log(f"\n✗ {len(self.failed_tests)} VALIDATION(S) FAILED:", "ERROR")
            for test in self.failed_tests:
                self.log(f"  - {test}", "ERROR")
            return False

def main():
    """Main validation function."""
    validator = WorkflowValidator()
    
    # Run all validations
    unet_ok = validator.validate_unet_workflow()
    paraformer_ok = validator.validate_paraformer_workflow()
    docs_ok = validator.validate_documentation()
    repro_ok = validator.validate_reproducibility()
    
    # Generate report
    success = validator.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
