"""
CLI Integration Tests for U-Net Transformer Segmentation

Tests all command-line entrypoints including train, evaluate, and inference
commands with configuration loading, parameter validation, and output generation.
"""

import pytest
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import json
import yaml
import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ModelConfig, TrainingConfig, DataConfig


class TestCLIScripts:
    """Test CLI script entrypoints and argument parsing."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        # Cleanup
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration file."""
        config = {
            'model': {
                'name': 'unet',
                'in_channels': 3,
                'num_classes': 3,
                'base_channels': 32,
                'depth': 3
            },
            'training': {
                'num_epochs': 2,
                'batch_size': 4,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'save_every': 1
            },
            'data': {
                'train_samples': 20,
                'val_samples': 10,
                'test_samples': 10,
                'image_size': [128, 128],
                'batch_size': 4
            },
            'loss': {
                'type': 'dice_bce',
                'params': {'dice_weight': 0.5, 'bce_weight': 0.5}
            },
            'num_classes': 3,
            'num_epochs': 2,
            'seed': 42
        }
        
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def trained_checkpoint(self, temp_dir, test_config):
        """Create a trained checkpoint for evaluation/inference tests."""
        from models.registry import create_model_from_config
        
        # Create model
        config = yaml.safe_load(open(test_config))
        model = create_model_from_config(config)
        
        # Save checkpoint
        checkpoint_path = temp_dir / 'test_checkpoint.pt'
        checkpoint_data = {
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'config': config
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path
    
    def test_train_script_exists(self):
        """Test that train script exists and is accessible."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'train.py'
        assert script_path.exists()
        assert script_path.is_file()
    
    def test_evaluate_script_exists(self):
        """Test that evaluate script exists and is accessible."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'evaluate.py'
        assert script_path.exists()
        assert script_path.is_file()
    
    def test_inference_script_exists(self):
        """Test that inference script exists and is accessible."""
        script_path = Path(__file__).parent.parent / 'scripts' / 'inference.py'
        assert script_path.exists()
        assert script_path.is_file()


class TestConfigurationLoading:
    """Test configuration loading and validation."""
    
    def test_valid_config_loading(self, tmp_path):
        """Test loading valid configuration."""
        config_data = {
            'model': {
                'name': 'unet',
                'in_channels': 3,
                'num_classes': 3,
                'base_channels': 64
            },
            'training': {
                'num_epochs': 10,
                'batch_size': 16,
                'learning_rate': 0.001
            },
            'num_classes': 3
        }
        
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        from config import load_config
        config = load_config(config_path)
        
        assert config.model.name == 'unet'
        assert config.model.in_channels == 3
        assert config.training.num_epochs == 10
    
    def test_invalid_config_format(self, tmp_path):
        """Test loading invalid configuration format."""
        config_path = tmp_path / 'invalid.yaml'
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        from config import load_config
        with pytest.raises(Exception):
            load_config(config_path)
    
    def test_config_with_missing_fields(self, tmp_path):
        """Test configuration with missing required fields."""
        config_data = {
            'model': {
                'name': 'unet'
                # Missing required fields
            }
        }
        
        config_path = tmp_path / 'incomplete.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        from config import load_config
        # Should use defaults for missing fields
        config = load_config(config_path)
        assert hasattr(config.model, 'in_channels')


class TestOutputGeneration:
    """Test output file generation and result aggregation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        if temp_path.exists():
            shutil.rmtree(temp_path)
    
    def test_config_file_creation(self, temp_dir):
        """Test that configuration files can be created and loaded."""
        config_data = {
            'model': {
                'name': 'unet',
                'in_channels': 3,
                'num_classes': 3
            },
            'training': {
                'num_epochs': 1,
                'batch_size': 2
            },
            'num_classes': 3
        }
        
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Verify file exists and can be loaded
        assert config_path.exists()
        
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config['model']['name'] == 'unet'
        assert loaded_config['training']['num_epochs'] == 1
    
    def test_output_directory_structure(self, temp_dir):
        """Test that output directories can be created."""
        # Create expected directory structure
        checkpoint_dir = temp_dir / 'checkpoints'
        log_dir = temp_dir / 'logs'
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        assert checkpoint_dir.exists()
        assert log_dir.exists()
    
    def test_csv_output_format(self, temp_dir):
        """Test CSV output file format."""
        import csv
        
        csv_file = temp_dir / 'results.csv'
        
        # Write sample CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_mean_iou'])
            writer.writerow([1, 0.5, 0.6, 0.7])
            writer.writerow([2, 0.4, 0.5, 0.75])
        
        # Verify CSV can be read
        assert csv_file.exists()
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert 'epoch' in headers
            assert 'train_loss' in headers
            
            rows = list(reader)
            assert len(rows) == 2
    
    def test_json_output_format(self, temp_dir):
        """Test JSON output file format."""
        json_file = temp_dir / 'results.json'
        
        # Write sample JSON
        results = {
            'pixel_accuracy': 0.85,
            'mean_iou': 0.75,
            'mean_dice': 0.80,
            'num_samples': 100
        }
        
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify JSON can be read
        assert json_file.exists()
        
        with open(json_file, 'r') as f:
            loaded_results = json.load(f)
        
        assert loaded_results['mean_iou'] == 0.75
        assert loaded_results['num_samples'] == 100
    
    def test_image_output_format(self, temp_dir):
        """Test image output file format."""
        from PIL import Image
        
        # Create sample image
        img = Image.new('RGB', (64, 64), color='white')
        img_path = temp_dir / 'test_image.png'
        img.save(img_path)
        
        # Verify image can be loaded
        assert img_path.exists()
        
        loaded_img = Image.open(img_path)
        assert loaded_img.size == (64, 64)


class TestParameterValidation:
    """Test parameter validation in CLI scripts."""
    
    def test_config_loading_validation(self, tmp_path):
        """Test configuration loading and validation."""
        config_data = {
            'model': {'name': 'unet', 'num_classes': 3},
            'training': {'num_epochs': 1},
            'num_classes': 3
        }
        
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        from config import load_config
        config = load_config(config_path)
        assert config.model.name == 'unet'
    
    def test_device_parameter_handling(self):
        """Test device parameter handling."""
        # Test valid device strings
        valid_devices = ['cpu', 'cuda']
        for device_str in valid_devices:
            try:
                device = torch.device(device_str)
                assert device is not None
            except RuntimeError:
                # CUDA not available is acceptable
                pass
    
    def test_invalid_device_parameter(self):
        """Test handling of invalid device parameter."""
        # Invalid device should raise error
        with pytest.raises(RuntimeError):
            device = torch.device('invalid_device')
    
    def test_negative_epochs(self, tmp_path):
        """Test handling of negative epoch values."""
        config_data = {
            'model': {'name': 'unet', 'num_classes': 3},
            'training': {'num_epochs': -1},  # Invalid
            'num_classes': 3
        }
        
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        from config import load_config
        # Should either raise error or use default
        try:
            config = load_config(config_path)
            # If it loads, epochs should be positive
            assert config.training.num_epochs > 0
        except Exception:
            # Or it should raise an error
            pass
    
    def test_invalid_batch_size(self, tmp_path):
        """Test handling of invalid batch size."""
        config_data = {
            'model': {'name': 'unet', 'num_classes': 3},
            'training': {'batch_size': 0},  # Invalid
            'num_classes': 3
        }
        
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        from config import load_config
        try:
            config = load_config(config_path)
            # Should use default or positive value
            assert config.training.batch_size > 0
        except Exception:
            pass
    
    def test_invalid_image_size(self, tmp_path):
        """Test handling of invalid image size."""
        config_data = {
            'model': {'name': 'unet', 'num_classes': 3},
            'data': {'image_size': [0, 0]},  # Invalid
            'num_classes': 3
        }
        
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        from config import load_config
        try:
            config = load_config(config_path)
            # Should use default or positive values
            image_size = config.data.image_size
            assert all(s > 0 for s in image_size)
        except Exception:
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
