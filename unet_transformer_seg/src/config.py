"""Configuration system with YAML loading and dataclass parsing."""

import os
import yaml
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional, Type, TypeVar, Union
from pathlib import Path

T = TypeVar('T')


@dataclass
class ModelConfig:
    """Base configuration for model architecture."""
    name: str = "unet_baseline"
    hidden_dim: int = 64
    num_layers: int = 4
    num_classes: int = 3
    input_channels: int = 3
    
    # U-Net specific
    encoder_channels: list = field(default_factory=lambda: [64, 128, 256, 512])
    decoder_channels: list = field(default_factory=lambda: [256, 128, 64, 32])
    
    # Transformer specific (for hybrid model)
    use_transformer: bool = False
    transformer_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    batch_size: int = 8
    image_size: tuple = field(default_factory=lambda: (256, 256))
    num_workers: int = 4
    
    # Toy dataset parameters
    num_shapes: int = 3
    noise_level: float = 0.1
    blur_sigma: float = 1.0
    occlusion_prob: float = 0.2
    class_imbalance: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    # Loss configuration
    loss_type: str = "dice_bce"
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    
    # Training options
    gradient_clip: Optional[float] = 1.0
    mixed_precision: bool = False
    accumulate_grad_batches: int = 1


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and logging."""
    name: str = "unet_experiment"
    seed: int = 42
    deterministic: bool = True
    
    # Logging
    log_every: int = 10
    save_every: int = 20
    eval_every: int = 10
    
    # Paths
    output_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


class ConfigLoader:
    """Utility class for loading and merging configurations."""
    
    @staticmethod
    def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
            
        return config_dict
    
    @staticmethod
    def dict_to_dataclass(config_dict: Dict[str, Any], config_class: Type[T]) -> T:
        """Convert dictionary to dataclass instance with validation."""
        if not is_dataclass(config_class):
            raise ValueError(f"{config_class} is not a dataclass")
        
        # Get field names and types
        field_names = {f.name for f in fields(config_class)}
        
        # Filter dictionary to only include valid fields
        filtered_dict = {}
        for key, value in config_dict.items():
            if key in field_names:
                field_type = next(f.type for f in fields(config_class) if f.name == key)
                
                # Handle nested dataclasses
                if is_dataclass(field_type) and isinstance(value, dict):
                    filtered_dict[key] = ConfigLoader.dict_to_dataclass(value, field_type)
                else:
                    filtered_dict[key] = value
        
        return config_class(**filtered_dict)
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    @classmethod
    def load_config(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Config:
        """
        Load configuration from YAML file with optional overrides.
        
        Args:
            config_path: Path to YAML configuration file
            overrides: Dictionary of configuration overrides
            
        Returns:
            Config: Parsed configuration object
        """
        # Start with default configuration
        config_dict = {}
        
        # Load from file if provided
        if config_path is not None:
            file_config = cls.load_yaml(config_path)
            config_dict = cls.merge_configs(config_dict, file_config)
        
        # Apply overrides if provided
        if overrides is not None:
            config_dict = cls.merge_configs(config_dict, overrides)
        
        # Convert to dataclass
        return cls.dict_to_dataclass(config_dict, Config)
    
    @staticmethod
    def save_config(config: Config, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dictionary
        config_dict = ConfigLoader.dataclass_to_dict(config)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def dataclass_to_dict(obj: Any) -> Any:
        """Convert dataclass to dictionary recursively."""
        if is_dataclass(obj):
            return {
                field.name: ConfigLoader.dataclass_to_dict(getattr(obj, field.name))
                for field in fields(obj)
            }
        elif isinstance(obj, (list, tuple)):
            return [ConfigLoader.dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: ConfigLoader.dataclass_to_dict(value) for key, value in obj.items()}
        else:
            return obj


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """Convenience function to load configuration."""
    return ConfigLoader.load_config(config_path, overrides)


def save_config(config: Config, path: Union[str, Path]) -> None:
    """Convenience function to save configuration."""
    ConfigLoader.save_config(config, path)