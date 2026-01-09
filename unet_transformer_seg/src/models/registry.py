"""Model registry for configuration-based model building."""

import logging
from typing import Dict, Type, Any, Optional, Callable
import torch.nn as nn

from ..config import ModelConfig


logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for model classes and factory functions."""
    
    _models: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Callable) -> None:
        """
        Register a model class or factory function.
        
        Args:
            name: Model name for registry
            model_class: Model class or factory function
        """
        if name in cls._models:
            logger.warning(f"Overriding existing model registration: {name}")
        
        cls._models[name] = model_class
        logger.info(f"Registered model: {name}")
    
    @classmethod
    def create(cls, config: ModelConfig) -> nn.Module:
        """
        Create model instance from configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model name not found in registry
        """
        model_name = config.name
        
        if model_name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {available_models}"
            )
        
        model_factory = cls._models[model_name]
        
        try:
            # Try to create model with config
            model = model_factory(config)
            logger.info(f"Created model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to create model '{model_name}': {e}")
            raise
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a registered model.
        
        Args:
            name: Model name
            
        Returns:
            Dictionary with model information
        """
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        model_class = cls._models[name]
        
        info = {
            'name': name,
            'class': model_class.__name__ if hasattr(model_class, '__name__') else str(model_class),
            'module': model_class.__module__ if hasattr(model_class, '__module__') else 'unknown',
        }
        
        # Try to get docstring
        if hasattr(model_class, '__doc__') and model_class.__doc__:
            info['description'] = model_class.__doc__.strip().split('\n')[0]
        
        return info


def register_model(name: str) -> Callable:
    """
    Decorator for registering model classes.
    
    Args:
        name: Model name for registry
        
    Returns:
        Decorator function
    """
    def decorator(model_class: Callable) -> Callable:
        ModelRegistry.register(name, model_class)
        return model_class
    
    return decorator


def create_model(config: ModelConfig) -> nn.Module:
    """
    Convenience function to create model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Model instance
    """
    return ModelRegistry.create(config)


def list_available_models() -> list:
    """List all available model names."""
    return ModelRegistry.list_models()


def get_model_info(name: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    return ModelRegistry.get_model_info(name)