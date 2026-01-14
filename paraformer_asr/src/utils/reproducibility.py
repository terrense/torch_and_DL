"""Utilities for reproducible training and seed control."""

import os
import random
import logging
import platform
from typing import Optional, Dict, Any

import numpy as np
import torch


logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducible training.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enable deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for deterministic behavior
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Enable deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    else:
        # Allow non-deterministic operations for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(False)
    
    logger.info(f"Set random seed to {seed}, deterministic={deterministic}")


def set_deterministic(deterministic: bool = True) -> None:
    """
    Enable or disable deterministic operations for reproducibility.
    启用或禁用确定性操作以实现可复现性。

    Args / 参数:
        deterministic (bool): Whether to enable deterministic operations
                             是否启用确定性操作

    Deterministic Mode / 确定性模式:
        When enabled / 启用时:
        - torch.backends.cudnn.deterministic = True
          使用确定性cuDNN算法
        - torch.backends.cudnn.benchmark = False
          禁用cuDNN自动调优
        - Sets CUBLAS_WORKSPACE_CONFIG environment variable
          设置CUBLAS_WORKSPACE_CONFIG环境变量
        - Enables deterministic algorithms in PyTorch
          在PyTorch中启用确定性算法

    Trade-offs / 权衡:
        Deterministic = True:
        + Reproducible results / 可复现的结果
        + Same output for same input / 相同输入产生相同输出
        - Slower performance / 性能较慢
        - Some operations not supported / 某些操作不支持

        Deterministic = False:
        + Faster performance / 性能更快
        + All operations supported / 支持所有操作
        - Non-reproducible results / 结果不可复现
        - Different output each run / 每次运行输出不同

    Example / 示例:
        >>> set_deterministic(True)  # For reproducible experiments
        >>> set_deterministic(False) # For faster training
    """
    if deterministic:
        # Enable deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for deterministic behavior
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Enable deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't support deterministic mode
                logger.warning("Could not enable deterministic algorithms")
    else:
        # Allow non-deterministic operations for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(False)
    
    logger.info(f"Set deterministic mode to {deterministic}")


def get_environment_info() -> Dict[str, Any]:
    """
    Collect environment information for reproducibility logging.
    
    Returns:
        Dictionary containing environment information
    """
    env_info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        env_info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        })
    
    # Environment variables that affect reproducibility
    reproducibility_env_vars = [
        'PYTHONHASHSEED',
        'CUBLAS_WORKSPACE_CONFIG',
        'CUDA_LAUNCH_BLOCKING',
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
    ]
    
    env_vars = {}
    for var in reproducibility_env_vars:
        value = os.environ.get(var)
        if value is not None:
            env_vars[var] = value
    
    if env_vars:
        env_info['environment_variables'] = env_vars
    
    return env_info


def log_environment_info() -> None:
    """Log environment information for reproducibility."""
    env_info = get_environment_info()
    
    logger.info("Environment Information:")
    logger.info(f"  Python: {env_info['python_version']}")
    logger.info(f"  Platform: {env_info['platform']}")
    logger.info(f"  PyTorch: {env_info['torch_version']}")
    logger.info(f"  NumPy: {env_info['numpy_version']}")
    logger.info(f"  CUDA Available: {env_info['cuda_available']}")
    
    if env_info['cuda_available']:
        logger.info(f"  CUDA Version: {env_info['cuda_version']}")
        logger.info(f"  cuDNN Version: {env_info['cudnn_version']}")
        logger.info(f"  GPU Count: {env_info['gpu_count']}")
        for i, gpu_name in enumerate(env_info['gpu_names']):
            logger.info(f"  GPU {i}: {gpu_name}")
    
    if 'environment_variables' in env_info:
        logger.info("  Environment Variables:")
        for var, value in env_info['environment_variables'].items():
            logger.info(f"    {var}: {value}")


def validate_reproducibility(
    seed: int,
    num_iterations: int = 10,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that operations are reproducible with the given seed.
    
    Args:
        seed: Random seed to test
        num_iterations: Number of iterations to test
        tolerance: Tolerance for floating point comparisons
        
    Returns:
        True if operations are reproducible, False otherwise
    """
    logger.info(f"Validating reproducibility with seed {seed}")
    
    # Test random number generation reproducibility
    results = []
    
    for i in range(num_iterations):
        set_seed(seed, deterministic=True)
        
        # Generate some random numbers
        python_rand = random.random()
        numpy_rand = np.random.random()
        torch_rand = torch.rand(1).item()
        
        # Create a small tensor operation
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        result = torch.mm(x, y).sum().item()
        
        results.append((python_rand, numpy_rand, torch_rand, result))
    
    # Check if all results are identical
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        for j, (val1, val2) in enumerate(zip(first_result, result)):
            if abs(val1 - val2) > tolerance:
                logger.warning(
                    f"Reproducibility validation failed at iteration {i}, "
                    f"component {j}: {val1} != {val2} (diff: {abs(val1 - val2)})"
                )
                return False
    
    logger.info("Reproducibility validation passed")
    return True


class ReproducibilityContext:
    """Context manager for reproducible operations."""
    
    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = {}
    
    def __enter__(self):
        # Save original state
        self.original_state = {
            'python_state': random.getstate(),
            'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }
        
        if torch.cuda.is_available():
            self.original_state['cuda_state'] = torch.cuda.get_rng_state()
        
        # Set reproducible state
        set_seed(self.seed, self.deterministic)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        random.setstate(self.original_state['python_state'])
        np.random.set_state(self.original_state['numpy_state'])
        torch.set_rng_state(self.original_state['torch_state'])
        
        if torch.cuda.is_available() and 'cuda_state' in self.original_state:
            torch.cuda.set_rng_state(self.original_state['cuda_state'])
        
        torch.backends.cudnn.deterministic = self.original_state['cudnn_deterministic']
        torch.backends.cudnn.benchmark = self.original_state['cudnn_benchmark']


def setup_reproducible_training(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """
    Setup reproducible training environment and return environment info.
    
    Args:
        seed: Random seed
        deterministic: Whether to enable deterministic operations
        
    Returns:
        Dictionary containing environment information
    """
    # Set seed and deterministic behavior
    set_seed(seed, deterministic)
    
    # Log environment information
    log_environment_info()
    
    # Validate reproducibility
    is_reproducible = validate_reproducibility(seed)
    if not is_reproducible:
        logger.warning("Reproducibility validation failed!")
    
    # Return environment info for logging
    env_info = get_environment_info()
    env_info['seed'] = seed
    env_info['deterministic'] = deterministic
    env_info['reproducible'] = is_reproducible
    
    return env_info