"""Tensor validation and assertion utilities."""

import re
import logging
from typing import Union, Optional, Tuple, List, Any

import torch
import numpy as np


logger = logging.getLogger(__name__)


class TensorValidationError(Exception):
    """Custom exception for tensor validation errors."""
    pass


def parse_shape_pattern(pattern: str) -> List[Union[str, int]]:
    """
    Parse shape pattern string into list of dimension specifications.
    
    Args:
        pattern: Shape pattern like 'B,C,H,W' or 'B,3,256,256'
        
    Returns:
        List of dimension specifications (strings for variables, ints for fixed)
    """
    # Remove whitespace and split by comma
    dims = [dim.strip() for dim in pattern.split(',')]
    
    parsed_dims = []
    for dim in dims:
        # Try to parse as integer
        try:
            parsed_dims.append(int(dim))
        except ValueError:
            # Keep as string variable
            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', dim):
                raise ValueError(f"Invalid dimension name: {dim}")
            parsed_dims.append(dim)
    
    return parsed_dims


def assert_shape(
    tensor: torch.Tensor,
    expected_pattern: str,
    name: str = "tensor",
    allow_batch_dim: bool = True
) -> None:
    """
    Assert tensor matches expected shape pattern.
    
    Args:
        tensor: Input tensor to validate
        expected_pattern: Expected shape pattern like 'B,C,H,W' or 'B,3,256,256'
        name: Name of tensor for error messages
        allow_batch_dim: Whether to allow flexible batch dimension
        
    Raises:
        TensorValidationError: If shape doesn't match pattern
    """
    if not isinstance(tensor, torch.Tensor):
        raise TensorValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    expected_dims = parse_shape_pattern(expected_pattern)
    actual_shape = list(tensor.shape)
    
    if len(actual_shape) != len(expected_dims):
        raise TensorValidationError(
            f"{name} shape mismatch: expected {len(expected_dims)} dimensions "
            f"({expected_pattern}), got {len(actual_shape)} dimensions {actual_shape}"
        )
    
    # Check each dimension
    for i, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_dims)):
        if isinstance(expected_dim, int):
            # Fixed dimension size
            if actual_dim != expected_dim:
                raise TensorValidationError(
                    f"{name} dimension {i} mismatch: expected {expected_dim}, "
                    f"got {actual_dim}. Full shape: {actual_shape} vs pattern {expected_pattern}"
                )
        elif isinstance(expected_dim, str):
            # Variable dimension - just check it's positive
            if actual_dim <= 0:
                raise TensorValidationError(
                    f"{name} dimension {i} ({expected_dim}) must be positive, "
                    f"got {actual_dim}. Full shape: {actual_shape}"
                )
            # Special case for batch dimension
            if expected_dim.lower() == 'b' and allow_batch_dim and actual_dim == 0:
                raise TensorValidationError(
                    f"{name} batch dimension cannot be 0. Full shape: {actual_shape}"
                )


def assert_dtype(
    tensor: torch.Tensor,
    expected_dtype: Union[torch.dtype, List[torch.dtype]],
    name: str = "tensor"
) -> None:
    """
    Assert tensor has expected dtype.
    
    Args:
        tensor: Input tensor to validate
        expected_dtype: Expected dtype or list of acceptable dtypes
        name: Name of tensor for error messages
        
    Raises:
        TensorValidationError: If dtype doesn't match
    """
    if not isinstance(tensor, torch.Tensor):
        raise TensorValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if isinstance(expected_dtype, list):
        if tensor.dtype not in expected_dtype:
            raise TensorValidationError(
                f"{name} dtype mismatch: expected one of {expected_dtype}, "
                f"got {tensor.dtype}"
            )
    else:
        if tensor.dtype != expected_dtype:
            raise TensorValidationError(
                f"{name} dtype mismatch: expected {expected_dtype}, "
                f"got {tensor.dtype}"
            )


def assert_range(
    tensor: torch.Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "tensor",
    check_all: bool = False
) -> None:
    """
    Assert tensor values are within expected range.
    
    Args:
        tensor: Input tensor to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of tensor for error messages
        check_all: If True, check all values. If False, check min/max only
        
    Raises:
        TensorValidationError: If values are outside range
    """
    if not isinstance(tensor, torch.Tensor):
        raise TensorValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if tensor.numel() == 0:
        return  # Empty tensor is valid
    
    if check_all:
        # Check all values (slower but thorough)
        if min_val is not None:
            if torch.any(tensor < min_val):
                actual_min = tensor.min().item()
                raise TensorValidationError(
                    f"{name} contains values below minimum: "
                    f"expected >= {min_val}, got min = {actual_min}"
                )
        
        if max_val is not None:
            if torch.any(tensor > max_val):
                actual_max = tensor.max().item()
                raise TensorValidationError(
                    f"{name} contains values above maximum: "
                    f"expected <= {max_val}, got max = {actual_max}"
                )
    else:
        # Check min/max only (faster)
        actual_min = tensor.min().item()
        actual_max = tensor.max().item()
        
        if min_val is not None and actual_min < min_val:
            raise TensorValidationError(
                f"{name} minimum value out of range: "
                f"expected >= {min_val}, got {actual_min}"
            )
        
        if max_val is not None and actual_max > max_val:
            raise TensorValidationError(
                f"{name} maximum value out of range: "
                f"expected <= {max_val}, got {actual_max}"
            )


def check_nan_inf(
    tensor: torch.Tensor,
    name: str = "tensor",
    raise_on_error: bool = True
) -> Tuple[bool, bool]:
    """
    Check for NaN and Inf values in tensor.
    
    Args:
        tensor: Input tensor to check
        name: Name of tensor for error messages
        raise_on_error: Whether to raise exception on NaN/Inf detection
        
    Returns:
        Tuple of (has_nan, has_inf)
        
    Raises:
        TensorValidationError: If NaN/Inf detected and raise_on_error=True
    """
    if not isinstance(tensor, torch.Tensor):
        if raise_on_error:
            raise TensorValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        return False, False
    
    if tensor.numel() == 0:
        return False, False  # Empty tensor is valid
    
    # Check for NaN
    has_nan = torch.isnan(tensor).any().item()
    
    # Check for Inf
    has_inf = torch.isinf(tensor).any().item()
    
    if raise_on_error:
        if has_nan:
            nan_count = torch.isnan(tensor).sum().item()
            raise TensorValidationError(
                f"{name} contains {nan_count} NaN values. "
                f"Shape: {tensor.shape}, dtype: {tensor.dtype}"
            )
        
        if has_inf:
            inf_count = torch.isinf(tensor).sum().item()
            pos_inf_count = torch.isposinf(tensor).sum().item()
            neg_inf_count = torch.isneginf(tensor).sum().item()
            raise TensorValidationError(
                f"{name} contains {inf_count} Inf values "
                f"({pos_inf_count} +Inf, {neg_inf_count} -Inf). "
                f"Shape: {tensor.shape}, dtype: {tensor.dtype}"
            )
    
    return has_nan, has_inf


def validate_tensor(
    tensor: torch.Tensor,
    shape_pattern: Optional[str] = None,
    dtype: Optional[Union[torch.dtype, List[torch.dtype]]] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "tensor",
    check_nan_inf_values: bool = True,
    allow_batch_dim: bool = True
) -> None:
    """
    Comprehensive tensor validation.
    
    Args:
        tensor: Input tensor to validate
        shape_pattern: Expected shape pattern like 'B,C,H,W'
        dtype: Expected dtype or list of acceptable dtypes
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of tensor for error messages
        check_nan_inf_values: Whether to check for NaN/Inf values
        allow_batch_dim: Whether to allow flexible batch dimension
        
    Raises:
        TensorValidationError: If any validation fails
    """
    # Basic type check
    if not isinstance(tensor, torch.Tensor):
        raise TensorValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    # Shape validation
    if shape_pattern is not None:
        assert_shape(tensor, shape_pattern, name, allow_batch_dim)
    
    # Dtype validation
    if dtype is not None:
        assert_dtype(tensor, dtype, name)
    
    # Range validation
    if min_val is not None or max_val is not None:
        assert_range(tensor, min_val, max_val, name)
    
    # NaN/Inf validation
    if check_nan_inf_values:
        check_nan_inf(tensor, name, raise_on_error=True)


def log_tensor_stats(
    tensor: torch.Tensor,
    name: str = "tensor",
    log_level: int = logging.INFO
) -> None:
    """
    Log comprehensive tensor statistics.
    
    Args:
        tensor: Input tensor
        name: Name of tensor for logging
        log_level: Logging level to use
    """
    if not isinstance(tensor, torch.Tensor):
        logger.log(log_level, f"{name}: Not a tensor (type: {type(tensor)})")
        return
    
    if tensor.numel() == 0:
        logger.log(log_level, f"{name}: Empty tensor, shape: {tensor.shape}")
        return
    
    # Basic info
    logger.log(log_level, f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    
    # Statistics
    if tensor.dtype.is_floating_point:
        stats = {
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
        }
        
        # Check for special values
        has_nan, has_inf = check_nan_inf(tensor, name, raise_on_error=False)
        if has_nan:
            stats['nan_count'] = torch.isnan(tensor).sum().item()
        if has_inf:
            stats['inf_count'] = torch.isinf(tensor).sum().item()
        
        logger.log(log_level, f"{name} stats: {stats}")
    else:
        # Integer or other dtypes
        logger.log(log_level, f"{name}: min={tensor.min().item()}, max={tensor.max().item()}")


class TensorValidator:
    """Context manager for tensor validation with custom error handling."""
    
    def __init__(self, name: str = "operation", log_errors: bool = True):
        self.name = name
        self.log_errors = log_errors
        self.errors = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is TensorValidationError:
            error_msg = f"Tensor validation failed in {self.name}: {exc_val}"
            if self.log_errors:
                logger.error(error_msg)
            self.errors.append(str(exc_val))
            return False  # Re-raise the exception
        return False
    
    def validate(self, tensor: torch.Tensor, **kwargs) -> bool:
        """
        Validate tensor and collect errors without raising.
        
        Returns:
            True if validation passed, False otherwise
        """
        try:
            validate_tensor(tensor, **kwargs)
            return True
        except TensorValidationError as e:
            self.errors.append(str(e))
            if self.log_errors:
                logger.warning(f"Validation failed in {self.name}: {e}")
            return False