"""Utility modules for the Paraformer ASR project."""

from .reproducibility import (
    set_seed,
    get_environment_info,
    log_environment_info,
    validate_reproducibility,
    ReproducibilityContext,
    setup_reproducible_training,
)

from .tensor_utils import (
    TensorValidationError,
    parse_shape_pattern,
    assert_shape,
    assert_dtype,
    assert_range,
    check_nan_inf,
    validate_tensor,
    log_tensor_stats,
    TensorValidator,
)

from .logging_utils import (
    LogEntry,
    ExperimentLogger,
    MetricsTracker,
    setup_experiment_logging,
)

from .checkpoint import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    # Reproducibility
    'set_seed',
    'get_environment_info', 
    'log_environment_info',
    'validate_reproducibility',
    'ReproducibilityContext',
    'setup_reproducible_training',
    
    # Tensor validation
    'TensorValidationError',
    'parse_shape_pattern',
    'assert_shape',
    'assert_dtype',
    'assert_range',
    'check_nan_inf',
    'validate_tensor',
    'log_tensor_stats',
    'TensorValidator',
    
    # Logging
    'LogEntry',
    'ExperimentLogger',
    'MetricsTracker',
    'setup_experiment_logging',
    
    # Checkpointing
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint',
]