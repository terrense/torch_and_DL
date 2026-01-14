"""Logging utilities for structured training progress tracking."""

import os
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict

import torch


def setup_logger(
    name: str = "logger",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup a simple logger with console and optional file output.
    设置一个简单的日志记录器，支持控制台和可选的文件输出。

    Args / 参数:
        name (str): Logger name / 日志记录器名称
        log_file (Optional[Union[str, Path]]): Optional path to log file
                                                可选的日志文件路径
        level (int): Logging level (logging.INFO, logging.DEBUG, etc.)
                    日志级别（logging.INFO, logging.DEBUG等）
        console_output (bool): Whether to output to console
                              是否输出到控制台

    Returns / 返回:
        logging.Logger: Configured logger instance
                       配置好的日志记录器实例

    Example / 示例:
        >>> logger = setup_logger("my_logger", "train.log")
        >>> logger.info("Training started")
        2024-01-14 10:30:00 - my_logger - INFO - Training started
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, float],
    prefix: str = ""
) -> None:
    """
    Log metrics dictionary to logger.
    将指标字典记录到日志记录器。

    Args / 参数:
        logger (logging.Logger): Logger instance / 日志记录器实例
        metrics (Dict[str, float]): Dictionary of metric name -> value
                                   指标名称 -> 值的字典
                                   Example: {'loss': 0.5, 'accuracy': 0.95}
        prefix (str): Optional prefix for log message
                     日志消息的可选前缀

    Example / 示例:
        >>> logger = setup_logger("train")
        >>> metrics = {'loss': 0.5, 'accuracy': 0.95}
        >>> log_metrics(logger, metrics, "Epoch 1")
        Epoch 1: loss=0.500000, accuracy=0.950000
    """
    metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
    if prefix:
        logger.info(f"{prefix}: {metrics_str}")
    else:
        logger.info(metrics_str)


@dataclass
class LogEntry:
    """Structured log entry for training metrics."""
    timestamp: str
    epoch: int
    step: int
    phase: str  # 'train', 'val', 'test'
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ExperimentLogger:
    """Comprehensive experiment logging with multiple output formats."""
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path] = "runs",
        log_level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{experiment_name}"
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging paths
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize log files
        self.json_log_path = self.log_dir / "metrics.json"
        self.csv_log_path = self.log_dir / "results.csv"
        self.text_log_path = self.log_dir / "train.log"
        
        # Setup Python logger
        self.logger = self._setup_logger(log_level, console_output, file_output)
        
        # Initialize metric storage
        self.log_entries: List[LogEntry] = []
        self.csv_fieldnames: Optional[List[str]] = None
        
        # Log experiment start
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Output directory: {self.run_dir}")
    
    def _setup_logger(
        self,
        log_level: int,
        console_output: bool,
        file_output: bool
    ) -> logging.Logger:
        """Setup Python logger with console and file handlers."""
        logger = logging.getLogger(f"experiment_{self.run_id}")
        logger.setLevel(log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            file_handler = logging.FileHandler(self.text_log_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        epoch: int,
        step: int,
        phase: str = "train"
    ) -> None:
        """
        Log metrics for current step.
        
        Args:
            metrics: Dictionary of metric name -> value
            epoch: Current epoch number
            step: Current step number
            phase: Training phase ('train', 'val', 'test')
        """
        timestamp = datetime.now().isoformat()
        
        # Create log entry
        entry = LogEntry(
            timestamp=timestamp,
            epoch=epoch,
            step=step,
            phase=phase,
            metrics=metrics
        )
        
        self.log_entries.append(entry)
        
        # Log to console/file
        metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch:03d}, Step {step:05d} [{phase}]: {metrics_str}")
        
        # Save to JSON (append mode)
        self._save_json_log(entry)
        
        # Save to CSV
        self._save_csv_log(entry)
    
    def _save_json_log(self, entry: LogEntry) -> None:
        """Save log entry to JSON file."""
        # Read existing entries if file exists
        entries = []
        if self.json_log_path.exists():
            try:
                with open(self.json_log_path, 'r') as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                entries = []
        
        # Add new entry
        entries.append(entry.to_dict())
        
        # Write back to file
        with open(self.json_log_path, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def _save_csv_log(self, entry: LogEntry) -> None:
        """Save log entry to CSV file."""
        # Flatten entry for CSV
        row = {
            'timestamp': entry.timestamp,
            'epoch': entry.epoch,
            'step': entry.step,
            'phase': entry.phase,
        }
        row.update(entry.metrics)
        
        # Initialize CSV file if needed
        file_exists = self.csv_log_path.exists()
        if not file_exists or self.csv_fieldnames is None:
            self.csv_fieldnames = list(row.keys())
        
        # Write to CSV
        with open(self.csv_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
    
    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters and configuration."""
        config_path = self.run_dir / "config.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Saved configuration to {config_path}")
    
    def log_model_summary(self, model: torch.nn.Module) -> None:
        """Log model architecture summary."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model Summary:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Save detailed model info
        model_info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_str': str(model),
        }
        
        model_info_path = self.run_dir / "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def get_run_dir(self) -> Path:
        """Get the run directory path."""
        return self.run_dir
    
    def get_checkpoint_dir(self) -> Path:
        """Get the checkpoint directory path."""
        return self.checkpoint_dir
    
    def get_log_dir(self) -> Path:
        """Get the log directory path."""
        return self.log_dir
    
    def close(self) -> None:
        """Close logger and finalize experiment."""
        self.logger.info(f"Experiment completed. Total log entries: {len(self.log_entries)}")
        
        # Close file handlers
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()


class MetricsTracker:
    """Simple metrics tracking for running averages."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.reset()
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics with new values."""
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(float(value))
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        averages = {}
        for name, values in self.metrics.items():
            if values:
                averages[name] = sum(values) / len(values)
        return averages
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest values for all metrics."""
        latest = {}
        for name, values in self.metrics.items():
            if values:
                latest[name] = values[-1]
        return latest
    
    def reset(self) -> None:
        """Reset all metrics."""
        for name in self.metrics:
            self.metrics[name] = []
    
    def get_count(self) -> int:
        """Get number of updates."""
        if not self.metrics:
            return 0
        return len(next(iter(self.metrics.values())))


def setup_experiment_logging(
    experiment_name: str,
    config: Dict[str, Any],
    output_dir: Union[str, Path] = "runs",
    log_level: int = logging.INFO
) -> ExperimentLogger:
    """
    Setup comprehensive experiment logging.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration dictionary to log
        output_dir: Base output directory
        log_level: Logging level
        
    Returns:
        ExperimentLogger instance
    """
    logger = ExperimentLogger(
        experiment_name=experiment_name,
        output_dir=output_dir,
        log_level=log_level
    )
    
    # Log configuration
    logger.log_hyperparameters(config)
    
    return logger