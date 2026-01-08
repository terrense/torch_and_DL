# Requirements Document

## Introduction

This specification defines a complete repository designed for "from-scratch" machine learning implementation experience. The repository contains two independent, peer-level subprojects that demonstrate building ML models without relying on prebuilt model libraries or external toolkits. Each subproject implements complete training pipelines, evaluation systems, and inference capabilities using only fundamental dependencies.

## Glossary

- **From-Scratch Implementation**: Building all model components, training loops, and utilities using only basic PyTorch operations without high-level abstractions
- **Peer-Level Subprojects**: Two independent projects at the same directory level that share no code
- **U-Net Transformer Segmentation**: Computer vision project combining U-Net architecture with transformer bottleneck for image segmentation
- **Paraformer ASR**: Automatic speech recognition project implementing a Paraformer-style sequence-to-sequence model
- **Toy Dataset**: Synthetic data generator that enables end-to-end training without external datasets
- **Tensor Contracts**: Explicit documentation of tensor shapes, dtypes, and value ranges for each module
- **Experiment Management**: System for configuration management, logging, checkpointing, and result aggregation

## Requirements

### Requirement 1

**User Story:** As a machine learning engineer, I want a repository with two independent subprojects, so that I can learn different ML architectures without code dependencies between projects.

#### Acceptance Criteria

1. THE Repository SHALL contain exactly two peer-level subprojects in separate directories
2. THE Repository SHALL ensure no shared code exists between the two subprojects
3. THE Repository SHALL provide independent dependency management for each subproject
4. THE Repository SHALL include a root README explaining both subprojects and their purposes
5. THE Repository SHALL enable running each subproject independently without the other

### Requirement 2

**User Story:** As a developer learning ML fundamentals, I want all model components implemented from scratch, so that I understand the underlying mathematics and operations.

#### Acceptance Criteria

1. THE Repository SHALL prohibit use of prebuilt model libraries including FunASR, ModelScope, HuggingFace transformers, torchaudio pipelines, mmseg, mmdet, detectron2, lightning, and accelerate
2. THE Repository SHALL limit dependencies to Python 3.10+, PyTorch (tensor + autograd), numpy, pyyaml, tqdm, matplotlib, optional torchvision for I/O, and optional FastAPI + uvicorn
3. THE Repository SHALL implement multi-head attention explicitly using PyTorch operations without torch.nn.MultiheadAttention
4. THE Repository SHALL provide from-scratch implementations of all model blocks, attention mechanisms, losses, training loops, metrics, logging, and checkpointing systems
5. THE Repository SHALL include complete toy data generation for end-to-end training without external datasets

### Requirement 3

**User Story:** As a computer vision practitioner, I want a U-Net transformer segmentation project, so that I can learn hybrid CNN-transformer architectures for image segmentation.

#### Acceptance Criteria

1. THE U-Net Transformer Segmentation Project SHALL implement a pure U-Net baseline variant
2. THE U-Net Transformer Segmentation Project SHALL implement a U-Net + Transformer bottleneck hybrid variant
3. THE U-Net Transformer Segmentation Project SHALL provide explicit tensor reshaping between [B,C,H,W] and [B,T,D] formats
4. THE U-Net Transformer Segmentation Project SHALL include synthetic segmentation dataset with shapes, noise, and occlusion
5. THE U-Net Transformer Segmentation Project SHALL implement Dice and IoU metrics from scratch

### Requirement 4

**User Story:** As a speech recognition researcher, I want a Paraformer-style ASR project, so that I can understand sequence-to-sequence models for speech processing.

#### Acceptance Criteria

1. THE Paraformer ASR Project SHALL implement an encoder using transformer/conformer-like architecture from scratch
2. THE Paraformer ASR Project SHALL include a predictor module for token alignment estimation
3. THE Paraformer ASR Project SHALL provide a decoder/refiner producing token logits
4. THE Paraformer ASR Project SHALL implement greedy decoding from logits
5. THE Paraformer ASR Project SHALL generate synthetic speech-like features correlated to token sequences

### Requirement 5

**User Story:** As a researcher conducting experiments, I want comprehensive experiment management, so that I can track, reproduce, and compare different model configurations.

#### Acceptance Criteria

1. THE Repository SHALL provide YAML configuration system with dataclass parsing
2. THE Repository SHALL implement seed control and deterministic training toggles
3. THE Repository SHALL log results to CSV and JSON formats with unique run identifiers
4. THE Repository SHALL include automatic summary scripts that aggregate results across runs
5. THE Repository SHALL create timestamped run directories with config, logs, metrics, and checkpoints

### Requirement 6

**User Story:** As a software engineer, I want strict tensor contracts and validation, so that I can debug shape mismatches and data flow issues quickly.

#### Acceptance Criteria

1. THE Repository SHALL document explicit tensor shape, dtype, and range contracts for each module
2. THE Repository SHALL provide runtime assertions with helpful error messages for tensor validation
3. THE Repository SHALL include NaN and infinity detection in losses and gradients
4. THE Repository SHALL implement shape assertion utilities with descriptive failure messages
5. THE Repository SHALL maintain contract documentation tables for all key functions and modules

### Requirement 7

**User Story:** As a quality assurance engineer, I want comprehensive testing coverage, so that I can verify model correctness and catch regressions.

#### Acceptance Criteria

1. THE Repository SHALL include unit tests for tokenizer, dataset, and model shape validation
2. THE Repository SHALL provide smoke tests that run 30-100 training steps on toy data
3. THE Repository SHALL verify loss decrease during smoke test training
4. THE Repository SHALL test tensor contracts and data pipeline functionality
5. THE Repository SHALL include CLI entrypoint testing for train, eval, and inference commands

### Requirement 8

**User Story:** As a developer following industry standards, I want well-organized project structure, so that I can navigate and maintain the codebase effectively.

#### Acceptance Criteria

1. THE Repository SHALL organize each subproject with configs/, src/, tests/, scripts/, and docs/ directories
2. THE Repository SHALL provide CLI entrypoints for train, eval, and inference operations
3. THE Repository SHALL include comprehensive documentation with SPEC.md, CONTRACTS.md, and ABLATIONS.md files
4. THE Repository SHALL implement model registry pattern for configuration-based model building
5. THE Repository SHALL provide shell scripts for common operations and example workflows

### Requirement 9

**User Story:** As a machine learning practitioner, I want complete file contents without placeholders, so that I can immediately run and experiment with the implementations.

#### Acceptance Criteria

1. THE Repository SHALL provide complete implementation of all essential files without TODO placeholders
2. THE Repository SHALL include working training loops with optimizer, scheduler, and gradient clipping options
3. THE Repository SHALL implement complete data loading pipelines with proper batching and masking
4. THE Repository SHALL provide functional inference scripts that produce meaningful outputs
5. THE Repository SHALL include all utility functions for logging, checkpointing, and result aggregation

### Requirement 10

**User Story:** As a learner exploring ML architectures, I want clear documentation and learning paths, so that I can understand the implementation progression and modify components systematically.

#### Acceptance Criteria

1. THE Repository SHALL provide a learning map with recommended code reading order
2. THE Repository SHALL explain how to replace toy datasets with real datasets
3. THE Repository SHALL document where and how to modify architectures for experimentation
4. THE Repository SHALL include troubleshooting guides for common shape and mask bugs
5. THE Repository SHALL provide exact commands for training, evaluation, inference, and result comparison