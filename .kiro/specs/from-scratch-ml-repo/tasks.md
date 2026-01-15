# Implementation Plan

- [x] 1. Set up repository structure and root documentation





  - Create root directory structure with both subprojects
  - Write comprehensive root README.md with quickstart commands and learning map
  - Set up independent requirements.txt files for each subproject
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 9.1, 10.1, 10.2_

- [x] 2. Implement core infrastructure utilities





  - [x] 2.1 Create configuration system with YAML and dataclass parsing


    - Implement config loading and validation utilities
    - Create base configuration dataclasses for models and training
    - Add configuration merging and override capabilities
    - _Requirements: 5.1, 8.4_

  - [x] 2.2 Implement seed control and deterministic training utilities


    - Create seed setting functions for Python, NumPy, and PyTorch
    - Add deterministic training toggles and environment logging
    - Implement reproducibility validation utilities
    - _Requirements: 5.2, 6.4_

  - [x] 2.3 Create tensor validation and assertion utilities


    - Implement shape assertion functions with descriptive error messages
    - Add dtype and range validation utilities
    - Create NaN/Inf detection and reporting functions
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 2.4 Implement logging and checkpointing systems


    - Create structured logging utilities for training progress
    - Implement checkpoint save/load functionality with metadata
    - Add experiment tracking with unique run IDs and timestamped directories
    - _Requirements: 5.3, 5.4, 5.5, 9.5_

- [x] 3. Build U-Net Transformer Segmentation project structure




  - [x] 3.1 Set up project directory structure and documentation


    - Create configs/, src/, tests/, scripts/, docs/ directories
    - Write project-specific README.md with exact commands
    - Create SPEC.md, CONTRACTS.md, and ABLATIONS.md documentation files
    - _Requirements: 8.1, 8.3, 10.3, 10.4, 10.5_

  - [x] 3.2 Implement toy shapes dataset generator


    - Create synthetic segmentation dataset with multiple shape types
    - Add configurable noise, blur, occlusion, and class imbalance options
    - Implement proper image and mask generation with alignment
    - _Requirements: 2.5, 3.4, 9.2_

  - [x] 3.3 Create data transforms and loading pipeline


    - Implement from-scratch transforms (resize, normalize, flips) with mask alignment
    - Create folder dataset loader for real image/mask pairs
    - Add batching and collation functions for variable-sized inputs
    - _Requirements: 3.3, 9.3_

- [x] 4. Implement U-Net segmentation models





  - [x] 4.1 Create basic building blocks and layers


    - Implement Conv->Norm->Act blocks with configurable parameters
    - Create downsampling and upsampling blocks with skip connections
    - Add model registry pattern for configuration-based building
    - _Requirements: 2.4, 8.4, 9.2_

  - [x] 4.2 Implement transformer components from scratch


    - Create multi-head attention without torch.nn.MultiheadAttention
    - Implement feed-forward networks with residual connections and layer normalization
    - Add positional encoding for spatial tokens with 2D awareness
    - _Requirements: 2.3, 3.3, 9.2_

  - [x] 4.3 Build pure U-Net baseline model


    - Implement encoder-decoder architecture with skip connections
    - Ensure proper spatial alignment and feature map sizing
    - Add configurable depth and channel dimensions
    - _Requirements: 3.1, 9.2_

  - [x] 4.4 Create U-Net + Transformer hybrid model


    - Implement tensor reshaping between [B,C,H,W] and [B,T,D] formats
    - Integrate transformer layers at bottleneck with clear conversion points
    - Add configuration switches between baseline and hybrid variants
    - _Requirements: 3.2, 3.3, 9.2_

- [x] 5. Implement segmentation training and evaluation systems





  - [x] 5.1 Create loss functions from scratch



    - Implement Dice loss with soft dice coefficient calculation
    - Create combined BCE + Dice loss with configurable weighting
    - Add proper handling of class imbalance and edge cases
    - _Requirements: 2.4, 3.5, 9.2_




  - [x] 5.2 Implement segmentation metrics

    - Create IoU calculation from scratch with per-class support
    - Implement pixel accuracy and per-class dice score metrics


    - Add metric aggregation and reporting utilities
    - _Requirements: 3.5, 9.2_


  - [-] 5.3 Build training loop with comprehensive features

    - Implement training loop with AdamW optimizer and optional scheduler
    - Add gradient clipping, mixed precision, and checkpoint management
    - Include per-epoch metrics logging and results CSV generation
    - _Requirements: 5.3, 5.4, 9.2, 9.5_

  - [x] 5.4 Create evaluation and inference scripts

    - Implement evaluation pipeline with metric calculation and logging
    - Create inference script for single images with visualization
    - Add prediction overlay generation and result saving
    - _Requirements: 8.2, 9.4_

- [x] 6. Build Paraformer ASR project structure





  - [x] 6.1 Set up project directory structure and documentation

    - Create configs/, src/, tests/, scripts/, docs/ directories
    - Write project-specific README.md with exact commands
    - Create SPEC.md, CONTRACTS.md, and ABLATIONS.md documentation files
    - _Requirements: 8.1, 8.3, 10.3, 10.4, 10.5_



  - [x] 6.2 Implement toy sequence-to-sequence dataset





    - Generate synthetic speech-like features correlated to token sequences
    - Create variable-length feature and token sequences with controllable difficulty
    - Add proper padding, masking, and attention mask generation


    - _Requirements: 2.5, 4.5, 9.2_

  - [x] 6.3 Create tokenizer and sequence utilities




    - Implement character-level tokenizer from scratch with vocabulary management
    - Add encode/decode utilities with proper handling of special tokens
    - Create sequence collation functions with padding and mask generation
    - _Requirements: 4.5, 9.2_

- [x] 7. Implement Paraformer ASR model components




  - [x] 7.1 Create transformer layers and attention mechanisms


    - Implement multi-head attention from scratch without torch.nn.MultiheadAttention
    - Create feed-forward networks with residual connections and layer normalization
    - Add attention masking for padded positions and causal attention
    - _Requirements: 2.3, 4.2, 9.2_

  - [x] 7.2 Build encoder stack with clear contracts


    - Implement multi-layer transformer/conformer-style encoder
    - Add bidirectional self-attention with proper masking
    - Ensure clear tensor contracts and shape documentation
    - _Requirements: 4.1, 6.1, 9.2_



  - [x] 7.3 Implement predictor module for alignment estimation





    - Create predictor that estimates token boundaries in feature sequences
    - Add clear documentation of what the predictor outputs and how it's used
    - Implement proper conditioning for decoder processing


    - _Requirements: 4.2, 6.1, 9.2_

  - [x] 7.4 Create decoder/refiner for token generation





    - Implement decoder that produces token logits from encoder features
    - Add attention over encoder outputs with proper masking
    - Integrate predictor signals for improved alignment
    - _Requirements: 4.3, 9.2_

- [x] 8. Implement ASR training and inference systems





  - [x] 8.1 Create sequence loss functions


    - Implement masked cross-entropy for variable-length sequences
    - Add optional auxiliary losses for predictor training
    - Handle padding and sequence length variations properly
    - _Requirements: 2.4, 9.2_

  - [x] 8.2 Build greedy decoding system


    - Implement greedy decoding from logits with proper masking
    - Add tokenizer integration for text generation
    - Create inference pipeline from features to decoded text
    - _Requirements: 4.4, 9.4_

  - [x] 8.3 Implement training loop with sequence-specific features


    - Create training loop with optimizer, scheduler, and gradient clipping
    - Add token accuracy calculation and sequence-level metrics
    - Include comprehensive logging and checkpoint management
    - _Requirements: 5.3, 5.4, 9.2, 9.5_

  - [x] 8.4 Create evaluation and inference scripts


    - Implement evaluation pipeline with token accuracy and loss calculation
    - Create inference script for feature sequences with text output
    - Add optional FastAPI service for JSON feature input and text output
    - _Requirements: 8.2, 9.4_

- [-] 9. Implement comprehensive testing suites



  - [x] 9.1 Create unit tests for both projects


    - Write tokenizer and dataset shape validation tests
    - Test model component shapes and gradient flow
    - Add tensor contract validation and data pipeline correctness tests
    - _Requirements: 7.1, 7.4, 9.2_

  - [x] 9.2 Implement smoke tests for end-to-end training






    - Create smoke tests that run 30-100 training steps on toy data
    - Verify loss decrease during training and proper convergence behavior
    - Test checkpoint save/load functionality and reproducibility
    - _Requirements: 7.2, 7.3, 9.2_

  - [x] 9.3 Add CLI and integration testing





    - Test all CLI entrypoints for train, eval, and inference commands
    - Validate configuration loading and parameter validation
    - Test output file generation and result aggregation
    - _Requirements: 7.5, 8.2_

- [x] 10. Create scripts and automation




  - [x] 10.1 Write shell scripts for common operations


    - Create training scripts for baseline and hybrid variants
    - Add evaluation and inference automation scripts
    - Include result comparison and visualization scripts
    - _Requirements: 8.2, 10.5_

  - [x] 10.2 Implement result aggregation and summary system


    - Create summarize.py script that aggregates results across runs
    - Generate summary.csv and summary.md with experiment comparisons
    - Add visualization of training curves and metric comparisons
    - _Requirements: 5.4, 5.5_

- [-] 11. Complete documentation and final integration



  - [-] 11.1 Finalize all documentation files

    - Complete CONTRACTS.md with comprehensive tensor contract tables
    - Finish ABLATIONS.md with exact commands for variant comparisons
    - Add troubleshooting guides for common shape and mask bugs
    - _Requirements: 6.5, 10.4, 10.5_

  - [ ] 11.2 Validate end-to-end workflows
    - Test complete workflows from installation to result generation
    - Verify all commands in documentation work correctly
    - Ensure reproducibility across different environments
    - _Requirements: 9.1, 9.4, 9.5_

  - [ ] 11.3 Add performance optimization and profiling
    - Implement memory profiling and GPU usage monitoring
    - Add training throughput measurement and optimization
    - Create performance benchmarking scripts
    - _Requirements: 5.2_

  - [ ] 11.4 Create extended documentation and tutorials
    - Write detailed architecture explanation documents
    - Add code walkthrough tutorials for key components
    - Create extension guides for adding new model variants
    - _Requirements: 10.1, 10.3_