# Paraformer ASR Testing Suite / Paraformer ASR 测试套件

## Overview / 概述

This directory contains comprehensive tests for the Paraformer ASR (Automatic Speech Recognition) system. The tests are organized into different categories to ensure code quality, correctness, and reliability.

本目录包含Paraformer ASR（自动语音识别）系统的综合测试。测试分为不同类别，以确保代码质量、正确性和可靠性。

## Test Categories / 测试类别

### 1. Smoke Tests (test_smoke.py) / 冒烟测试

**Purpose / 目的:**
Smoke tests are quick, fundamental tests that verify the core functionality of the training system. They ensure that the basic training loop works correctly before running expensive, long-duration experiments.

冒烟测试是快速、基础的测试，用于验证训练系统的核心功能。它们确保基本训练循环正常工作，然后再运行昂贵的长时间实验。

**What Smoke Tests Verify / 冒烟测试验证内容:**

1. **Loss Decrease / 损失下降**
   - Verifies that the model loss decreases over 30-100 training steps
   - 验证模型损失在30-100个训练步骤中下降
   - Ensures the optimization process is working correctly
   - 确保优化过程正常工作

2. **Checkpoint Save/Load / 检查点保存/加载**
   - Tests that model weights can be saved and restored correctly
   - 测试模型权重能否正确保存和恢复
   - Verifies checkpoint integrity and reproducibility
   - 验证检查点完整性和可复现性

3. **Reproducibility / 可复现性**
   - Ensures training produces identical results with the same random seed
   - 确保使用相同随机种子的训练产生相同结果
   - Critical for scientific experiments and debugging
   - 对科学实验和调试至关重要

4. **Overfitting Capability / 过拟合能力**
   - Tests if the model can overfit on a single batch
   - 测试模型是否能在单个批次上过拟合
   - Sanity check that the model has sufficient capacity to learn
   - 健全性检查，确保模型有足够的学习能力

**Key Functions / 关键函数:**

#### `test_training_loss_decreases()`
```python
def test_training_loss_decreases(toy_dataset, tokenizer, model_config):
    """
    Test that loss decreases over 30-50 training steps.
    测试损失在30-50个训练步骤中下降。
    
    Args / 参数:
        toy_dataset: Small synthetic dataset for testing
                    用于测试的小型合成数据集
        tokenizer: Character tokenizer (CharTokenizer)
                  字符分词器
        model_config: Model configuration dictionary
                     模型配置字典
    
    Verifies / 验证:
        - Initial loss > Final loss (loss decreases)
        - 初始损失 > 最终损失（损失下降）
    """
```

#### `test_checkpoint_save_load()`
```python
def test_checkpoint_save_load(toy_dataset, tokenizer, model_config):
    """
    Test that checkpoints can be saved and loaded correctly.
    测试检查点能否正确保存和加载。
    
    Args / 参数:
        toy_dataset: Small synthetic dataset
                    小型合成数据集
        tokenizer: Character tokenizer
                  字符分词器
        model_config: Model configuration
                     模型配置
    
    Process / 流程:
        1. Train model for a few steps
           训练模型几个步骤
        2. Save checkpoint to disk
           将检查点保存到磁盘
        3. Load checkpoint into new model
           将检查点加载到新模型
        4. Verify weights are identical
           验证权重相同
    
    Output / 输出:
        Assertion passes if weights match exactly
        如果权重完全匹配则断言通过
    """
```

#### `test_reproducibility()`
```python
def test_reproducibility(toy_dataset, tokenizer, model_config):
    """
    Test that training is reproducible with same seed.
    测试使用相同种子的训练是否可复现。
    
    Args / 参数:
        toy_dataset: Small synthetic dataset
                    小型合成数据集
        tokenizer: Character tokenizer
                  字符分词器
        model_config: Model configuration
                     模型配置
    
    Process / 流程:
        1. Train model with seed=42 for N steps
           使用seed=42训练模型N步
        2. Train another model with seed=42 for N steps
           使用seed=42训练另一个模型N步
        3. Compare losses at each step
           比较每步的损失
    
    Verifies / 验证:
        - Losses are identical (within tolerance)
        - 损失相同（在容差范围内）
    """
```

#### `test_overfitting_on_single_batch()`
```python
def test_overfitting_on_single_batch(toy_dataset, tokenizer, model_config):
    """
    Test that model can overfit on a single batch (sanity check).
    测试模型能否在单个批次上过拟合（健全性检查）。
    
    Args / 参数:
        toy_dataset: Small synthetic dataset
                    小型合成数据集
        tokenizer: Character tokenizer
                  字符分词器
        model_config: Model configuration
                     模型配置
    
    Process / 流程:
        1. Get a single batch of data
           获取单个数据批次
        2. Train on same batch for 100 iterations
           在同一批次上训练100次迭代
        3. Verify loss decreases significantly
           验证损失显著下降
    
    Expected / 期望:
        Final loss < Initial loss * 0.5
        最终损失 < 初始损失 * 0.5
    
    Purpose / 目的:
        If model cannot overfit on single batch, it indicates:
        如果模型无法在单个批次上过拟合，表明：
        - Model architecture issues
          模型架构问题
        - Learning rate too low
          学习率过低
        - Gradient flow problems
          梯度流问题
    """
```

### 2. Model Tests (test_models.py) / 模型测试

**Purpose / 目的:**
Tests individual model components (encoder, decoder, predictor) to ensure correct architecture and tensor shapes.

测试各个模型组件（编码器、解码器、预测器），确保架构正确和张量形状正确。

**Key Tests / 关键测试:**
- Encoder forward pass and output shapes
  编码器前向传播和输出形状
- Decoder forward pass with attention
  带注意力机制的解码器前向传播
- Predictor boundary detection
  预测器边界检测
- Full Paraformer model integration
  完整Paraformer模型集成

### 3. Data Tests (test_data.py) / 数据测试

**Purpose / 目的:**
Tests data loading, preprocessing, and augmentation pipelines.

测试数据加载、预处理和增强管道。

**Key Tests / 关键测试:**
- Toy dataset generation
  玩具数据集生成
- Tokenizer encoding/decoding
  分词器编码/解码
- Batch collation and padding
  批次整理和填充
- Data augmentation correctness
  数据增强正确性

## Running Tests / 运行测试

### Run All Tests / 运行所有测试
```bash
cd paraformer_asr
python -m pytest tests/ -v
```

### Run Smoke Tests Only / 仅运行冒烟测试
```bash
python -m pytest tests/test_smoke.py -v
```

### Run Specific Test / 运行特定测试
```bash
python -m pytest tests/test_smoke.py::test_training_loss_decreases -v
```

### Run with Output / 运行并显示输出
```bash
python -m pytest tests/test_smoke.py -v -s
```

## Test Data / 测试数据

### Toy Dataset / 玩具数据集

The smoke tests use `ToySeq2SeqDataset`, which generates synthetic sequence-to-sequence data:

冒烟测试使用`ToySeq2SeqDataset`，它生成合成的序列到序列数据：

**Parameters / 参数:**
- `num_samples`: Number of samples (default: 40)
  样本数量（默认：40）
- `vocab_size`: Vocabulary size (default: 50)
  词汇表大小（默认：50）
- `feature_dim`: Feature dimension (default: 40)
  特征维度（默认：40）
- `max_feat_len`: Maximum feature length (default: 80)
  最大特征长度（默认：80）
- `max_token_len`: Maximum token length (default: 20)
  最大标记长度（默认：20）

**Data Format / 数据格式:**
```python
{
    'features': torch.Tensor,      # Shape: [batch, time, feature_dim]
                                   # 形状：[批次, 时间, 特征维度]
    'feature_lengths': torch.Tensor,  # Shape: [batch]
                                      # 形状：[批次]
    'tokens': torch.Tensor,        # Shape: [batch, seq_len]
                                   # 形状：[批次, 序列长度]
    'token_lengths': torch.Tensor  # Shape: [batch]
                                   # 形状：[批次]
}
```

## Test Configuration / 测试配置

### Model Configuration / 模型配置
```python
model_config = {
    'input_dim': 40,           # Input feature dimension / 输入特征维度
    'hidden_dim': 128,         # Hidden layer dimension / 隐藏层维度
    'vocab_size': 50,          # Vocabulary size / 词汇表大小
    'encoder_layers': 2,       # Number of encoder layers / 编码器层数
    'encoder_heads': 4,        # Number of attention heads / 注意力头数
    'decoder_layers': 1,       # Number of decoder layers / 解码器层数
    'decoder_heads': 4,        # Number of attention heads / 注意力头数
    'dropout': 0.1,            # Dropout rate / Dropout率
    'predictor_hidden_dim': 64 # Predictor hidden dimension / 预测器隐藏维度
}
```

## Troubleshooting / 故障排除

### Common Issues / 常见问题

1. **Import Errors / 导入错误**
   - Ensure you're running from the project root
     确保从项目根目录运行
   - Check that all dependencies are installed
     检查所有依赖项是否已安装

2. **CUDA Out of Memory / CUDA内存不足**
   - Reduce batch size in tests
     减少测试中的批次大小
   - Tests will automatically use CPU if CUDA unavailable
     如果CUDA不可用，测试将自动使用CPU

3. **Reproducibility Failures / 可复现性失败**
   - Some operations are non-deterministic on GPU
     某些操作在GPU上是非确定性的
   - Tests use tolerance for floating point comparisons
     测试使用容差进行浮点比较

## Best Practices / 最佳实践

1. **Run smoke tests before long experiments**
   在长时间实验之前运行冒烟测试
   - Catches basic issues early
     及早发现基本问题
   - Saves computational resources
     节省计算资源

2. **Use small datasets for testing**
   使用小数据集进行测试
   - Tests should complete in seconds, not minutes
     测试应在几秒钟内完成，而不是几分钟
   - Focus on correctness, not performance
     关注正确性，而不是性能

3. **Test edge cases**
   测试边缘情况
   - Empty sequences
     空序列
   - Single-element batches
     单元素批次
   - Maximum length sequences
     最大长度序列

## References / 参考资料

- PyTorch Testing: https://pytorch.org/docs/stable/testing.html
- Pytest Documentation: https://docs.pytest.org/
- Deep Learning Testing Best Practices: https://karpathy.github.io/2019/04/25/recipe/
