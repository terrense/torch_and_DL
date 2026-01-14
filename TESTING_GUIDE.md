# Testing Guide for ML Projects / ML项目测试指南

## Table of Contents / 目录

1. [What is Testing? / 什么是测试？](#what-is-testing)
2. [Types of Tests / 测试类型](#types-of-tests)
3. [Smoke Tests Explained / 冒烟测试详解](#smoke-tests-explained)
4. [Running Tests / 运行测试](#running-tests)
5. [Writing New Tests / 编写新测试](#writing-new-tests)
6. [Best Practices / 最佳实践](#best-practices)

---

## What is Testing? / 什么是测试？

**Testing** is the process of verifying that your code works correctly. In machine learning, testing is especially important because:

**测试**是验证代码正确工作的过程。在机器学习中，测试尤其重要，因为：

1. **Complexity / 复杂性**: ML systems have many components (data, model, training, evaluation)
   ML系统有许多组件（数据、模型、训练、评估）

2. **Non-determinism / 非确定性**: Random initialization and stochastic training can hide bugs
   随机初始化和随机训练可能隐藏错误

3. **Silent Failures / 静默失败**: ML models can "work" but perform poorly without obvious errors
   ML模型可能"工作"但性能不佳，没有明显错误

4. **Expensive Experiments / 昂贵的实验**: Training can take hours/days, so catching bugs early saves time
   训练可能需要数小时/数天，因此及早发现错误可以节省时间

---

## Types of Tests / 测试类型

### 1. Unit Tests / 单元测试

**Purpose / 目的**: Test individual functions or classes in isolation
测试单独的函数或类

**Example / 示例**:
```python
def test_tokenizer_encode():
    """Test that tokenizer correctly encodes text"""
    tokenizer = CharTokenizer(vocab_size=100)
    text = "hello"
    tokens = tokenizer.encode(text)
    assert len(tokens) == len(text)
    assert all(0 <= t < 100 for t in tokens)
```

**When to Use / 何时使用**:
- Testing data preprocessing functions
  测试数据预处理函数
- Testing model components (layers, attention, etc.)
  测试模型组件（层、注意力等）
- Testing utility functions
  测试实用函数

### 2. Integration Tests / 集成测试

**Purpose / 目的**: Test how multiple components work together
测试多个组件如何协同工作

**Example / 示例**:
```python
def test_model_forward_pass():
    """Test that model can process a batch"""
    model = ParaformerASR(...)
    features = torch.randn(4, 100, 80)  # batch of 4
    outputs = model(features)
    assert outputs['logits'].shape == (4, seq_len, vocab_size)
```

**When to Use / 何时使用**:
- Testing model + loss function
  测试模型 + 损失函数
- Testing data loader + model
  测试数据加载器 + 模型
- Testing training loop components
  测试训练循环组件

### 3. Smoke Tests / 冒烟测试

**Purpose / 目的**: Quick tests to verify basic functionality works
快速测试以验证基本功能正常工作

**Example / 示例**:
```python
def test_training_loss_decreases():
    """Test that model can learn (loss decreases)"""
    # Train for 50 steps
    # Verify: final_loss < initial_loss
```

**When to Use / 何时使用**:
- Before starting long training runs
  在开始长时间训练之前
- After changing model architecture
  更改模型架构后
- After updating dependencies
  更新依赖项后

### 4. End-to-End Tests / 端到端测试

**Purpose / 目的**: Test the entire pipeline from data to predictions
测试从数据到预测的整个管道

**Example / 示例**:
```python
def test_full_training_pipeline():
    """Test complete training workflow"""
    # Load data
    # Train model
    # Evaluate model
    # Save checkpoint
    # Load checkpoint
    # Make predictions
```

**When to Use / 何时使用**:
- Before releasing code
  发布代码之前
- Testing deployment pipelines
  测试部署管道
- Validating full workflows
  验证完整工作流

---

## Smoke Tests Explained / 冒烟测试详解

### What is a Smoke Test? / 什么是冒烟测试？

The term "smoke test" comes from hardware testing: when you power on a circuit board, if it starts smoking, there's a serious problem! 

"冒烟测试"一词来自硬件测试：当你给电路板通电时，如果它开始冒烟，就有严重问题！

In software, **smoke tests** are quick, basic tests that check if the system is fundamentally working.

在软件中，**冒烟测试**是快速、基本的测试，检查系统是否基本正常工作。

### Why Smoke Tests for ML? / 为什么ML需要冒烟测试？

Machine learning training can take hours or days. Smoke tests help you:

机器学习训练可能需要数小时或数天。冒烟测试帮助你：

1. **Catch bugs early / 及早发现错误**
   - Find issues in minutes, not hours
     在几分钟内发现问题，而不是几小时

2. **Validate changes / 验证更改**
   - Ensure code changes don't break training
     确保代码更改不会破坏训练

3. **Test on small data / 在小数据上测试**
   - Use toy datasets for fast iteration
     使用玩具数据集进行快速迭代

4. **Verify learning capability / 验证学习能力**
   - Confirm model can actually learn
     确认模型确实可以学习

### What Do Smoke Tests Check? / 冒烟测试检查什么？

#### 1. Loss Decrease / 损失下降

**What it tests / 测试内容**:
- Model can perform forward pass
  模型可以执行前向传播
- Loss function works correctly
  损失函数正常工作
- Gradients flow through model
  梯度流经模型
- Optimizer updates weights
  优化器更新权重
- Loss decreases over time
  损失随时间下降

**Why it matters / 为什么重要**:
If loss doesn't decrease, something is fundamentally broken:
如果损失不下降，说明有根本性问题：
- Model architecture issue
  模型架构问题
- Learning rate too high/low
  学习率过高/过低
- Gradient vanishing/exploding
  梯度消失/爆炸
- Data preprocessing error
  数据预处理错误

**Code Example / 代码示例**:
```python
def test_training_loss_decreases():
    # Train for 50 steps
    initial_losses = []  # First 5 steps
    final_losses = []    # Last 5 steps
    
    for step in range(50):
        loss = train_step(model, batch)
        if step < 5:
            initial_losses.append(loss)
        elif step >= 45:
            final_losses.append(loss)
    
    avg_initial = mean(initial_losses)
    avg_final = mean(final_losses)
    
    # Loss should decrease
    assert avg_final < avg_initial
```

#### 2. Checkpoint Save/Load / 检查点保存/加载

**What it tests / 测试内容**:
- Model state can be saved to disk
  模型状态可以保存到磁盘
- Saved state can be loaded back
  保存的状态可以加载回来
- Loaded weights match original weights
  加载的权重与原始权重匹配
- Training can resume from checkpoint
  训练可以从检查点恢复

**Why it matters / 为什么重要**:
Checkpointing is critical for:
检查点对以下方面至关重要：
- Long training runs (save progress)
  长时间训练（保存进度）
- Recovering from crashes
  从崩溃中恢复
- Model deployment
  模型部署
- Experiment reproducibility
  实验可复现性

**Code Example / 代码示例**:
```python
def test_checkpoint_save_load():
    # Train model
    model1 = train_model(steps=10)
    
    # Save checkpoint
    save_checkpoint(model1, "checkpoint.pt")
    
    # Load into new model
    model2 = create_model()
    load_checkpoint(model2, "checkpoint.pt")
    
    # Verify weights match
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
```

#### 3. Reproducibility / 可复现性

**What it tests / 测试内容**:
- Same seed produces same results
  相同种子产生相同结果
- Random number generators work correctly
  随机数生成器正常工作
- Deterministic operations are enabled
  启用确定性操作
- Training is repeatable
  训练可重复

**Why it matters / 为什么重要**:
Reproducibility is essential for:
可复现性对以下方面至关重要：
- Scientific experiments
  科学实验
- Debugging (reproduce bugs)
  调试（重现错误）
- Comparing models fairly
  公平比较模型
- Publishing results
  发布结果

**Code Example / 代码示例**:
```python
def test_reproducibility():
    # Train with seed=42
    set_seed(42)
    losses1 = train_model(steps=10)
    
    # Train again with seed=42
    set_seed(42)
    losses2 = train_model(steps=10)
    
    # Results should be identical
    for l1, l2 in zip(losses1, losses2):
        assert abs(l1 - l2) < 1e-6
```

#### 4. Overfitting on Single Batch / 单批次过拟合

**What it tests / 测试内容**:
- Model has sufficient capacity
  模型有足够的容量
- Model can memorize data
  模型可以记忆数据
- Optimization works correctly
  优化正常工作
- No gradient flow issues
  没有梯度流问题

**Why it matters / 为什么重要**:
If a model can't overfit on a single batch, it indicates:
如果模型无法在单个批次上过拟合，表明：
- Model is too small
  模型太小
- Learning rate is too low
  学习率太低
- Gradients are not flowing
  梯度没有流动
- Loss function is inappropriate
  损失函数不合适

**Code Example / 代码示例**:
```python
def test_overfitting_on_single_batch():
    # Get one batch
    batch = next(iter(dataloader))
    
    # Train on same batch 100 times
    losses = []
    for i in range(100):
        loss = train_step(model, batch)
        losses.append(loss)
    
    # Loss should decrease significantly
    initial_loss = mean(losses[:10])
    final_loss = mean(losses[-10:])
    
    assert final_loss < initial_loss * 0.5
```

---

## Running Tests / 运行测试

### Prerequisites / 先决条件

```bash
# Install pytest
pip install pytest

# Install project dependencies
pip install -r requirements.txt
```

### Basic Commands / 基本命令

```bash
# Run all tests in a directory
# 运行目录中的所有测试
pytest tests/

# Run specific test file
# 运行特定测试文件
pytest tests/test_smoke.py

# Run specific test function
# 运行特定测试函数
pytest tests/test_smoke.py::test_training_loss_decreases

# Run with verbose output
# 运行并显示详细输出
pytest tests/ -v

# Run with print statements visible
# 运行并显示print语句
pytest tests/ -v -s

# Stop at first failure
# 在第一次失败时停止
pytest tests/ -x

# Run tests matching pattern
# 运行匹配模式的测试
pytest tests/ -k "loss"
```

### Project-Specific Commands / 项目特定命令

#### Paraformer ASR
```bash
cd paraformer_asr

# Run all tests
python -m pytest tests/ -v

# Run smoke tests only
python -m pytest tests/test_smoke.py -v

# Run with output
python -m pytest tests/test_smoke.py -v -s
```

#### U-Net Transformer Segmentation
```bash
cd unet_transformer_seg

# Run all tests
python -m pytest tests/ -v

# Run smoke tests only
python -m pytest tests/test_smoke.py -v

# Run specific test
python -m pytest tests/test_smoke.py::test_overfitting_on_single_batch -v
```

---

## Writing New Tests / 编写新测试

### Test Structure / 测试结构

```python
import pytest
import torch

# 1. Fixtures: Reusable test data
# 1. 固定装置：可重用的测试数据
@pytest.fixture
def toy_dataset():
    """Create test dataset"""
    return ToyDataset(num_samples=10)

@pytest.fixture
def model():
    """Create test model"""
    return MyModel(hidden_dim=128)

# 2. Test function: Must start with "test_"
# 2. 测试函数：必须以"test_"开头
def test_model_forward(model, toy_dataset):
    """
    Test model forward pass.
    测试模型前向传播。
    """
    # Arrange: Setup test data
    # 准备：设置测试数据
    batch = toy_dataset[0:4]
    
    # Act: Run the code being tested
    # 执行：运行被测试的代码
    outputs = model(batch)
    
    # Assert: Verify results
    # 断言：验证结果
    assert outputs.shape == (4, 10)
    assert not torch.isnan(outputs).any()
```

### Test Naming Conventions / 测试命名约定

```python
# Good test names / 好的测试名称
def test_model_forward_pass():
    """Clear what is being tested"""

def test_loss_decreases_during_training():
    """Describes expected behavior"""

def test_checkpoint_saves_all_parameters():
    """Specific and descriptive"""

# Bad test names / 不好的测试名称
def test_1():
    """Not descriptive"""

def test_stuff():
    """Too vague"""

def test_model():
    """What about the model?"""
```

### Assertion Best Practices / 断言最佳实践

```python
# Good assertions / 好的断言
assert loss < 1.0, f"Loss too high: {loss}"
assert outputs.shape == expected_shape, \
    f"Shape mismatch: {outputs.shape} != {expected_shape}"

# Check for NaN/Inf
assert torch.isfinite(loss).all(), "Loss contains NaN or Inf"

# Check ranges
assert 0 <= accuracy <= 1, f"Accuracy out of range: {accuracy}"

# Bad assertions / 不好的断言
assert loss  # What are we checking?
assert True  # Always passes
```

---

## Best Practices / 最佳实践

### 1. Keep Tests Fast / 保持测试快速

```python
# Good: Small dataset, few steps
# 好：小数据集，少步骤
def test_training():
    dataset = ToyDataset(num_samples=10)
    train(model, dataset, steps=50)

# Bad: Large dataset, many steps
# 不好：大数据集，多步骤
def test_training():
    dataset = RealDataset(num_samples=100000)
    train(model, dataset, epochs=100)
```

### 2. Test One Thing / 测试一件事

```python
# Good: Tests one specific behavior
# 好：测试一个特定行为
def test_loss_decreases():
    """Test that loss decreases"""
    # ... test loss decrease only

def test_checkpoint_saves():
    """Test that checkpoint saves"""
    # ... test checkpoint only

# Bad: Tests multiple things
# 不好：测试多件事
def test_everything():
    """Test loss, checkpoint, and accuracy"""
    # ... tests too many things
```

### 3. Use Fixtures for Reusable Data / 使用固定装置处理可重用数据

```python
# Good: Reusable fixture
# 好：可重用的固定装置
@pytest.fixture
def model():
    return MyModel(hidden_dim=128)

def test_forward(model):
    outputs = model(inputs)

def test_backward(model):
    loss = model(inputs)
    loss.backward()

# Bad: Duplicate setup
# 不好：重复设置
def test_forward():
    model = MyModel(hidden_dim=128)  # Duplicate
    outputs = model(inputs)

def test_backward():
    model = MyModel(hidden_dim=128)  # Duplicate
    loss = model(inputs)
```

### 4. Clean Up Resources / 清理资源

```python
# Good: Use context managers
# 好：使用上下文管理器
def test_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(model, tmpdir)
        # tmpdir automatically cleaned up
        # tmpdir自动清理

# Good: Use fixtures with cleanup
# 好：使用带清理的固定装置
@pytest.fixture
def temp_dir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)  # Cleanup
```

### 5. Test Edge Cases / 测试边缘情况

```python
def test_model_with_edge_cases():
    """Test model with edge cases"""
    
    # Empty input
    # 空输入
    outputs = model(torch.empty(0, 10))
    assert outputs.shape[0] == 0
    
    # Single sample
    # 单个样本
    outputs = model(torch.randn(1, 10))
    assert outputs.shape == (1, num_classes)
    
    # Maximum length
    # 最大长度
    outputs = model(torch.randn(1, max_length, 10))
    assert outputs.shape[1] == max_length
```

---

## Common Issues and Solutions / 常见问题和解决方案

### Issue 1: Tests are too slow / 测试太慢

**Solution / 解决方案**:
- Use smaller datasets
  使用更小的数据集
- Reduce number of training steps
  减少训练步骤数
- Use CPU instead of GPU for small tests
  对小测试使用CPU而不是GPU
- Run tests in parallel: `pytest -n auto`
  并行运行测试

### Issue 2: Tests are flaky (sometimes pass, sometimes fail) / 测试不稳定

**Solution / 解决方案**:
- Set random seeds
  设置随机种子
- Use deterministic operations
  使用确定性操作
- Increase tolerance for floating point comparisons
  增加浮点比较的容差
- Check for race conditions
  检查竞态条件

### Issue 3: CUDA out of memory / CUDA内存不足

**Solution / 解决方案**:
```python
# Use smaller batch sizes
batch_size = 2  # Instead of 32

# Use smaller models
model = MyModel(hidden_dim=64)  # Instead of 512

# Clear cache
torch.cuda.empty_cache()

# Use CPU for tests
device = 'cpu'
```

### Issue 4: Import errors / 导入错误

**Solution / 解决方案**:
```python
# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Or run from project root
cd project_root
python -m pytest tests/
```

---

## Summary / 总结

**Key Takeaways / 要点**:

1. **Test Early and Often / 及早且频繁地测试**
   - Run smoke tests before long experiments
     在长时间实验之前运行冒烟测试
   - Catch bugs when they're easy to fix
     在容易修复时发现错误

2. **Keep Tests Fast / 保持测试快速**
   - Use small datasets
     使用小数据集
   - Test on toy problems
     在玩具问题上测试
   - Tests should run in seconds
     测试应在几秒钟内运行

3. **Test the Right Things / 测试正确的事情**
   - Loss decrease (learning works)
     损失下降（学习有效）
   - Checkpointing (can save/load)
     检查点（可以保存/加载）
   - Reproducibility (deterministic)
     可复现性（确定性）
   - Overfitting (model capacity)
     过拟合（模型容量）

4. **Write Clear Tests / 编写清晰的测试**
   - Descriptive names
     描述性名称
   - Good documentation
     良好的文档
   - Clear assertions
     清晰的断言
   - One test per behavior
     每个行为一个测试

**Remember / 记住**:
> "Testing shows the presence, not the absence of bugs." - Edsger Dijkstra
> "测试显示错误的存在，而不是不存在。" - Edsger Dijkstra

Good tests give you confidence that your code works, but they can't prove it's perfect. Keep testing, keep improving!

好的测试让你对代码的工作充满信心，但它们无法证明它是完美的。继续测试，继续改进！
