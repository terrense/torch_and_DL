# U-Net Transformer Segmentation Testing Suite / U-Net Transformer 分割测试套件

## Overview / 概述

This directory contains comprehensive tests for the U-Net Transformer Segmentation system. The tests ensure that the segmentation models work correctly for medical image analysis and other segmentation tasks.

本目录包含U-Net Transformer分割系统的综合测试。测试确保分割模型在医学图像分析和其他分割任务中正常工作。

## Test Categories / 测试类别

### 1. Smoke Tests (test_smoke.py) / 冒烟测试

**Purpose / 目的:**
Smoke tests verify the core training functionality works correctly before running full experiments. They are fast, lightweight tests that catch major issues early.

冒烟测试验证核心训练功能在运行完整实验之前正常工作。它们是快速、轻量级的测试，可以及早发现主要问题。

**What Smoke Tests Verify / 冒烟测试验证内容:**

1. **Loss Decrease / 损失下降**
   - Verifies segmentation loss decreases over 30-50 training steps
   - 验证分割损失在30-50个训练步骤中下降
   - Tests on toy shape dataset (circles, squares)
   - 在玩具形状数据集（圆形、方形）上测试

2. **Checkpoint Save/Load / 检查点保存/加载**
   - Tests model state persistence
   - 测试模型状态持久化
   - Verifies weights are restored correctly
   - 验证权重正确恢复

3. **Reproducibility / 可复现性**
   - Ensures deterministic training with fixed seed
   - 确保使用固定种子的确定性训练
   - Critical for medical imaging applications
   - 对医学成像应用至关重要

4. **Overfitting Capability / 过拟合能力**
   - Tests model capacity on single batch
   - 测试模型在单个批次上的容量
   - Validates model can learn patterns
   - 验证模型可以学习模式

**Key Functions / 关键函数:**

#### `test_training_loss_decreases()`
```python
def test_training_loss_decreases(toy_dataset, model_config):
    """
    Test that segmentation loss decreases over training steps.
    测试分割损失在训练步骤中下降。
    
    Args / 参数:
        toy_dataset: ToyShapesDataset with synthetic shapes
                    包含合成形状的ToyShapesDataset
                    - num_samples: 40
                    - image_size: 64x64
                    - num_classes: 3 (background, circle, square)
                      3个类别（背景、圆形、方形）
        
        model_config: Dictionary with model parameters
                     包含模型参数的字典
                     {
                         'in_channels': 3,      # RGB input / RGB输入
                         'num_classes': 3,      # Output classes / 输出类别
                         'base_channels': 32,   # Base channel count / 基础通道数
                         'depth': 3             # U-Net depth / U-Net深度
                     }
    
    Process / 流程:
        1. Create U-Net model and loss function
           创建U-Net模型和损失函数
        2. Train for 50 steps on toy data
           在玩具数据上训练50步
        3. Compare initial vs final loss
           比较初始损失与最终损失
    
    Expected / 期望:
        avg_final_loss < avg_initial_loss
        平均最终损失 < 平均初始损失
    
    Data Format / 数据格式:
        images: torch.Tensor [batch, 3, 64, 64]  # RGB images / RGB图像
        masks: torch.Tensor [batch, 64, 64]      # Segmentation masks / 分割掩码
    """
```

#### `test_checkpoint_save_load()`
```python
def test_checkpoint_save_load(toy_dataset, model_config):
    """
    Test checkpoint save and load functionality.
    测试检查点保存和加载功能。
    
    Args / 参数:
        toy_dataset: Synthetic shape dataset
                    合成形状数据集
        model_config: Model configuration dictionary
                     模型配置字典
    
    Process / 流程:
        1. Train model for 5 steps
           训练模型5步
        2. Save checkpoint with CheckpointManager
           使用CheckpointManager保存检查点
           - Saves model_state_dict
             保存model_state_dict
           - Saves optimizer_state_dict
             保存optimizer_state_dict
           - Saves training metadata
             保存训练元数据
        3. Load checkpoint into new model
           将检查点加载到新模型
        4. Compare all parameters
           比较所有参数
    
    Checkpoint Format / 检查点格式:
        {
            'epoch': int,                    # Training epoch / 训练轮次
            'model_state_dict': OrderedDict, # Model weights / 模型权重
            'optimizer_state_dict': dict,    # Optimizer state / 优化器状态
            'best_val_metric': float         # Best validation metric / 最佳验证指标
        }
    
    Verifies / 验证:
        torch.allclose(param1, param2, atol=1e-6) for all parameters
        所有参数的torch.allclose(param1, param2, atol=1e-6)
    """
```

#### `test_reproducibility()`
```python
def test_reproducibility(toy_dataset, model_config):
    """
    Test training reproducibility with same random seed.
    测试使用相同随机种子的训练可复现性。
    
    Args / 参数:
        toy_dataset: Synthetic dataset
                    合成数据集
        model_config: Model configuration
                     模型配置
    
    Process / 流程:
        1. Set seed=42, train for 15 steps
           设置seed=42，训练15步
        2. Set seed=42 again, train for 15 steps
           再次设置seed=42，训练15步
        3. Compare losses at each step
           比较每步的损失
    
    Reproducibility Requirements / 可复现性要求:
        - Same random seed
          相同的随机种子
        - Same model initialization
          相同的模型初始化
        - Same data order
          相同的数据顺序
        - Deterministic operations
          确定性操作
    
    Tolerance / 容差:
        abs(loss1 - loss2) < 1e-4
        - Allows for minor floating point differences
          允许微小的浮点差异
        - GPU operations may have small variations
          GPU操作可能有小的变化
    
    Output / 输出:
        List of losses: [float] * 15
        损失列表：[float] * 15
    """
```

#### `test_overfitting_on_single_batch()`
```python
def test_overfitting_on_single_batch(toy_dataset, model_config):
    """
    Test model can overfit on single batch (sanity check).
    测试模型能否在单个批次上过拟合（健全性检查）。
    
    Args / 参数:
        toy_dataset: Toy shapes dataset
                    玩具形状数据集
        model_config: Model configuration
                     模型配置
    
    Purpose / 目的:
        Overfitting on a single batch proves:
        在单个批次上过拟合证明：
        1. Model has sufficient capacity
           模型有足够的容量
        2. Gradients flow correctly
           梯度正确流动
        3. Loss function is appropriate
           损失函数合适
        4. Optimizer is working
           优化器正常工作
    
    Process / 流程:
        1. Get single batch (4 images)
           获取单个批次（4张图像）
        2. Train on same batch 100 times
           在同一批次上训练100次
        3. Monitor loss decrease
           监控损失下降
    
    Expected Behavior / 期望行为:
        - Loss should decrease monotonically
          损失应单调递减
        - Final loss < Initial loss * 0.5
          最终损失 < 初始损失 * 0.5
        - Model should memorize the batch
          模型应记住该批次
    
    Data / 数据:
        images: [4, 3, 64, 64]  # 4 RGB images / 4张RGB图像
        masks: [4, 64, 64]      # 4 segmentation masks / 4个分割掩码
    
    If Test Fails / 如果测试失败:
        - Check learning rate (may be too low)
          检查学习率（可能太低）
        - Check model architecture
          检查模型架构
        - Check loss function
          检查损失函数
        - Check gradient flow
          检查梯度流
    """
```

### 2. Model Tests (test_models.py) / 模型测试

**Purpose / 目的:**
Tests individual model components and architectures.

测试各个模型组件和架构。

**Key Tests / 关键测试:**
- U-Net encoder-decoder structure
  U-Net编码器-解码器结构
- Transformer attention mechanisms
  Transformer注意力机制
- Skip connections
  跳跃连接
- Output shape verification
  输出形状验证

### 3. Data Tests (test_data.py) / 数据测试

**Purpose / 目的:**
Tests data loading and preprocessing pipelines.

测试数据加载和预处理管道。

**Key Tests / 关键测试:**
- Toy shape generation
  玩具形状生成
- Image transformations
  图像变换
- Data augmentation
  数据增强
- Batch collation
  批次整理

## Running Tests / 运行测试

### Run All Tests / 运行所有测试
```bash
cd unet_transformer_seg
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

### Run with Detailed Output / 运行并显示详细输出
```bash
python -m pytest tests/test_smoke.py -v -s
```

## Test Data / 测试数据

### Toy Shapes Dataset / 玩具形状数据集

The smoke tests use `ToyShapesDataset`, which generates synthetic segmentation data:

冒烟测试使用`ToyShapesDataset`，它生成合成分割数据：

**Parameters / 参数:**
```python
ToyShapesDataset(
    num_samples=40,           # Number of images / 图像数量
    image_size=64,            # Image dimensions (64x64) / 图像尺寸
    num_classes=3,            # Classes: background, circle, square
                              # 类别：背景、圆形、方形
    shape_types=['circle', 'square']  # Shape types to generate
                                      # 要生成的形状类型
)
```

**Data Format / 数据格式:**
```python
# Single sample / 单个样本
image: torch.Tensor  # Shape: [3, 64, 64], Range: [0, 1]
                     # 形状：[3, 64, 64]，范围：[0, 1]
                     # RGB image with shapes
                     # 包含形状的RGB图像

mask: torch.Tensor   # Shape: [64, 64], Values: {0, 1, 2}
                     # 形状：[64, 64]，值：{0, 1, 2}
                     # 0: background / 背景
                     # 1: circle / 圆形
                     # 2: square / 方形

# Batch / 批次
images: [batch_size, 3, 64, 64]
masks: [batch_size, 64, 64]
```

## Test Configuration / 测试配置

### Model Configuration / 模型配置
```python
model_config = {
    'in_channels': 3,        # Input channels (RGB) / 输入通道（RGB）
    'num_classes': 3,        # Output classes / 输出类别
    'base_channels': 32,     # Base channel count / 基础通道数
    'depth': 3               # U-Net depth / U-Net深度
}
```

### Loss Functions / 损失函数

**Dice-BCE Loss / Dice-BCE损失:**
```python
loss_fn = get_loss_function('dice_bce', num_classes=3)

# Combines / 结合:
# 1. Dice Loss: Measures overlap between prediction and ground truth
#    Dice损失：测量预测和真实值之间的重叠
# 2. Binary Cross Entropy: Pixel-wise classification loss
#    二元交叉熵：逐像素分类损失
```

## Performance Expectations / 性能期望

### Smoke Test Timing / 冒烟测试时间
- All smoke tests should complete in < 30 seconds
  所有冒烟测试应在30秒内完成
- Individual tests: 5-10 seconds each
  单个测试：每个5-10秒

### Loss Decrease Expectations / 损失下降期望
- Initial loss: ~1.0-2.0 (random initialization)
  初始损失：~1.0-2.0（随机初始化）
- After 50 steps: Should decrease by at least 20%
  50步后：应至少下降20%
- Overfitting test: Should decrease by > 50%
  过拟合测试：应下降>50%

## Troubleshooting / 故障排除

### Common Issues / 常见问题

1. **Shape Mismatch Errors / 形状不匹配错误**
   ```python
   # Check input/output shapes
   # 检查输入/输出形状
   print(f"Input shape: {images.shape}")
   print(f"Output shape: {outputs.shape}")
   print(f"Expected: [batch, num_classes, H, W]")
   ```

2. **Loss Not Decreasing / 损失不下降**
   - Check learning rate (try 1e-3 to 1e-4)
     检查学习率（尝试1e-3到1e-4）
   - Verify loss function is appropriate
     验证损失函数是否合适
   - Check for gradient flow issues
     检查梯度流问题

3. **CUDA Out of Memory / CUDA内存不足**
   - Reduce batch size
     减少批次大小
   - Use smaller image size
     使用更小的图像尺寸
   - Tests automatically fall back to CPU
     测试自动回退到CPU

4. **Reproducibility Issues / 可复现性问题**
   - Ensure deterministic mode is enabled
     确保启用确定性模式
   - Some GPU operations are non-deterministic
     某些GPU操作是非确定性的
   - Use tolerance in comparisons
     在比较中使用容差

## Best Practices / 最佳实践

1. **Always run smoke tests before training**
   在训练前始终运行冒烟测试
   - Catches configuration errors early
     及早发现配置错误
   - Validates data pipeline
     验证数据管道
   - Ensures model architecture is correct
     确保模型架构正确

2. **Use appropriate test data**
   使用适当的测试数据
   - Small datasets for speed
     小数据集以提高速度
   - Diverse shapes for coverage
     多样化形状以提高覆盖率
   - Known ground truth for validation
     已知真实值用于验证

3. **Monitor test metrics**
   监控测试指标
   - Loss curves should be smooth
     损失曲线应平滑
   - No NaN or Inf values
     无NaN或Inf值
   - Gradients should be reasonable magnitude
     梯度应具有合理的量级

## Medical Imaging Considerations / 医学成像考虑因素

When adapting these tests for medical imaging:

将这些测试应用于医学成像时：

1. **Class Imbalance / 类别不平衡**
   - Medical images often have severe class imbalance
     医学图像通常有严重的类别不平衡
   - Use weighted loss functions
     使用加权损失函数
   - Test with realistic class distributions
     使用真实的类别分布进行测试

2. **Image Resolution / 图像分辨率**
   - Medical images are often high resolution (512x512+)
     医学图像通常是高分辨率（512x512+）
   - Test with appropriate patch sizes
     使用适当的补丁大小进行测试
   - Consider memory constraints
     考虑内存限制

3. **Multi-class Segmentation / 多类分割**
   - Test with realistic number of classes
     使用真实的类别数量进行测试
   - Verify class-specific metrics
     验证特定类别的指标
   - Check boundary handling
     检查边界处理

## References / 参考资料

- U-Net Paper: https://arxiv.org/abs/1505.04597
- Dice Loss: https://arxiv.org/abs/1606.04797
- Medical Image Segmentation: https://arxiv.org/abs/1902.09063
- PyTorch Testing: https://pytorch.org/docs/stable/testing.html
