# Quick Reference Card / 快速参考卡

## Smoke Tests Cheat Sheet / 冒烟测试速查表

---

## What is a Smoke Test? / 什么是冒烟测试？

**English**: Quick, basic tests that verify core functionality works before running expensive experiments.

**中文**: 快速、基本的测试，在运行昂贵的实验之前验证核心功能是否正常工作。

**Origin / 起源**: From hardware testing - if it smokes when powered on, there's a problem!
来自硬件测试 - 如果通电时冒烟，就有问题！

---

## Quick Commands / 快速命令

```bash
# Run all smoke tests / 运行所有冒烟测试
pytest tests/test_smoke.py -v

# Run specific test / 运行特定测试
pytest tests/test_smoke.py::test_training_loss_decreases -v

# Run with output / 运行并显示输出
pytest tests/test_smoke.py -v -s

# Stop at first failure / 在第一次失败时停止
pytest tests/test_smoke.py -x
```

---

## Four Core Tests / 四个核心测试

### 1. Loss Decrease / 损失下降
**Tests / 测试**: Model can learn
**Time / 时间**: ~5-10 seconds
**Checks / 检查**: 
- ✓ Forward pass works / 前向传播工作
- ✓ Loss function works / 损失函数工作
- ✓ Gradients flow / 梯度流动
- ✓ Optimizer updates / 优化器更新

### 2. Checkpoint Save/Load / 检查点保存/加载
**Tests / 测试**: Can save and restore model
**Time / 时间**: ~3-5 seconds
**Checks / 检查**:
- ✓ Save to disk / 保存到磁盘
- ✓ Load from disk / 从磁盘加载
- ✓ Weights match / 权重匹配

### 3. Reproducibility / 可复现性
**Tests / 测试**: Same seed = same results
**Time / 时间**: ~10-15 seconds
**Checks / 检查**:
- ✓ Deterministic operations / 确定性操作
- ✓ Seed control works / 种子控制工作
- ✓ Results identical / 结果相同

### 4. Overfitting / 过拟合
**Tests / 测试**: Model can memorize data
**Time / 时间**: ~5-10 seconds
**Checks / 检查**:
- ✓ Model capacity / 模型容量
- ✓ Learning capability / 学习能力
- ✓ Loss decreases > 50% / 损失下降 > 50%

---

## When to Run / 何时运行

✅ **Before long training runs** / 长时间训练之前
✅ **After changing model architecture** / 更改模型架构后
✅ **After updating dependencies** / 更新依赖项后
✅ **Before committing code** / 提交代码之前
✅ **When debugging issues** / 调试问题时

---

## Expected Timing / 预期时间

| Test / 测试 | Time / 时间 |
|-------------|-------------|
| Loss Decrease / 损失下降 | 5-10s |
| Checkpoint / 检查点 | 3-5s |
| Reproducibility / 可复现性 | 10-15s |
| Overfitting / 过拟合 | 5-10s |
| **Total / 总计** | **< 30s** |

---

## Common Issues / 常见问题

### Issue 1: Tests too slow / 测试太慢
**Solution / 解决方案**: Reduce batch size, use smaller model
减少批次大小，使用更小的模型

### Issue 2: CUDA out of memory / CUDA内存不足
**Solution / 解决方案**: Tests auto-fallback to CPU
测试自动回退到CPU

### Issue 3: Loss not decreasing / 损失不下降
**Solution / 解决方案**: Check learning rate, model architecture
检查学习率、模型架构

### Issue 4: Reproducibility fails / 可复现性失败
**Solution / 解决方案**: Enable deterministic mode
启用确定性模式

---

## Key Functions / 关键函数

### setup_logger()
```python
logger = setup_logger("my_logger", "train.log")
logger.info("Training started")
```

### log_metrics()
```python
metrics = {'loss': 0.5, 'accuracy': 0.95}
log_metrics(logger, metrics, "Epoch 1")
```

### set_deterministic()
```python
set_deterministic(True)  # For reproducibility
set_deterministic(False) # For speed
```

---

## Test Data / 测试数据

### Paraformer ASR
```python
ToySeq2SeqDataset(
    num_samples=40,
    vocab_size=50,
    feature_dim=40,
    max_feat_len=80
)
```

### U-Net Segmentation
```python
ToyShapesDataset(
    num_samples=40,
    image_size=64,
    num_classes=3
)
```

---

## Documentation / 文档

📖 **Full Testing Guide**: `TESTING_GUIDE.md`
📖 **Paraformer Tests**: `paraformer_asr/tests/README.md`
📖 **U-Net Tests**: `unet_transformer_seg/tests/README.md`
📖 **Summary**: `SMOKE_TESTS_SUMMARY.md`

---

## Remember / 记住

> **"If smoke tests fail, don't start training!"**
> **"如果冒烟测试失败，不要开始训练！"**

Smoke tests save time by catching issues early.
冒烟测试通过及早发现问题来节省时间。

---

## Quick Checklist / 快速检查清单

Before starting training / 开始训练之前:

- [ ] Run smoke tests / 运行冒烟测试
- [ ] All tests pass / 所有测试通过
- [ ] Loss decreases / 损失下降
- [ ] Checkpoints work / 检查点工作
- [ ] Reproducible / 可复现
- [ ] Model can overfit / 模型可以过拟合

If all ✓, you're ready to train! / 如果全部✓，你就可以开始训练了！
