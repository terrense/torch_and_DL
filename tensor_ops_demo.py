#!/usr/bin/env python3
"""
PyTorch张量操作演示脚本
"""

import torch

print("PyTorch张量操作全面学习")
print("使用大张量 [2, 10, 16] = 320个元素")
print()

# 创建基础张量
X = torch.arange(320).reshape(2, 10, 16).float()
print(f"原始张量 X: {X.shape}")
print(f"前8个元素: {X.flatten()[:8].tolist()}")
print(f"是否连续: {X.is_contiguous()}")
print()

print("=" * 60)
print("1. PERMUTE 操作 - 重新排列维度")
print("=" * 60)

# 各种permute操作
permute_cases = [
    (2, 0, 1),  # [16, 2, 10]
    (1, 2, 0),  # [10, 16, 2] 
    (0, 2, 1),  # [2, 16, 10]
    (2, 1, 0),  # [16, 10, 2]
]

for perm in permute_cases:
    result = X.permute(*perm)
    print(f"permute{perm}: {X.shape} -> {result.shape}")
    print(f"  是否连续: {result.is_contiguous()}")
    print(f"  步长: {result.stride()}")
    print()

print("=" * 60)
print("2. TRANSPOSE 操作 - 交换两个维度")
print("=" * 60)

transpose_cases = [
    (0, 1),  # 交换维度0和1
    (0, 2),  # 交换维度0和2
    (1, 2),  # 交换维度1和2
]

for dim0, dim1 in transpose_cases:
    result = X.transpose(dim0, dim1)
    print(f"transpose({dim0}, {dim1}): {X.shape} -> {result.shape}")
    print(f"  是否连续: {result.is_contiguous()}")
    print()

print("=" * 60)
print("3. VIEW vs RESHAPE 操作")
print("=" * 60)

# view和reshape对比
shapes = [(320,), (4, 80), (8, 5, 8)]

print("连续张量:")
for shape in shapes:
    view_result = X.view(*shape)
    reshape_result = X.reshape(*shape)
    print(f"  {X.shape} -> {shape}")
    print(f"    view: 成功, reshape: 成功")

print("\n非连续张量:")
X_non_contiguous = X.transpose(0, 2)
print(f"转置后: {X_non_contiguous.shape}, 连续性: {X_non_contiguous.is_contiguous()}")

for shape in shapes:
    try:
        view_result = X_non_contiguous.view(*shape)
        view_success = "成功"
    except:
        view_success = "失败"
    
    reshape_result = X_non_contiguous.reshape(*shape)
    print(f"  {X_non_contiguous.shape} -> {shape}")
    print(f"    view: {view_success}, reshape: 成功")

print()

print("=" * 60)
print("4. SQUEEZE & UNSQUEEZE 操作")
print("=" * 60)

# 创建有单维度的张量
X_with_ones = torch.arange(320).reshape(1, 2, 10, 16, 1).float()
print(f"带单维度的张量: {X_with_ones.shape}")

# squeeze操作
squeeze_all = X_with_ones.squeeze()
print(f"squeeze(): {X_with_ones.shape} -> {squeeze_all.shape}")

squeeze_dim0 = X_with_ones.squeeze(0)
print(f"squeeze(0): {X_with_ones.shape} -> {squeeze_dim0.shape}")

# unsqueeze操作
base = torch.arange(320).reshape(2, 10, 16).float()
for dim in range(4):
    result = base.unsqueeze(dim)
    print(f"unsqueeze({dim}): {base.shape} -> {result.shape}")

print()

print("=" * 60)
print("5. FLATTEN 操作")
print("=" * 60)

flatten_all = X.flatten()
print(f"flatten(): {X.shape} -> {flatten_all.shape}")

flatten_12 = X.flatten(1, 2)
print(f"flatten(1, 2): {X.shape} -> {flatten_12.shape}")

flatten_01 = X.flatten(0, 1)
print(f"flatten(0, 1): {X.shape} -> {flatten_01.shape}")

print()

print("=" * 60)
print("6. EXPAND & REPEAT 操作")
print("=" * 60)

# 创建小张量用于扩展
small_X = torch.arange(6).reshape(1, 2, 3).float()
print(f"小张量: {small_X.shape}")

# expand操作
expand_result = small_X.expand(4, 2, 3)
print(f"expand(4, 2, 3): {small_X.shape} -> {expand_result.shape}")
print(f"  内存: 原始={small_X.numel()}, 扩展后视图={expand_result.numel()}")

# repeat操作
repeat_result = small_X.repeat(4, 1, 1)
print(f"repeat(4, 1, 1): {small_X.shape} -> {repeat_result.shape}")
print(f"  内存: 原始={small_X.numel()}, 重复后={repeat_result.numel()}")

print()

print("=" * 60)
print("总结:")
print("- permute: 重新排列所有维度，可能破坏连续性")
print("- transpose: 只交换两个维度，通常破坏连续性")
print("- view: 改变形状，要求连续内存")
print("- reshape: 改变形状，更灵活，必要时会复制")
print("- squeeze/unsqueeze: 删除/添加大小为1的维度")
print("- flatten: 展平指定维度范围")
print("- expand: 扩展维度，不复制数据，共享内存")
print("- repeat: 重复数据，实际复制内存")
print("=" * 60)