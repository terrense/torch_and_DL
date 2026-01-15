#!/usr/bin/env python3
"""
PyTorch张量操作全面学习脚本
演示permute, transpose, view, reshape, squeeze, unsqueeze等操作的区别
使用[2, 10, 16]的大张量来建立直觉
"""

import torch
import numpy as np

def print_tensor_info(tensor, name):
    """打印张量的基本信息"""
    print(f"{name}:")
    print(f"  形状: {tensor.shape}")
    print(f"  步长: {tensor.stride()}")
    print(f"  是否连续: {tensor.is_contiguous()}")
    print(f"  数据类型: {tensor.dtype}")
    print(f"  前几个元素: {tensor.flatten()[:8].tolist()}")
    print()

def demonstrate_permute():
    """演示permute操作"""
    print("=" * 60)
    print("1. PERMUTE 操作 - 重新排列维度")
    print("=" * 60)
    
    # 创建[2, 10, 16]张量
    X = torch.arange(320).reshape(2, 10, 16).float()
    print_tensor_info(X, "原始张量 X [2, 10, 16]")
    
    # 各种permute操作
    cases = [
        (2, 0, 1),  # [16, 2, 10]
        (1, 2, 0),  # [10, 16, 2] 
        (0, 2, 1),  # [2, 16, 10]
        (2, 1, 0),  # [16, 10, 2]
    ]
    
    for perm in cases:
        result = X.permute(*perm)
        print(f"permute{perm}: {X.shape} -> {result.shape}")
        print(f"  映射关系: X[a,b,c] -> result[{chr(97+perm.index(0))},{chr(97+perm.index(1))},{chr(97+perm.index(2))}]")
        print(f"  是否连续: {result.is_contiguous()}")
        print()

def demonstrate_transpose():
    """演示transpose操作"""
    print("=" * 60)
    print("2. TRANSPOSE 操作 - 交换两个维度")
    print("=" * 60)
    
    X = torch.arange(320).reshape(2, 10, 16).float()
    print_tensor_info(X, "原始张量 X [2, 10, 16]")
    
    # 各种transpose操作
    cases = [
        (0, 1),  # 交换维度0和1
        (0, 2),  # 交换维度0和2
        (1, 2),  # 交换维度1和2
    ]
    
    for dim0, dim1 in cases:
        result = X.transpose(dim0, dim1)
        print(f"transpose({dim0}, {dim1}): {X.shape} -> {result.shape}")
        print(f"  等价于 permute: {X.permute(*[i if i not in [dim0, dim1] else (dim1 if i == dim0 else dim0) for i in range(3)]).shape}")
        print(f"  是否连续: {result.is_contiguous()}")
        print()

def demonstrate_view_reshape():
    """演示view和reshape的区别"""
    print("=" * 60)
    print("3. VIEW vs RESHAPE 操作 - 改变形状")
    print("=" * 60)
    
    X = torch.arange(320).reshape(2, 10, 16).float()
    print_tensor_info(X, "原始张量 X [2, 10, 16]")
    
    # view操作 - 要求连续内存
    print("VIEW操作 (要求连续内存):")
    view_cases = [
        (320,),           # 展平
        (4, 80),          # 2D
        (8, 5, 8),        # 3D重组
        (2, 5, 2, 16),    # 4D
    ]
    
    for shape in view_cases:
        try:
            result = X.view(*shape)
            print(f"  view{shape}: 成功 -> {result.shape}")
        except Exception as e:
            print(f"  view{shape}: 失败 - {e}")
    
    print()
    
    # reshape操作 - 更灵活
    print("RESHAPE操作 (更灵活):")
    for shape in view_cases:
        result = X.reshape(*shape)
        print(f"  reshape{shape}: 成功 -> {result.shape}")
    
    print()
    
    # 非连续张量的情况
    print("非连续张量的情况:")
    X_transposed = X.transpose(0, 2)  # 创建非连续张量
    print_tensor_info(X_transposed, "转置后的张量 (非连续)")
    
    try:
        view_result = X_transposed.view(320)
        print("  view(320): 成功")
    except Exception as e:
        print(f"  view(320): 失败 - {e}")
    
    reshape_result = X_transposed.reshape(320)
    print("  reshape(320): 成功")
    print()

def demonstrate_squeeze_unsqueeze():
    """演示squeeze和unsqueeze操作"""
    print("=" * 60)
    print("4. SQUEEZE & UNSQUEEZE 操作 - 添加/删除维度")
    print("=" * 60)
    
    # 创建有单维度的张量
    X = torch.arange(320).reshape(1, 2, 10, 16, 1).float()
    print_tensor_info(X, "原始张量 X [1, 2, 10, 16, 1]")
    
    # squeeze操作
    print("SQUEEZE操作 (删除大小为1的维度):")
    squeeze_all = X.squeeze()
    print(f"  squeeze(): {X.shape} -> {squeeze_all.shape}")
    
    squeeze_dim0 = X.squeeze(0)
    print(f"  squeeze(0): {X.shape} -> {squeeze_dim0.shape}")
    
    squeeze_dim4 = X.squeeze(4)
    print(f"  squeeze(4): {X.shape} -> {squeeze_dim4.shape}")
    
    try:
        squeeze_dim1 = X.squeeze(1)  # 维度1大小不是1
        print(f"  squeeze(1): {X.shape} -> {squeeze_dim1.shape}")
    except:
        print(f"  squeeze(1): 维度1大小为{X.shape[1]}，不能squeeze")
    
    print()
    
    # unsqueeze操作
    print("UNSQUEEZE操作 (添加大小为1的维度):")
    base = torch.arange(320).reshape(2, 10, 16).float()
    print_tensor_info(base, "基础张量 [2, 10, 16]")
    
    for dim in range(4):
        result = base.unsqueeze(dim)
        print(f"  unsqueeze({dim}): {base.shape} -> {result.shape}")
    
    print()

def demonstrate_flatten_unflatten():
    """演示flatten和unflatten操作"""
    print("=" * 60)
    print("5. FLATTEN & UNFLATTEN 操作 - 展平和反展平")
    print("=" * 60)
    
    X = torch.arange(320).reshape(2, 10, 16).float()
    print_tensor_info(X, "原始张量 X [2, 10, 16]")
    
    # flatten操作
    print("FLATTEN操作:")
    flatten_all = X.flatten()
    print(f"  flatten(): {X.shape} -> {flatten_all.shape}")
    
    flatten_12 = X.flatten(1, 2)
    print(f"  flatten(1, 2): {X.shape} -> {flatten_12.shape}")
    
    flatten_01 = X.flatten(0, 1)
    print(f"  flatten(0, 1): {X.shape} -> {flatten_01.shape}")
    
    print()
    
    # unflatten操作
    print("UNFLATTEN操作:")
    flat = X.flatten()
    unflat1 = flat.unflatten(0, (2, 10, 16))
    print(f"  unflatten(0, (2,10,16)): {flat.shape} -> {unflat1.shape}")
    
    partial_flat = X.flatten(1, 2)
    unflat2 = partial_flat.unflatten(1, (10, 16))
    print(f"  unflatten(1, (10,16)): {partial_flat.shape} -> {unflat2.shape}")
    
    print()

def demonstrate_expand_repeat():
    """演示expand和repeat操作"""
    print("=" * 60)
    print("6. EXPAND & REPEAT 操作 - 扩展张量")
    print("=" * 60)
    
    # 创建小张量用于扩展
    X = torch.arange(6).reshape(1, 2, 3).float()
    print_tensor_info(X, "原始张量 X [1, 2, 3]")
    
    # expand操作 - 不复制数据
    print("EXPAND操作 (不复制数据，共享内存):")
    expand1 = X.expand(4, 2, 3)
    print(f"  expand(4, 2, 3): {X.shape} -> {expand1.shape}")
    print(f"  内存使用: 原始={X.numel()}, 扩展后={expand1.numel()}, 实际存储={X.numel()}")
    
    # expand只能扩展大小为1的维度，不能改变非单维度的大小
    expand2 = X.expand(4, -1, -1)  # -1表示保持原维度
    print(f"  expand(4, -1, -1): {X.shape} -> {expand2.shape}")
    
    # 演示expand的限制
    print("  expand的限制:")
    try:
        invalid_expand = X.expand(4, 2, 6)  # 试图将维度2从3扩展到6
        print(f"    expand(4, 2, 6): 成功")
    except RuntimeError as e:
        print(f"    expand(4, 2, 6): 失败 - 不能扩展非单维度")
        print(f"    错误: {str(e)[:60]}...")
    
    # 演示正确的expand用法
    print("  正确的expand用法 - 只扩展单维度:")
    Y = torch.arange(6).reshape(1, 1, 6).float()  # 创建有多个单维度的张量
    print(f"    Y形状: {Y.shape}")
    expand_Y = Y.expand(3, 4, 6)
    print(f"    Y.expand(3, 4, 6): {Y.shape} -> {expand_Y.shape}")
    
    print()
    
    # repeat操作 - 复制数据
    print("REPEAT操作 (复制数据):")
    repeat1 = X.repeat(4, 1, 1)
    print(f"  repeat(4, 1, 1): {X.shape} -> {repeat1.shape}")
    print(f"  内存使用: 原始={X.numel()}, 重复后={repeat1.numel()}")
    
    repeat2 = X.repeat(2, 3, 2)
    print(f"  repeat(2, 3, 2): {X.shape} -> {repeat2.shape}")
    print(f"  内存使用: 原始={X.numel()}, 重复后={repeat2.numel()}")
    
    print()
    
    # 关键区别演示
    print("EXPAND vs REPEAT 关键区别:")
    print("1. expand只能扩展大小为1的维度")
    print("2. repeat可以在任何维度上重复任意次数")
    print("3. expand共享内存，repeat复制内存")
    print("4. expand更节省内存，但有使用限制")
    print()

def demonstrate_memory_layout():
    """演示内存布局的影响"""
    print("=" * 60)
    print("7. 内存布局和性能影响")
    print("=" * 60)
    
    X = torch.arange(320).reshape(2, 10, 16).float()
    
    operations = [
        ("原始", X),
        ("permute(2,0,1)", X.permute(2, 0, 1)),
        ("transpose(0,2)", X.transpose(0, 2)),
        ("contiguous()", X.permute(2, 0, 1).contiguous()),
    ]
    
    for name, tensor in operations:
        print(f"{name}:")
        print(f"  形状: {tensor.shape}")
        print(f"  步长: {tensor.stride()}")
        print(f"  连续性: {tensor.is_contiguous()}")
        print(f"  存储大小: {tensor.storage().size()}")
        print()

def main():
    print("PyTorch张量操作全面学习")
    print("使用大张量 [2, 10, 16] = 320个元素")
    print()
    
    demonstrate_permute()
    demonstrate_transpose()
    demonstrate_view_reshape()
    demonstrate_squeeze_unsqueeze()
    demonstrate_flatten_unflatten()
    demonstrate_expand_repeat()
    demonstrate_memory_layout()
    
    print("=" * 60)
    print("总结:")
    print("- permute: 重新排列所有维度")
    print("- transpose: 只交换两个维度")
    print("- view: 改变形状，要求连续内存")
    print("- reshape: 改变形状，更灵活")
    print("- squeeze/unsqueeze: 删除/添加大小为1的维度")
    print("- flatten/unflatten: 展平/反展平指定维度")
    print("- expand: 扩展维度，不复制数据")
    print("- repeat: 重复数据，复制内存")
    print("=" * 60)

if __name__ == "__main__":
    main()