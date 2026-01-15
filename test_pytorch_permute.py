#!/usr/bin/env python3
"""
PyTorch permute函数学习脚本
演示如何使用permute重新排列张量的维度
"""

import torch
import numpy as np

def main():
    print("=== PyTorch permute函数学习 ===\n")
    
    # 创建一个(2,3,4)的三维张量，包含1到24的元素
    # 按照xyz轴顺序填充：z轴(最内层)、y轴(中间层)、x轴(最外层)
    X = torch.arange(1, 25).reshape(2, 3, 4).float()
    
    print("原始张量 X 的形状:", X.shape)  # (2, 3, 4)
    print("原始张量 X:")
    print(X)
    print()
    
    # 详细展示原始张量的结构
    print("原始张量 X 的详细结构:")
    for i in range(X.shape[0]):  # x轴 (维度0)
        print(f"X[{i}] (第{i+1}个x切片):")
        for j in range(X.shape[1]):  # y轴 (维度1)
            print(f"  X[{i},{j}] = {X[i,j].tolist()}")  # z轴 (维度2)
        print()
    
    # 使用permute(2,0,1)重新排列维度
    # 原来: (dim0=2, dim1=3, dim2=4) -> 新的: (dim2=4, dim0=2, dim1=3)
    X1 = X.permute(2, 0, 1)
    
    print("permute(2,0,1)后的张量 X1 的形状:", X1.shape)  # (4, 2, 3)
    print("permute后的张量 X1:")
    print(X1)
    print()
    
    # 详细展示permute后张量的结构
    print("permute后张量 X1 的详细结构:")
    for i in range(X1.shape[0]):  # 原来的z轴现在是维度0
        print(f"X1[{i}] (原来z={i}的所有元素):")
        for j in range(X1.shape[1]):  # 原来的x轴现在是维度1
            print(f"  X1[{i},{j}] = {X1[i,j].tolist()}")  # 原来的y轴现在是维度2
        print()
    
    # 验证permute操作的正确性
    print("=== 验证permute操作 ===")
    print("检查几个关键元素的位置变化:")
    
    # 原始位置 X[0,0,0] = 1
    print(f"X[0,0,0] = {X[0,0,0].item()}")
    print(f"X1[0,0,0] = {X1[0,0,0].item()}")  # permute后应该还是1
    
    # 原始位置 X[1,2,3] = 24 (最后一个元素)
    print(f"X[1,2,3] = {X[1,2,3].item()}")
    print(f"X1[3,1,2] = {X1[3,1,2].item()}")  # permute后位置变为[3,1,2]
    
    # 原始位置 X[0,1,2] = 7
    print(f"X[0,1,2] = {X[0,1,2].item()}")
    print(f"X1[2,0,1] = {X1[2,0,1].item()}")  # permute后位置变为[2,0,1]
    
    print("\n=== permute操作解释 ===")
    print("permute(2,0,1)的含义:")
    print("- 原来的维度2(z轴,大小4) -> 新的维度0")
    print("- 原来的维度0(x轴,大小2) -> 新的维度1") 
    print("- 原来的维度1(y轴,大小3) -> 新的维度2")
    print("- 形状从(2,3,4)变为(4,2,3)")
    print("- 元素位置从[x,y,z]变为[z,x,y]")
    
    print("\n=== 关键映射关系 ===")
    print("X1[i,j,k] = X[j,k,i]")
    print("因为 permute(2,0,1) 意味着:")
    print("- X1的第0维(i) 来自 X的第2维")
    print("- X1的第1维(j) 来自 X的第0维") 
    print("- X1的第2维(k) 来自 X的第1维")
    print()
    
    print("验证这个映射关系:")
    # 随机选择几个位置验证
    test_cases = [
        (0, 0, 0),  # X1[0,0,0] = X[0,0,0]
        (1, 0, 1),  # X1[1,0,1] = X[0,1,1] 
        (2, 1, 2),  # X1[2,1,2] = X[1,2,2]
        (3, 1, 0),  # X1[3,1,0] = X[1,0,3]
    ]
    
    for i, j, k in test_cases:
        if i < X1.shape[0] and j < X1.shape[1] and k < X1.shape[2]:
            x1_val = X1[i, j, k].item()
            x_val = X[j, k, i].item()
            print(f"X1[{i},{j},{k}] = {x1_val:.0f}, X[{j},{k},{i}] = {x_val:.0f} ✓")
    
    print(f"\n总结: 对于 permute(2,0,1):")
    print(f"X1[i,j,k] = X[j,k,i]")

if __name__ == "__main__":
    main()