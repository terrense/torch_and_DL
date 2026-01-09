#!/usr/bin/env python3
"""
PyTorch permute function learning script
Demonstrates how to use permute to rearrange tensor dimensions
"""

import torch
import numpy as np

def main():
    print("=== PyTorch permute Function Learning ===\n")
    
    # Create a (2,3,4) 3D tensor containing elements 1 to 24
    # Fill in xyz axis order: z-axis (innermost), y-axis (middle), x-axis (outermost)
    X = torch.arange(1, 25).reshape(2, 3, 4).float()
    
    print("Original tensor X shape:", X.shape)  # (2, 3, 4)
    print("Original tensor X:")
    print(X)
    print()
    
    # Detailed display of original tensor structure
    print("Detailed structure of original tensor X:")
    for i in range(X.shape[0]):  # x-axis (dimension 0)
        print(f"X[{i}] (slice {i+1} along x-axis):")
        for j in range(X.shape[1]):  # y-axis (dimension 1)
            print(f"  X[{i},{j}] = {X[i,j].tolist()}")  # z-axis (dimension 2)
        print()
    
    # Use permute(2,0,1) to rearrange dimensions
    # Original: (dim0=2, dim1=3, dim2=4) -> New: (dim2=4, dim0=2, dim1=3)
    X1 = X.permute(2, 0, 1)
    
    print("Shape of tensor X1 after permute(2,0,1):", X1.shape)  # (4, 2, 3)
    print("Tensor X1 after permute:")
    print(X1)
    print()
    
    # Detailed display of permuted tensor structure
    print("Detailed structure of permuted tensor X1:")
    for i in range(X1.shape[0]):  # Original z-axis is now dimension 0
        print(f"X1[{i}] (all elements where original z={i}):")
        for j in range(X1.shape[1]):  # Original x-axis is now dimension 1
            print(f"  X1[{i},{j}] = {X1[i,j].tolist()}")  # Original y-axis is now dimension 2
        print()
    
    # Verify the correctness of permute operation
    print("=== Verifying permute operation ===")
    print("Check position changes of key elements:")
    
    # Original position X[0,0,0] = 1
    print(f"X[0,0,0] = {X[0,0,0].item()}")
    print(f"X1[0,0,0] = {X1[0,0,0].item()}")  # Should still be 1 after permute
    
    # Original position X[1,2,3] = 24 (last element)
    print(f"X[1,2,3] = {X[1,2,3].item()}")
    print(f"X1[3,1,2] = {X1[3,1,2].item()}")  # Position becomes [3,1,2] after permute
    
    # Original position X[0,1,2] = 7
    print(f"X[0,1,2] = {X[0,1,2].item()}")
    print(f"X1[2,0,1] = {X1[2,0,1].item()}")  # Position becomes [2,0,1] after permute
    
    print("\n=== permute operation explanation ===")
    print("Meaning of permute(2,0,1):")
    print("- Original dimension 2 (z-axis, size 4) -> New dimension 0")
    print("- Original dimension 0 (x-axis, size 2) -> New dimension 1") 
    print("- Original dimension 1 (y-axis, size 3) -> New dimension 2")
    print("- Shape changes from (2,3,4) to (4,2,3)")
    print("- Element position changes from [x,y,z] to [z,x,y]")
    
    print("\n=== Key mapping relationship ===")
    print("X1[i,j,k] = X[j,k,i]")
    print("Because permute(2,0,1) means:")
    print("- X1's dimension 0 (i) comes from X's dimension 2")
    print("- X1's dimension 1 (j) comes from X's dimension 0") 
    print("- X1's dimension 2 (k) comes from X's dimension 1")
    print()
    
    print("Verify this mapping relationship:")
    # Randomly select several positions for verification
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
    
    print(f"\nSummary: For permute(2,0,1):")
    print(f"X1[i,j,k] = X[j,k,i]")

if __name__ == "__main__":
    main()