#!/usr/bin/env python3
"""
PyTorch tensor operations demonstration script
"""

import torch

print("PyTorch Tensor Operations Comprehensive Learning")
print("Using large tensor [2, 10, 16] = 320 elements")
print()

# Create base tensor
X = torch.arange(320).reshape(2, 10, 16).float()
print(f"Original tensor X: {X.shape}")
print(f"First 8 elements: {X.flatten()[:8].tolist()}")
print(f"Is contiguous: {X.is_contiguous()}")
print()

print("=" * 60)
print("1. PERMUTE Operations - Rearrange dimensions")
print("=" * 60)

# Various permute operations
permute_cases = [
    (2, 0, 1),  # [16, 2, 10]
    (1, 2, 0),  # [10, 16, 2] 
    (0, 2, 1),  # [2, 16, 10]
    (2, 1, 0),  # [16, 10, 2]
]

for perm in permute_cases:
    result = X.permute(*perm)
    print(f"permute{perm}: {X.shape} -> {result.shape}")
    print(f"  Is contiguous: {result.is_contiguous()}")
    print(f"  Stride: {result.stride()}")
    print()

print("=" * 60)
print("2. TRANSPOSE Operations - Swap two dimensions")
print("=" * 60)

transpose_cases = [
    (0, 1),  # Swap dimensions 0 and 1
    (0, 2),  # Swap dimensions 0 and 2
    (1, 2),  # Swap dimensions 1 and 2
]

for dim0, dim1 in transpose_cases:
    result = X.transpose(dim0, dim1)
    print(f"transpose({dim0}, {dim1}): {X.shape} -> {result.shape}")
    print(f"  Is contiguous: {result.is_contiguous()}")
    print()

print("=" * 60)
print("3. VIEW vs RESHAPE Operations")
print("=" * 60)

# view and reshape comparison
shapes = [(320,), (4, 80), (8, 5, 8)]

print("Contiguous tensor:")
for shape in shapes:
    view_result = X.view(*shape)
    reshape_result = X.reshape(*shape)
    print(f"  {X.shape} -> {shape}")
    print(f"    view: success, reshape: success")

print("\nNon-contiguous tensor:")
X_non_contiguous = X.transpose(0, 2)
print(f"After transpose: {X_non_contiguous.shape}, contiguous: {X_non_contiguous.is_contiguous()}")

for shape in shapes:
    try:
        view_result = X_non_contiguous.view(*shape)
        view_success = "success"
    except:
        view_success = "failed"
    
    reshape_result = X_non_contiguous.reshape(*shape)
    print(f"  {X_non_contiguous.shape} -> {shape}")
    print(f"    view: {view_success}, reshape: success")

print()

print("=" * 60)
print("4. SQUEEZE & UNSQUEEZE Operations")
print("=" * 60)

# Create tensor with singleton dimensions
X_with_ones = torch.arange(320).reshape(1, 2, 10, 16, 1).float()
print(f"Tensor with singleton dims: {X_with_ones.shape}")

# squeeze operations
squeeze_all = X_with_ones.squeeze()
print(f"squeeze(): {X_with_ones.shape} -> {squeeze_all.shape}")

squeeze_dim0 = X_with_ones.squeeze(0)
print(f"squeeze(0): {X_with_ones.shape} -> {squeeze_dim0.shape}")

# unsqueeze operations
base = torch.arange(320).reshape(2, 10, 16).float()
for dim in range(4):
    result = base.unsqueeze(dim)
    print(f"unsqueeze({dim}): {base.shape} -> {result.shape}")

print()

print("=" * 60)
print("5. FLATTEN Operations")
print("=" * 60)

flatten_all = X.flatten()
print(f"flatten(): {X.shape} -> {flatten_all.shape}")

flatten_12 = X.flatten(1, 2)
print(f"flatten(1, 2): {X.shape} -> {flatten_12.shape}")

flatten_01 = X.flatten(0, 1)
print(f"flatten(0, 1): {X.shape} -> {flatten_01.shape}")

print()

print("=" * 60)
print("6. EXPAND & REPEAT Operations")
print("=" * 60)

# Create small tensor for expansion
small_X = torch.arange(6).reshape(1, 2, 3).float()
print(f"Small tensor: {small_X.shape}")

# expand operation
expand_result = small_X.expand(4, 2, 3)
print(f"expand(4, 2, 3): {small_X.shape} -> {expand_result.shape}")
print(f"  Memory: original={small_X.numel()}, expanded view={expand_result.numel()}")

# repeat operation
repeat_result = small_X.repeat(4, 1, 1)
print(f"repeat(4, 1, 1): {small_X.shape} -> {repeat_result.shape}")
print(f"  Memory: original={small_X.numel()}, repeated={repeat_result.numel()}")

print()

print("=" * 60)
print("Summary:")
print("- permute: rearrange all dimensions, may break contiguity")
print("- transpose: swap only two dimensions, usually breaks contiguity")
print("- view: change shape, requires contiguous memory")
print("- reshape: change shape, more flexible, copies when necessary")
print("- squeeze/unsqueeze: remove/add dimensions of size 1")
print("- flatten: flatten specified dimension ranges")
print("- expand: extend dimensions, no data copy, shares memory")
print("- repeat: duplicate data, actual memory copy")
print("=" * 60)