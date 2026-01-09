#!/usr/bin/env python3
"""
PyTorch tensor operations comprehensive learning script
Demonstrates differences between permute, transpose, view, reshape, squeeze, unsqueeze, etc.
Uses large tensor [2, 10, 16] to build intuition
"""

import torch
import numpy as np

def print_tensor_info(tensor, name):
    """Print basic tensor information"""
    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Stride: {tensor.stride()}")
    print(f"  Is contiguous: {tensor.is_contiguous()}")
    print(f"  Data type: {tensor.dtype}")
    print(f"  First few elements: {tensor.flatten()[:8].tolist()}")
    print()

def demonstrate_permute():
    """Demonstrate permute operations"""
    print("=" * 60)
    print("1. PERMUTE Operations - Rearrange dimensions")
    print("=" * 60)
    
    # Create [2, 10, 16] tensor
    X = torch.arange(320).reshape(2, 10, 16).float()
    print_tensor_info(X, "Original tensor X [2, 10, 16]")
    
    # Various permute operations
    cases = [
        (2, 0, 1),  # [16, 2, 10]
        (1, 2, 0),  # [10, 16, 2] 
        (0, 2, 1),  # [2, 16, 10]
        (2, 1, 0),  # [16, 10, 2]
    ]
    
    for perm in cases:
        result = X.permute(*perm)
        print(f"permute{perm}: {X.shape} -> {result.shape}")
        print(f"  Mapping: X[a,b,c] -> result[{chr(97+perm.index(0))},{chr(97+perm.index(1))},{chr(97+perm.index(2))}]")
        print(f"  Is contiguous: {result.is_contiguous()}")
        print()

def demonstrate_transpose():
    """Demonstrate transpose operations"""
    print("=" * 60)
    print("2. TRANSPOSE Operations - Swap two dimensions")
    print("=" * 60)
    
    X = torch.arange(320).reshape(2, 10, 16).float()
    print_tensor_info(X, "Original tensor X [2, 10, 16]")
    
    # Various transpose operations
    cases = [
        (0, 1),  # Swap dimensions 0 and 1
        (0, 2),  # Swap dimensions 0 and 2
        (1, 2),  # Swap dimensions 1 and 2
    ]
    
    for dim0, dim1 in cases:
        result = X.transpose(dim0, dim1)
        print(f"transpose({dim0}, {dim1}): {X.shape} -> {result.shape}")
        print(f"  Equivalent permute: {X.permute(*[i if i not in [dim0, dim1] else (dim1 if i == dim0 else dim0) for i in range(3)]).shape}")
        print(f"  Is contiguous: {result.is_contiguous()}")
        print()

def demonstrate_view_reshape():
    """Demonstrate differences between view and reshape"""
    print("=" * 60)
    print("3. VIEW vs RESHAPE Operations - Change shape")
    print("=" * 60)
    
    X = torch.arange(320).reshape(2, 10, 16).float()
    print_tensor_info(X, "Original tensor X [2, 10, 16]")
    
    # view operations - require contiguous memory
    print("VIEW Operations (require contiguous memory):")
    view_cases = [
        (320,),           # Flatten
        (4, 80),          # 2D
        (8, 5, 8),        # 3D reorganization
        (2, 5, 2, 16),    # 4D
    ]
    
    for shape in view_cases:
        try:
            result = X.view(*shape)
            print(f"  view{shape}: success -> {result.shape}")
        except Exception as e:
            print(f"  view{shape}: failed - {e}")
    
    print()
    
    # reshape operations - more flexible
    print("RESHAPE Operations (more flexible):")
    for shape in view_cases:
        result = X.reshape(*shape)
        print(f"  reshape{shape}: success -> {result.shape}")
    
    print()
    
    # Non-contiguous tensor case
    print("Non-contiguous tensor case:")
    X_transposed = X.transpose(0, 2)  # Create non-contiguous tensor
    print_tensor_info(X_transposed, "Transposed tensor (non-contiguous)")
    
    try:
        view_result = X_transposed.view(320)
        print("  view(320): success")
    except Exception as e:
        print(f"  view(320): failed - {e}")
    
    reshape_result = X_transposed.reshape(320)
    print("  reshape(320): success")
    print()

def demonstrate_squeeze_unsqueeze():
    """Demonstrate squeeze and unsqueeze operations"""
    print("=" * 60)
    print("4. SQUEEZE & UNSQUEEZE Operations - Add/remove dimensions")
    print("=" * 60)
    
    # Create tensor with singleton dimensions
    X = torch.arange(320).reshape(1, 2, 10, 16, 1).float()
    print_tensor_info(X, "Original tensor X [1, 2, 10, 16, 1]")
    
    # squeeze operations
    print("SQUEEZE Operations (remove dimensions of size 1):")
    squeeze_all = X.squeeze()
    print(f"  squeeze(): {X.shape} -> {squeeze_all.shape}")
    
    squeeze_dim0 = X.squeeze(0)
    print(f"  squeeze(0): {X.shape} -> {squeeze_dim0.shape}")
    
    squeeze_dim4 = X.squeeze(4)
    print(f"  squeeze(4): {X.shape} -> {squeeze_dim4.shape}")
    
    try:
        squeeze_dim1 = X.squeeze(1)  # Dimension 1 size is not 1
        print(f"  squeeze(1): {X.shape} -> {squeeze_dim1.shape}")
    except:
        print(f"  squeeze(1): dimension 1 size is {X.shape[1]}, cannot squeeze")
    
    print()
    
    # unsqueeze operations
    print("UNSQUEEZE Operations (add dimensions of size 1):")
    base = torch.arange(320).reshape(2, 10, 16).float()
    print_tensor_info(base, "Base tensor [2, 10, 16]")
    
    for dim in range(4):
        result = base.unsqueeze(dim)
        print(f"  unsqueeze({dim}): {base.shape} -> {result.shape}")
    
    print()

def demonstrate_flatten_unflatten():
    """Demonstrate flatten and unflatten operations"""
    print("=" * 60)
    print("5. FLATTEN & UNFLATTEN Operations - Flatten and unflatten")
    print("=" * 60)
    
    X = torch.arange(320).reshape(2, 10, 16).float()
    print_tensor_info(X, "Original tensor X [2, 10, 16]")
    
    # flatten operations
    print("FLATTEN Operations:")
    flatten_all = X.flatten()
    print(f"  flatten(): {X.shape} -> {flatten_all.shape}")
    
    flatten_12 = X.flatten(1, 2)
    print(f"  flatten(1, 2): {X.shape} -> {flatten_12.shape}")
    
    flatten_01 = X.flatten(0, 1)
    print(f"  flatten(0, 1): {X.shape} -> {flatten_01.shape}")
    
    print()
    
    # unflatten operations
    print("UNFLATTEN Operations:")
    flat = X.flatten()
    unflat1 = flat.unflatten(0, (2, 10, 16))
    print(f"  unflatten(0, (2,10,16)): {flat.shape} -> {unflat1.shape}")
    
    partial_flat = X.flatten(1, 2)
    unflat2 = partial_flat.unflatten(1, (10, 16))
    print(f"  unflatten(1, (10,16)): {partial_flat.shape} -> {unflat2.shape}")
    
    print()

def demonstrate_expand_repeat():
    """Demonstrate expand and repeat operations"""
    print("=" * 60)
    print("6. EXPAND & REPEAT Operations - Extend tensors")
    print("=" * 60)
    
    # Create small tensor for expansion
    X = torch.arange(6).reshape(1, 2, 3).float()
    print_tensor_info(X, "Original tensor X [1, 2, 3]")
    
    # expand operations - no data copy
    print("EXPAND Operations (no data copy, shared memory):")
    expand1 = X.expand(4, 2, 3)
    print(f"  expand(4, 2, 3): {X.shape} -> {expand1.shape}")
    print(f"  Memory usage: original={X.numel()}, expanded={expand1.numel()}, actual storage={X.numel()}")
    
    # expand can only extend dimensions of size 1, cannot change non-singleton dimensions
    expand2 = X.expand(4, -1, -1)  # -1 means keep original dimension
    print(f"  expand(4, -1, -1): {X.shape} -> {expand2.shape}")
    
    # Demonstrate expand limitations
    print("  Expand limitations:")
    try:
        invalid_expand = X.expand(4, 2, 6)  # Try to expand dimension 2 from 3 to 6
        print(f"    expand(4, 2, 6): success")
    except RuntimeError as e:
        print(f"    expand(4, 2, 6): failed - cannot expand non-singleton dimension")
        print(f"    Error: {str(e)[:60]}...")
    
    # Demonstrate correct expand usage
    print("  Correct expand usage - only expand singleton dimensions:")
    Y = torch.arange(6).reshape(1, 1, 6).float()  # Create tensor with multiple singleton dimensions
    print(f"    Y shape: {Y.shape}")
    expand_Y = Y.expand(3, 4, 6)
    print(f"    Y.expand(3, 4, 6): {Y.shape} -> {expand_Y.shape}")
    
    print()
    
    # repeat operations - copy data
    print("REPEAT Operations (copy data):")
    repeat1 = X.repeat(4, 1, 1)
    print(f"  repeat(4, 1, 1): {X.shape} -> {repeat1.shape}")
    print(f"  Memory usage: original={X.numel()}, repeated={repeat1.numel()}")
    
    repeat2 = X.repeat(2, 3, 2)
    print(f"  repeat(2, 3, 2): {X.shape} -> {repeat2.shape}")
    print(f"  Memory usage: original={X.numel()}, repeated={repeat2.numel()}")
    
    print()
    
    # Key differences demonstration
    print("EXPAND vs REPEAT Key Differences:")
    print("1. expand can only extend dimensions of size 1")
    print("2. repeat can repeat any dimension any number of times")
    print("3. expand shares memory, repeat copies memory")
    print("4. expand saves memory but has usage limitations")
    print()

def demonstrate_memory_layout():
    """Demonstrate memory layout impact"""
    print("=" * 60)
    print("7. Memory Layout and Performance Impact")
    print("=" * 60)
    
    X = torch.arange(320).reshape(2, 10, 16).float()
    
    operations = [
        ("Original", X),
        ("permute(2,0,1)", X.permute(2, 0, 1)),
        ("transpose(0,2)", X.transpose(0, 2)),
        ("contiguous()", X.permute(2, 0, 1).contiguous()),
    ]
    
    for name, tensor in operations:
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Stride: {tensor.stride()}")
        print(f"  Contiguity: {tensor.is_contiguous()}")
        print(f"  Storage size: {tensor.storage().size()}")
        print()

def main():
    print("PyTorch Tensor Operations Comprehensive Learning")
    print("Using large tensor [2, 10, 16] = 320 elements")
    print()
    
    demonstrate_permute()
    demonstrate_transpose()
    demonstrate_view_reshape()
    demonstrate_squeeze_unsqueeze()
    demonstrate_flatten_unflatten()
    demonstrate_expand_repeat()
    demonstrate_memory_layout()
    
    print("=" * 60)
    print("Summary:")
    print("- permute: rearrange all dimensions")
    print("- transpose: swap only two dimensions")
    print("- view: change shape, requires contiguous memory")
    print("- reshape: change shape, more flexible")
    print("- squeeze/unsqueeze: remove/add dimensions of size 1")
    print("- flatten/unflatten: flatten/unflatten specified dimensions")
    print("- expand: extend dimensions, no data copy")
    print("- repeat: duplicate data, copy memory")
    print("=" * 60)

if __name__ == "__main__":
    main()