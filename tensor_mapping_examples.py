#!/usr/bin/env python3
"""
Detailed demonstration of tensor operation mapping relationships
"""

import torch

def demonstrate_permute_mapping():
    print("=" * 60)
    print("PERMUTE Mapping Relationships Explained")
    print("=" * 60)
    
    # Use smaller tensor for easier understanding
    X = torch.arange(24).reshape(2, 3, 4).float()
    print(f"Original tensor X: {X.shape}")
    print("X =")
    print(X)
    print()
    
    # Demonstrate different permute operations
    permute_ops = [
        (2, 0, 1),  # z,x,y
        (1, 2, 0),  # y,z,x
        (0, 2, 1),  # x,z,y
    ]
    
    for perm in permute_ops:
        result = X.permute(*perm)
        print(f"permute{perm}: {X.shape} -> {result.shape}")
        
        # Show mapping relationship
        dim_names = ['x', 'y', 'z']
        old_to_new = {}
        for new_pos, old_pos in enumerate(perm):
            old_to_new[old_pos] = new_pos
        
        mapping = []
        for i in range(3):
            mapping.append(f"{dim_names[i]}(dim{i}) -> dim{old_to_new[i]}")
        
        print(f"  Dimension mapping: {', '.join(mapping)}")
        print(f"  Index mapping: result[i,j,k] = X[{chr(97+perm.index(0))},{chr(97+perm.index(1))},{chr(97+perm.index(2))}]")
        
        # Verify specific positions
        print("  Verification:")
        test_positions = [(0,0,0), (1,2,3), (0,1,2)]
        for pos in test_positions:
            if all(pos[i] < X.shape[i] for i in range(3)):
                x_val = X[pos].item()
                # Calculate position in result
                new_pos = tuple(pos[perm[i]] for i in range(3))
                if all(new_pos[i] < result.shape[i] for i in range(3)):
                    result_val = result[new_pos].item()
                    print(f"    X{pos} = {x_val:.0f} -> result{new_pos} = {result_val:.0f}")
        print()

def demonstrate_transpose_vs_permute():
    print("=" * 60)
    print("TRANSPOSE vs PERMUTE Comparison")
    print("=" * 60)
    
    X = torch.arange(120).reshape(2, 10, 6).float()
    print(f"Original tensor: {X.shape}")
    
    # transpose operations
    t01 = X.transpose(0, 1)  # Swap dimensions 0 and 1
    t02 = X.transpose(0, 2)  # Swap dimensions 0 and 2
    t12 = X.transpose(1, 2)  # Swap dimensions 1 and 2
    
    # Equivalent permute operations
    p01 = X.permute(1, 0, 2)  # Equivalent to transpose(0,1)
    p02 = X.permute(2, 1, 0)  # Equivalent to transpose(0,2)
    p12 = X.permute(0, 2, 1)  # Equivalent to transpose(1,2)
    
    comparisons = [
        ("transpose(0,1)", t01, "permute(1,0,2)", p01),
        ("transpose(0,2)", t02, "permute(2,1,0)", p02),
        ("transpose(1,2)", t12, "permute(0,2,1)", p12),
    ]
    
    for t_name, t_result, p_name, p_result in comparisons:
        print(f"{t_name}: {X.shape} -> {t_result.shape}")
        print(f"{p_name}: {X.shape} -> {p_result.shape}")
        print(f"  Results equal: {torch.equal(t_result, p_result)}")
        print()

def demonstrate_view_reshape_difference():
    print("=" * 60)
    print("VIEW vs RESHAPE Memory Layout Impact")
    print("=" * 60)
    
    X = torch.arange(24).reshape(2, 3, 4).float()
    print(f"Original tensor: {X.shape}, contiguous: {X.is_contiguous()}")
    print(f"Stride: {X.stride()}")
    
    # view and reshape on contiguous tensor
    print("\nContiguous tensor operations:")
    view_result = X.view(6, 4)
    reshape_result = X.reshape(6, 4)
    print(f"view(6,4): success, stride: {view_result.stride()}")
    print(f"reshape(6,4): success, stride: {reshape_result.stride()}")
    print(f"Results equal: {torch.equal(view_result, reshape_result)}")
    
    # Non-contiguous tensor
    print("\nNon-contiguous tensor operations:")
    X_t = X.transpose(0, 2)  # Create non-contiguous tensor
    print(f"After transpose: {X_t.shape}, contiguous: {X_t.is_contiguous()}")
    print(f"Stride: {X_t.stride()}")
    
    try:
        view_result = X_t.view(24)
        print("view(24): success")
    except Exception as e:
        print(f"view(24): failed - {str(e)[:50]}...")
    
    reshape_result = X_t.reshape(24)
    print(f"reshape(24): success, contiguous: {reshape_result.is_contiguous()}")
    
    # Use contiguous() then view
    X_t_cont = X_t.contiguous()
    view_after_cont = X_t_cont.view(24)
    print(f"contiguous().view(24): success")
    print(f"All three methods equal: {torch.equal(reshape_result, view_after_cont)}")

def demonstrate_expand_repeat_difference():
    print("=" * 60)
    print("EXPAND vs REPEAT Memory Usage Comparison")
    print("=" * 60)
    
    # Create small tensor
    X = torch.arange(6).reshape(1, 2, 3).float()
    print(f"Original tensor: {X.shape}")
    print("X =")
    print(X)
    
    # expand operation - no data copy
    expanded = X.expand(3, 2, 3)
    print(f"\nexpand(3,2,3): {X.shape} -> {expanded.shape}")
    print("expanded =")
    print(expanded)
    print(f"Original data pointer: {X.data_ptr()}")
    print(f"Expanded data pointer: {expanded.data_ptr()}")
    print(f"Shares memory: {X.data_ptr() == expanded.data_ptr()}")
    
    # repeat operation - copy data
    repeated = X.repeat(3, 1, 1)
    print(f"\nrepeat(3,1,1): {X.shape} -> {repeated.shape}")
    print("repeated =")
    print(repeated)
    print(f"Repeated data pointer: {repeated.data_ptr()}")
    print(f"Shares memory: {X.data_ptr() == repeated.data_ptr()}")
    
    # Modify original data to see impact
    print(f"\nModify original data X[0,0,0] = 999:")
    X[0, 0, 0] = 999
    print(f"Original: X[0,0,0] = {X[0,0,0].item()}")
    print(f"Expand: expanded[0,0,0] = {expanded[0,0,0].item()}")
    print(f"Repeat: repeated[0,0,0] = {repeated[0,0,0].item()}")

if __name__ == "__main__":
    print("PyTorch Tensor Operation Mapping Relationships Detailed Demo")
    print("Using concrete examples to build intuitive understanding")
    print()
    
    demonstrate_permute_mapping()
    demonstrate_transpose_vs_permute()
    demonstrate_view_reshape_difference()
    demonstrate_expand_repeat_difference()
    
    print("=" * 60)
    print("Key Points:")
    print("1. permute can arbitrarily rearrange dimensions, transpose only swaps two")
    print("2. view requires contiguous memory, reshape is more flexible but may copy data")
    print("3. expand shares memory, repeat copies data")
    print("4. Most operations affect tensor contiguity")
    print("5. Use .contiguous() to reorganize memory layout")
    print("=" * 60)