#!/usr/bin/env python3
"""
张量操作映射关系详细演示
"""

import torch

def demonstrate_permute_mapping():
    print("=" * 60)
    print("PERMUTE 映射关系详解")
    print("=" * 60)
    
    # 使用较小的张量便于理解
    X = torch.arange(24).reshape(2, 3, 4).float()
    print(f"原始张量 X: {X.shape}")
    print("X =")
    print(X)
    print()
    
    # 演示不同的permute操作
    permute_ops = [
        (2, 0, 1),  # z,x,y
        (1, 2, 0),  # y,z,x
        (0, 2, 1),  # x,z,y
    ]
    
    for perm in permute_ops:
        result = X.permute(*perm)
        print(f"permute{perm}: {X.shape} -> {result.shape}")
        
        # 显示映射关系
        dim_names = ['x', 'y', 'z']
        old_to_new = {}
        for new_pos, old_pos in enumerate(perm):
            old_to_new[old_pos] = new_pos
        
        mapping = []
        for i in range(3):
            mapping.append(f"{dim_names[i]}(dim{i}) -> dim{old_to_new[i]}")
        
        print(f"  维度映射: {', '.join(mapping)}")
        print(f"  索引映射: result[i,j,k] = X[{chr(97+perm.index(0))},{chr(97+perm.index(1))},{chr(97+perm.index(2))}]")
        
        # 验证几个具体位置
        print("  验证:")
        test_positions = [(0,0,0), (1,2,3), (0,1,2)]
        for pos in test_positions:
            if all(pos[i] < X.shape[i] for i in range(3)):
                x_val = X[pos].item()
                # 计算在result中的位置
                new_pos = tuple(pos[perm[i]] for i in range(3))
                if all(new_pos[i] < result.shape[i] for i in range(3)):
                    result_val = result[new_pos].item()
                    print(f"    X{pos} = {x_val:.0f} -> result{new_pos} = {result_val:.0f}")
        print()

def demonstrate_transpose_vs_permute():
    print("=" * 60)
    print("TRANSPOSE vs PERMUTE 对比")
    print("=" * 60)
    
    X = torch.arange(120).reshape(2, 10, 6).float()
    print(f"原始张量: {X.shape}")
    
    # transpose操作
    t01 = X.transpose(0, 1)  # 交换维度0和1
    t02 = X.transpose(0, 2)  # 交换维度0和2
    t12 = X.transpose(1, 2)  # 交换维度1和2
    
    # 等价的permute操作
    p01 = X.permute(1, 0, 2)  # 等价于transpose(0,1)
    p02 = X.permute(2, 1, 0)  # 等价于transpose(0,2)
    p12 = X.permute(0, 2, 1)  # 等价于transpose(1,2)
    
    comparisons = [
        ("transpose(0,1)", t01, "permute(1,0,2)", p01),
        ("transpose(0,2)", t02, "permute(2,1,0)", p02),
        ("transpose(1,2)", t12, "permute(0,2,1)", p12),
    ]
    
    for t_name, t_result, p_name, p_result in comparisons:
        print(f"{t_name}: {X.shape} -> {t_result.shape}")
        print(f"{p_name}: {X.shape} -> {p_result.shape}")
        print(f"  结果相同: {torch.equal(t_result, p_result)}")
        print()

def demonstrate_view_reshape_difference():
    print("=" * 60)
    print("VIEW vs RESHAPE 内存布局影响")
    print("=" * 60)
    
    X = torch.arange(24).reshape(2, 3, 4).float()
    print(f"原始张量: {X.shape}, 连续性: {X.is_contiguous()}")
    print(f"步长: {X.stride()}")
    
    # 连续张量的view和reshape
    print("\n连续张量操作:")
    view_result = X.view(6, 4)
    reshape_result = X.reshape(6, 4)
    print(f"view(6,4): 成功, 步长: {view_result.stride()}")
    print(f"reshape(6,4): 成功, 步长: {reshape_result.stride()}")
    print(f"结果相同: {torch.equal(view_result, reshape_result)}")
    
    # 非连续张量
    print("\n非连续张量操作:")
    X_t = X.transpose(0, 2)  # 创建非连续张量
    print(f"转置后: {X_t.shape}, 连续性: {X_t.is_contiguous()}")
    print(f"步长: {X_t.stride()}")
    
    try:
        view_result = X_t.view(24)
        print("view(24): 成功")
    except Exception as e:
        print(f"view(24): 失败 - {str(e)[:50]}...")
    
    reshape_result = X_t.reshape(24)
    print(f"reshape(24): 成功, 连续性: {reshape_result.is_contiguous()}")
    
    # 使用contiguous()后再view
    X_t_cont = X_t.contiguous()
    view_after_cont = X_t_cont.view(24)
    print(f"contiguous().view(24): 成功")
    print(f"三种方法结果相同: {torch.equal(reshape_result, view_after_cont)}")

def demonstrate_expand_repeat_difference():
    print("=" * 60)
    print("EXPAND vs REPEAT 内存使用对比")
    print("=" * 60)
    
    # 创建小张量
    X = torch.arange(6).reshape(1, 2, 3).float()
    print(f"原始张量: {X.shape}")
    print("X =")
    print(X)
    
    # expand操作 - 不复制数据
    expanded = X.expand(3, 2, 3)
    print(f"\nexpand(3,2,3): {X.shape} -> {expanded.shape}")
    print("expanded =")
    print(expanded)
    print(f"原始数据指针: {X.data_ptr()}")
    print(f"扩展后数据指针: {expanded.data_ptr()}")
    print(f"共享内存: {X.data_ptr() == expanded.data_ptr()}")
    
    # repeat操作 - 复制数据
    repeated = X.repeat(3, 1, 1)
    print(f"\nrepeat(3,1,1): {X.shape} -> {repeated.shape}")
    print("repeated =")
    print(repeated)
    print(f"重复后数据指针: {repeated.data_ptr()}")
    print(f"共享内存: {X.data_ptr() == repeated.data_ptr()}")
    
    # 修改原始数据看影响
    print(f"\n修改原始数据 X[0,0,0] = 999:")
    X[0, 0, 0] = 999
    print(f"原始: X[0,0,0] = {X[0,0,0].item()}")
    print(f"expand: expanded[0,0,0] = {expanded[0,0,0].item()}")
    print(f"repeat: repeated[0,0,0] = {repeated[0,0,0].item()}")

if __name__ == "__main__":
    print("PyTorch张量操作映射关系详细演示")
    print("使用具体例子建立直觉理解")
    print()
    
    demonstrate_permute_mapping()
    demonstrate_transpose_vs_permute()
    demonstrate_view_reshape_difference()
    demonstrate_expand_repeat_difference()
    
    print("=" * 60)
    print("关键要点:")
    print("1. permute可以任意重排维度，transpose只能交换两个维度")
    print("2. view要求连续内存，reshape更灵活但可能复制数据")
    print("3. expand共享内存，repeat复制数据")
    print("4. 大多数操作会影响张量的连续性")
    print("5. 使用.contiguous()可以重新整理内存布局")
    print("=" * 60)