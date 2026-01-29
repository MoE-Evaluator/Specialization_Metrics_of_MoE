#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算 Rademacher 复杂度

根据 results_all.jsonl 数据，计算专家选择的 Rademacher 复杂度。

步骤：
1. 数据准备：收集 m 个 tokens，记录每个 token 的 Expert indices 和 Expert outputs
2. Monte Carlo 噪声注入：生成 S 个 Rademacher 变量向量 σ ∈ {-1, +1}^m
3. 计算相关性：对每个随机向量，计算 Expert outputs 与随机噪声的对齐程度

注意：领域区分在上层通过不同的输入文件实现，本脚本不进行领域过滤。

用法:
    python rademacher_complexity.py \
        --input_file results_deepseek/expert_statistics/aligned_results.jsonl \
        --num_samples 1000 \
        --num_simulations 1000 \
        --shared_expert_id 4 \
        --output_file results_deepseek/expert_statistics/rademacher_complexity.json
"""

import json
import os
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from collections import defaultdict


def load_aligned_data(input_file: str) -> List[Dict]:
    """
    加载对齐后的数据
    
    Args:
        input_file: aligned_results.jsonl 文件路径
    
    Returns:
        数据记录列表
    """
    print(f"正在加载数据: {input_file}")
    records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行JSON解析失败: {e}")
                continue
    
    print(f"加载完成: {len(records)} 条记录")
    return records




def extract_expert_data_by_layer(record: Dict, use_weights: bool = True, use_logits: bool = False) -> Dict[str, Dict]:
    """
    从记录中按层提取专家数据
    
    Args:
        record: 数据记录
        use_weights: 是否使用 weights 作为 expert outputs
        use_logits: 是否使用 logits 作为 expert outputs
    
    Returns:
        字典，格式为 {layer_id: {'expert_indices': [...], 'expert_outputs': [...]}}
    """
    layers = record.get('layers', {})
    layer_data_dict = {}
    
    for layer_id, layer_data in layers.items():
        expert_ids = layer_data.get('expert_ids', [])
        weights = layer_data.get('weights', [])
        logits = layer_data.get('logits', [])
        
        # 确定使用哪个作为 expert output
        if use_logits and logits:
            outputs = logits
        elif use_weights and weights:
            outputs = weights
        else:
            # 如果都没有，使用 expert_ids 的 one-hot 编码（1.0）
            outputs = [1.0] * len(expert_ids)
        
        # 确保长度一致
        min_len = min(len(expert_ids), len(outputs))
        
        expert_indices = []
        expert_outputs = []
        
        for i in range(min_len):
            expert_id = expert_ids[i]
            output = float(outputs[i])
            expert_indices.append(expert_id)
            expert_outputs.append(output)
        
        layer_data_dict[layer_id] = {
            'expert_indices': expert_indices,
            'expert_outputs': expert_outputs
        }
    
    return layer_data_dict


def prepare_data_by_layer(records: List[Dict], num_samples: Optional[int] = None,
                          use_weights: bool = True, use_logits: bool = False,
                          random_seed: Optional[int] = None) -> tuple:
    """
    准备数据：按层提取每个 token 的 Expert indices 和 Expert outputs
    
    Args:
        records: 数据记录列表
        num_samples: 采样数量，如果为None则使用所有数据
        use_weights: 是否使用 weights 作为 expert outputs
        use_logits: 是否使用 logits 作为 expert outputs
        random_seed: 随机种子（用于采样）
    
    Returns:
        (prepared_data_by_layer, sampled_records): 
        - prepared_data_by_layer: 字典，格式为 {layer_id: [{'expert_indices': [...], 'expert_outputs': [...]}, ...]}
        - sampled_records: 对应的原始记录列表
    """
    print(f"\n准备数据（采样数量: {num_samples if num_samples else '全部'}）...")
    
    # 采样
    if num_samples is not None and num_samples < len(records):
        if random_seed is not None:
            np.random.seed(random_seed)
        indices = np.random.choice(len(records), size=num_samples, replace=False)
        sampled_records = [records[i] for i in indices]
    else:
        sampled_records = records
    
    # 按层组织数据
    prepared_data_by_layer = defaultdict(list)
    
    for record in tqdm(sampled_records, desc="提取专家数据（按层）"):
        layer_data_dict = extract_expert_data_by_layer(record, use_weights, use_logits)
        for layer_id, expert_data in layer_data_dict.items():
            prepared_data_by_layer[layer_id].append(expert_data)
    
    print(f"数据准备完成: {len(sampled_records)} 个 tokens")
    print(f"层数: {len(prepared_data_by_layer)}")
    for layer_id in sorted(prepared_data_by_layer.keys(), key=int):
        print(f"  层 {layer_id}: {len(prepared_data_by_layer[layer_id])} 个 tokens")
    
    return dict(prepared_data_by_layer), sampled_records


def generate_rademacher_vectors(m: int, S: int, random_seed: Optional[int] = None) -> np.ndarray:
    """
    生成 Rademacher 变量向量
    
    Args:
        m: 样本数量
        S: 模拟次数
        random_seed: 随机种子
    
    Returns:
        Rademacher 变量矩阵，形状为 (S, m)，每个元素为 -1 或 +1
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 生成随机向量，每个元素为 -1 或 +1
    rademacher_vectors = np.random.choice([-1, 1], size=(S, m))
    
    return rademacher_vectors


def calculate_rademacher_complexity_by_layer(prepared_data_by_layer: Dict[str, List[Dict]], 
                                            S: int = 1000, random_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    按层计算 Rademacher 复杂度
    
    对于每一层，对于每个专家 j，计算：
    R_{l,j} = E_σ [|(1/m) * sum_i (σ_i · e_{i,j})|]
    
    其中：
    - l: 层索引
    - m: 样本数量
    - σ_i: Rademacher 变量（-1 或 +1）
    - e_{i,j}: 专家 j 在 token x_i 上的输出
    - E_σ: 对所有 Rademacher 向量求期望（通过 Monte Carlo 模拟）
    
    Args:
        prepared_data_by_layer: 按层组织的数据，格式为 {layer_id: [{'expert_indices': [...], 'expert_outputs': [...]}, ...]}
        S: Monte Carlo 模拟次数
        random_seed: 随机种子
    
    Returns:
        包含按层 Rademacher 复杂度结果的字典，以及矩阵数据
    """
    print(f"\n按层计算 Rademacher 复杂度:")
    print(f"  模拟次数 S: {S}")
    
    layer_results = {}
    all_layer_ids = sorted(prepared_data_by_layer.keys(), key=int)
    
    # 收集所有层的所有专家ID，以确定矩阵大小
    all_expert_ids_set = set()
    for layer_id in all_layer_ids:
        layer_data = prepared_data_by_layer[layer_id]
        for data in layer_data:
            all_expert_ids_set.update(data['expert_indices'])
    all_expert_ids = sorted(all_expert_ids_set)
    num_experts = len(all_expert_ids)
    
    print(f"  总层数: {len(all_layer_ids)}")
    print(f"  总专家数: {num_experts}")
    
    # 为每一层计算 Rademacher 复杂度
    for layer_id in tqdm(all_layer_ids, desc="处理各层"):
        layer_data = prepared_data_by_layer[layer_id]
        m = len(layer_data)
        
        # 提取该层的所有唯一专家ID
        layer_expert_ids = set()
        for data in layer_data:
            layer_expert_ids.update(data['expert_indices'])
        layer_expert_ids = sorted(layer_expert_ids)
        
        # 为每个 token 构建 expert output 向量（固定维度，对应所有专家）
        expert_output_vectors = []
        for data in layer_data:
            expert_indices = data['expert_indices']
            expert_outputs = data['expert_outputs']
            
            # 创建固定维度的向量（对应所有可能的专家）
            vector = np.zeros(num_experts)
            for expert_id, output in zip(expert_indices, expert_outputs):
                if expert_id in all_expert_ids:
                    idx = all_expert_ids.index(expert_id)
                    vector[idx] = output
            
            expert_output_vectors.append(vector)
        
        expert_output_matrix = np.array(expert_output_vectors)  # 形状: (m, num_experts)
        
        # 生成 Rademacher 变量向量
        rademacher_vectors = generate_rademacher_vectors(m, S, random_seed)
        
        # 对每个随机向量计算相关性
        correlations = []
        
        for s in range(S):
            sigma = rademacher_vectors[s]  # 形状: (m,)
            
            # 计算每个专家的相关性
            expert_correlations = []
            for expert_idx in range(num_experts):
                expert_outputs = expert_output_matrix[:, expert_idx]  # 形状: (m,)
                # σ_i · e_i 的归一化求和
                correlation = np.mean(sigma * expert_outputs)
                expert_correlations.append(correlation)
            
            correlations.append(expert_correlations)
        
        correlations = np.array(correlations)  # 形状: (S, num_experts)
        
        # 对所有 S 次模拟求平均（对每个专家）
        mean_correlations = np.mean(correlations, axis=0)  # 形状: (num_experts,)
        
        # 计算每个专家的 Rademacher 复杂度（取绝对值）
        expert_rademacher = {expert_id: float(np.abs(complexity)) 
                            for expert_id, complexity in zip(all_expert_ids, mean_correlations)}
        
        # 计算标准差
        std_correlations = np.std(correlations, axis=0)
        expert_std = {expert_id: float(std) 
                     for expert_id, std in zip(all_expert_ids, std_correlations)}
        
        # 该层的全局 Rademacher 复杂度
        layer_rademacher_complexity = np.max(np.abs(mean_correlations))
        
        # 计算每个专家的 Rademacher 复杂度（取绝对值）
        rademacher_matrix_row = [float(np.abs(corr)) for corr in mean_correlations]
        
        layer_results[layer_id] = {
            'rademacher_complexity': float(layer_rademacher_complexity),
            'num_samples': m,
            'expert_rademacher': expert_rademacher,  # 每个专家的复杂度字典
            'expert_std': expert_std,
            'mean_correlations': {expert_id: float(corr) 
                                for expert_id, corr in zip(all_expert_ids, mean_correlations)},
            'rademacher_matrix_row': rademacher_matrix_row  # 用于构建矩阵，包含所有专家的复杂度
        }
        
        # 打印该层的统计信息
        non_zero_count = sum(1 for val in rademacher_matrix_row if val > 1e-8)
        print(f"  层 {layer_id}: {non_zero_count}/{num_experts} 个专家有非零复杂度")
    
    # 构建矩阵：行是层，列是专家
    rademacher_matrix = np.zeros((len(all_layer_ids), num_experts))
    for i, layer_id in enumerate(all_layer_ids):
        rademacher_matrix[i, :] = layer_results[layer_id]['rademacher_matrix_row']
    
    results = {
        'num_simulations': S,
        'num_experts': num_experts,
        'all_expert_ids': all_expert_ids,
        'all_layer_ids': all_layer_ids,
        'layer_results': layer_results,
        'rademacher_matrix': rademacher_matrix.tolist()  # 转换为列表以便 JSON 序列化
    }
    
    return results


def extract_shared_expert_output_by_layer(record: Dict,
                                          use_weights: bool = True, use_logits: bool = False) -> Dict[str, float]:
    """
    从记录中按层提取 shared expert 的输出
    
    注意：shared expert 默认被选择，不在 expert_ids 中（expert_ids 只包含路由专家）。
    我们使用该层的聚合输出作为 shared expert 的输出，这相当于 Dense 模型的输出
    （所有 token 都经过 shared expert）。
    
    Args:
        record: 数据记录
        use_weights: 是否使用 weights 作为 expert outputs
        use_logits: 是否使用 logits 作为 expert outputs
    
    Returns:
        字典，格式为 {layer_id: output_value}
    """
    layers = record.get('layers', {})
    layer_outputs = {}
    
    for layer_id, layer_data in layers.items():
        weights = layer_data.get('weights', [])
        logits = layer_data.get('logits', [])
        
        # shared expert 不在 expert_ids 中，使用该层的聚合输出
        # 这相当于 Dense 模型的输出（所有 token 都经过 shared expert）
        if use_weights and weights:
            # 使用所有 weights 的平均值
            output = float(np.mean(weights)) if weights else 1.0
        elif use_logits and logits:
            # 使用所有 logits 的平均值
            output = float(np.mean(logits)) if logits else 1.0
        else:
            # 默认值：shared expert 总是贡献 1.0
            output = 1.0
        
        layer_outputs[layer_id] = output
    
    return layer_outputs


def calculate_shared_expert_rademacher_by_layer(records: List[Dict],
                                                S: int = 1000, random_seed: Optional[int] = None,
                                                use_weights: bool = True, use_logits: bool = False) -> Dict[str, Any]:
    """
    按层计算 shared expert 的 Rademacher 复杂度
    
    由于 shared expert 在所有 token 上都会被选择（默认全选，类似 Dense 模型），
    我们按层分别计算所有 token 的 shared expert output。
    
    注意：shared expert 不在 expert_ids 中（expert_ids 只包含路由专家），
    我们使用该层的聚合输出作为 shared expert 的输出。
    
    Args:
        records: 原始数据记录列表
        S: Monte Carlo 模拟次数
        random_seed: 随机种子
        use_weights: 是否使用 weights 作为 expert outputs
        use_logits: 是否使用 logits 作为 expert outputs
    
    Returns:
        包含按层 shared expert Rademacher 复杂度结果的字典
    """
    m = len(records)
    print(f"\n按层计算 Shared Expert 的 Rademacher 复杂度（类似 Dense 模型）:")
    print(f"  样本数量 m: {m}")
    print(f"  模拟次数 S: {S}")
    print(f"  注意: Shared expert 默认被选择，不在 expert_ids 中（expert_ids 只包含路由专家）")
    
    # 按层提取所有 token 的 shared expert output
    print("提取 shared expert 输出（按层）...")
    shared_expert_outputs_by_layer = defaultdict(list)
    
    for record in tqdm(records, desc="提取 shared expert 数据"):
        layer_outputs = extract_shared_expert_output_by_layer(
            record, use_weights, use_logits
        )
        for layer_id, output in layer_outputs.items():
            shared_expert_outputs_by_layer[layer_id].append(output)
    
    # 为每一层计算 Rademacher 复杂度
    layer_results = {}
    all_layer_ids = sorted(shared_expert_outputs_by_layer.keys(), key=int)
    
    print(f"\n计算各层的 Rademacher 复杂度...")
    for layer_id in tqdm(all_layer_ids, desc="处理各层"):
        layer_outputs = np.array(shared_expert_outputs_by_layer[layer_id])
        layer_m = len(layer_outputs)
        
        # 生成 Rademacher 变量向量
        rademacher_vectors = generate_rademacher_vectors(layer_m, S, random_seed)
        
        # 对每个随机向量计算相关性
        correlations = []
        for s in range(S):
            sigma = rademacher_vectors[s]  # 形状: (layer_m,)
            # 计算 (1/m) * sum_i (σ_i · e_i)，其中 e_i 是 shared expert 在 token i 上的输出
            correlation = np.mean(sigma * layer_outputs)
            correlations.append(correlation)
        
        correlations = np.array(correlations)  # 形状: (S,)
        
        # 对所有 S 次模拟求平均
        mean_correlation = np.mean(correlations)
        
        # 计算 Rademacher 复杂度（取绝对值）
        rademacher_complexity = np.abs(mean_correlation)
        
        # 计算标准差
        std_correlation = np.std(correlations)
        
        layer_results[layer_id] = {
            'rademacher_complexity': float(rademacher_complexity),
            'mean_correlation': float(mean_correlation),
            'std_correlation': float(std_correlation),
            'num_samples': layer_m
        }
    
    results = {
        'num_simulations': S,
        'all_layer_ids': all_layer_ids,
        'layer_results': layer_results
    }
    
    return results


def save_rademacher_matrix_csv(results: Dict[str, Any], input_file: str):
    """
    保存 Rademacher 复杂度矩阵为 CSV 文件
    
    矩阵格式：行是层，列是专家
    
    Args:
        results: 包含 rademacher_matrix 的结果字典
        input_file: 输入文件路径（用于确定输出目录）
    """
    import csv
    
    input_dir = os.path.dirname(input_file)
    if not input_dir:
        input_dir = '.'
    
    csv_file = os.path.join(input_dir, 'rademacher_complexity.csv')
    
    rademacher_matrix = np.array(results['rademacher_matrix'])
    all_layer_ids = results['all_layer_ids']
    all_expert_ids = results['all_expert_ids']
    
    # 写入 CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头：第一行是专家ID
        header = ['Layer'] + [f'Expert_{eid}' for eid in all_expert_ids]
        writer.writerow(header)
        
        # 写入每一层的数据
        for i, layer_id in enumerate(all_layer_ids):
            row = [f'Layer_{layer_id}'] + [f'{val:.8f}' for val in rademacher_matrix[i, :]]
            writer.writerow(row)
    
    print(f"\nRademacher 复杂度矩阵已保存到: {csv_file}")
    print(f"  矩阵大小: {len(all_layer_ids)} 层 x {len(all_expert_ids)} 专家")
    print(f"  包含的专家ID: {all_expert_ids}")
    
    # 显示每层每个专家的复杂度摘要（只显示非零值）
    print(f"\n各层专家复杂度摘要（仅显示非零值）:")
    for layer_id in all_layer_ids:
        layer_result = results['layer_results'][layer_id]
        expert_rademacher = layer_result['expert_rademacher']
        non_zero_experts = {eid: val for eid, val in expert_rademacher.items() if val > 1e-8}
        if non_zero_experts:
            print(f"  层 {layer_id}: {len(non_zero_experts)} 个专家有非零复杂度")
            # 显示前5个最大的
            sorted_experts = sorted(non_zero_experts.items(), key=lambda x: x[1], reverse=True)
            for expert_id, complexity in sorted_experts[:5]:
                print(f"    Expert {expert_id}: {complexity:.6f}")


def save_results(results: Dict[str, Any], output_file: str):
    """
    保存结果到 JSON 文件
    
    Args:
        results: 结果字典
        output_file: 输出文件路径
    """
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='计算 Rademacher 复杂度')
    parser.add_argument('--input_file', type=str, required=True,
                       help='results_all.jsonl 文件路径')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='采样数量，如果为None则使用所有数据')
    parser.add_argument('--num_simulations', type=int, default=1000,
                       help='Monte Carlo 模拟次数 S')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--use_weights', action='store_true', default=True,
                       help='使用 weights 作为 expert outputs')
    parser.add_argument('--use_logits', action='store_true', default=False,
                       help='使用 logits 作为 expert outputs（如果同时指定，优先使用 logits）')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 步骤1: 加载数据
    print("=" * 80)
    print("步骤1: 加载数据")
    print("=" * 80)
    records = load_aligned_data(args.input_file)
    
    # 步骤2: 准备数据（按层）
    print("\n" + "=" * 80)
    print("步骤2: 准备数据（按层）")
    print("=" * 80)
    prepared_data_by_layer, sampled_records = prepare_data_by_layer(
        records,
        num_samples=args.num_samples,
        use_weights=args.use_weights,
        use_logits=args.use_logits,
        random_seed=args.random_seed
    )
    
    # 步骤3: 按层计算所有专家的 Rademacher 复杂度
    print("\n" + "=" * 80)
    print("步骤3: 按层计算所有专家的 Rademacher 复杂度")
    print("=" * 80)
    results = calculate_rademacher_complexity_by_layer(
        prepared_data_by_layer,
        S=args.num_simulations,
        random_seed=args.random_seed
    )
    
    # 步骤4: 按层计算 Shared Expert 的 Rademacher 复杂度
    print("\n" + "=" * 80)
    print("步骤4: 按层计算 Shared Expert 的 Rademacher 复杂度")
    print("=" * 80)
    shared_expert_results = calculate_shared_expert_rademacher_by_layer(
        sampled_records,
        S=args.num_simulations,
        random_seed=args.random_seed,
        use_weights=args.use_weights,
        use_logits=args.use_logits
    )
    
    # 合并结果
    results['shared_expert'] = shared_expert_results
    
    # 步骤5: 保存 CSV 矩阵
    print("\n" + "=" * 80)
    print("步骤5: 保存 Rademacher 复杂度矩阵（CSV）")
    print("=" * 80)
    save_rademacher_matrix_csv(results, args.input_file)
    
    # 步骤6: 保存 JSON 结果
    print("\n" + "=" * 80)
    print("步骤6: 保存详细结果（JSON）")
    print("=" * 80)
    save_results(results, args.output_file)
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("结果摘要")
    print("=" * 80)
    print(f"模拟次数: {results['num_simulations']}")
    print(f"总专家数: {results['num_experts']}")
    print(f"总层数: {len(results['all_layer_ids'])}")
    
    # 显示每层的 Rademacher 复杂度
    print(f"\n各层的 Rademacher 复杂度:")
    for layer_id in results['all_layer_ids']:
        layer_result = results['layer_results'][layer_id]
        print(f"  层 {layer_id}: {layer_result['rademacher_complexity']:.6f} (样本数: {layer_result['num_samples']})")
    
    # 显示 Shared Expert 各层的 Rademacher 复杂度
    print(f"\nShared Expert 各层的 Rademacher 复杂度:")
    for layer_id in shared_expert_results['all_layer_ids']:
        layer_result = shared_expert_results['layer_results'][layer_id]
        print(f"  层 {layer_id}: {layer_result['rademacher_complexity']:.6f} (样本数: {layer_result['num_samples']})")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

