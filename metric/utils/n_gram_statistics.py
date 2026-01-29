#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
n-gram 路径统计脚本

功能：
1. 从 results_all.jsonl 加载数据
2. 按 request_id 分组，提取每一层的专家序列
3. 计算 n-gram 路径统计：
   - 所有 n-gram 路径总数
   - 专家重复路由数量（从expert A到expert A）
   - 重复路由占所有路径的比例
4. 为每一层单独计算统计
5. 生成转移矩阵图像和统计报告

用法:
    python n_gram_statistics.py \
        --input_file results_deepseek/expert_statistics/results_all.jsonl \
        --n 2 \
        --output_dir results_deepseek/expert_statistics/n_gram_stats
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from tqdm import tqdm

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_jsonl_data(input_file: str) -> List[Dict]:
    """
    加载JSONL文件数据
    
    Args:
        input_file: JSONL文件路径
    
    Returns:
        数据列表
    """
    print(f"正在加载数据: {input_file}")
    records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                records.append(data)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行JSON解析失败: {e}")
                continue
    
    print(f"加载完成: {len(records)} 条记录")
    return records


def extract_expert_sequences_by_request(records: List[Dict]) -> Dict[int, Dict[int, List[int]]]:
    """
    按request_id分组，提取每一层的专家序列（使用每个token的第一个expert_id）
    
    注意：按照数据在原始文件中的自然顺序处理，不对token_index进行排序
    
    Args:
        records: 记录列表（每行是一个token的记录，按自然顺序）
    
    Returns:
        字典: {request_id: {layer_idx: [expert_id, ...]}}
    """
    # 按request_id分组，保持记录在原始数据中的顺序
    request_sequences = defaultdict(lambda: defaultdict(list))
    
    print("\n正在提取专家序列（按自然顺序，不排序）...")
    for record in tqdm(records, desc="处理记录"):
        request_id = record.get('request_id')
        if request_id is None:
            continue
        
        layers = record.get('layers', {})
        if not layers:
            continue
        
        # 遍历每一层
        for layer_key, layer_data in layers.items():
            try:
                layer_idx = int(layer_key)
            except (ValueError, TypeError):
                continue
            
            expert_ids = layer_data.get('expert_ids', [])
            if expert_ids and len(expert_ids) > 0:
                # 使用第一个expert_id作为该token在该层的主要专家
                primary_expert = expert_ids[0]
                # 按照记录在原始数据中的顺序追加
                request_sequences[request_id][layer_idx].append(primary_expert)
    
    print(f"提取了 {len(request_sequences)} 个请求的专家序列")
    return dict(request_sequences)


def calculate_ngram_statistics(
    request_sequences: Dict[int, Dict[int, List[int]]],
    n: int
) -> Dict[int, Dict[str, Any]]:
    """
    计算每一层的n-gram统计信息
    
    使用滑动窗口方式统计：从每个序列的每个可能位置开始，每次滑动1个token，
    提取n-gram路径并统计转移。
    
    Args:
        request_sequences: 按request分组的专家序列
        n: n-gram的n值
    
    Returns:
        字典: {layer_idx: {
            'total_ngram_paths': int,  # 所有n-gram路径总数
            'unique_ngram_paths': int,  # 唯一n-gram路径数量（set）
            'total_transitions': int,  # 所有相邻expert对（路由）的总数
            'unique_transitions': int,  # 唯一转移路径数量（set）
            'self_loop_transitions': int,  # 专家重复路由数量（从A到A）
            'self_loop_ratio': float,  # 重复路由比例（self_loop_transitions / total_transitions）
            'transition_matrix': np.ndarray,  # 转移矩阵（source_expert x target_expert）
            'transition_counts': Dict[Tuple[int, int], int],  # 转移计数
        }}
    """
    print(f"\n正在计算 {n}-gram 统计信息（滑动窗口，每次滑动1个token）...")
    
    # 首先确定所有层和所有专家
    all_layers = set()
    all_experts = set()
    
    for request_data in request_sequences.values():
        for layer_idx, expert_seq in request_data.items():
            all_layers.add(layer_idx)
            all_experts.update(expert_seq)
    
    num_layers = max(all_layers) + 1 if all_layers else 0
    num_experts = max(all_experts) + 1 if all_experts else 0
    
    print(f"检测到 {num_layers} 层，{num_experts} 个专家")
    
    # 统计每一层的数据
    layer_statistics = {}
    
    for layer_idx in range(num_layers):
        # 统计该层所有request的n-gram路径和相邻expert对
        total_ngram_paths = 0
        unique_ngram_paths_set = set()  # 用于统计唯一的n-gram路径
        total_transitions = 0
        unique_transitions_set = set()  # 用于统计唯一的转移路径
        self_loop_transitions = 0
        transition_counts = Counter()  # (source, target) -> count
        
        # 双重循环：遍历每个request的序列，对每个序列做滑动窗口
        for request_id, layer_data in request_sequences.items():
            expert_seq = layer_data.get(layer_idx, [])
            
            if len(expert_seq) < n:
                continue
            
            # 滑动窗口提取n-gram路径：从序列的每个可能位置开始，每次滑动1个token
            for start in range(len(expert_seq) - n + 1):
                ngram = tuple(expert_seq[start:start+n])
                total_ngram_paths += 1
                unique_ngram_paths_set.add(ngram)  # 添加到set中统计唯一路径
                
                # 对于n-gram，只统计从第一个到最后一个的转移（而不是所有相邻对）
                source = ngram[0]
                target = ngram[n - 1]
                transition = (source, target)
                
                transition_counts[transition] += 1
                total_transitions += 1
                unique_transitions_set.add(transition)  # 添加到set中统计唯一转移
                
                # 只有当整个n-gram的所有expert都相同时，才计数为self-loop
                if len(set(ngram)) == 1:  # 所有expert都相同
                    self_loop_transitions += 1
        
        # 计算比例（重复路由占所有路由的比例）
        self_loop_ratio = self_loop_transitions / total_transitions if total_transitions > 0 else 0.0
        
        # 构建转移矩阵
        transition_matrix = np.zeros((num_experts, num_experts), dtype=np.int64)
        for (source, target), count in transition_counts.items():
            if 0 <= source < num_experts and 0 <= target < num_experts:
                transition_matrix[source, target] = count
        
        layer_statistics[layer_idx] = {
            'total_ngram_paths': total_ngram_paths,
            'unique_ngram_paths': len(unique_ngram_paths_set),
            'total_transitions': total_transitions,
            'unique_transitions': len(unique_transitions_set),
            'self_loop_transitions': self_loop_transitions,
            'self_loop_ratio': self_loop_ratio,
            'transition_matrix': transition_matrix,
            'transition_counts': dict(transition_counts),
        }
        
        print(f"层 {layer_idx}: n-gram路径={total_ngram_paths} (唯一={len(unique_ngram_paths_set)}), "
              f"总路由={total_transitions} (唯一={len(unique_transitions_set)}), "
              f"重复路由={self_loop_transitions}, 比例={self_loop_ratio:.4f}")
    
    return layer_statistics


def visualize_transition_matrix(
    transition_matrix: np.ndarray,
    layer_idx: int,
    output_dir: str,
    save_formats: List[str] = None,
    title_suffix: str = ""
) -> List[str]:
    """
    可视化转移矩阵
    
    Args:
        transition_matrix: 转移矩阵（source_expert x target_expert）
        layer_idx: 层索引
        output_dir: 输出目录
        save_formats: 保存格式列表
        title_suffix: 标题后缀
    
    Returns:
        保存的文件路径列表
    """
    if save_formats is None:
        save_formats = ['png', 'pdf']
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_experts = transition_matrix.shape[0]
    
    # 创建热力图
    fig_width = max(12, num_experts * 0.3)
    fig_height = max(8, num_experts * 0.3)
    plt.figure(figsize=(fig_width, fig_height))
    
    # 使用对数尺度显示（如果最大值较大）
    matrix_to_plot = transition_matrix.astype(float)
    if matrix_to_plot.max() > 100:
        matrix_to_plot = np.log1p(matrix_to_plot)
        log_scale_note = " (log scale)"
    else:
        log_scale_note = ""
    
    sns.heatmap(
        matrix_to_plot,
        cmap='YlOrRd',
        xticklabels=[f'E{i}' for i in range(num_experts)],
        yticklabels=[f'E{i}' for i in range(num_experts)],
        cbar=True,
        cbar_kws={'label': f'Transition Count{log_scale_note}'},
        fmt='.0f',
        annot=False,  # 如果矩阵太大，不标注数值
    )
    
    title = f'Layer {layer_idx} Expert Transition Matrix{title_suffix}{log_scale_note}'
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Target Expert', fontsize=12)
    plt.ylabel('Source Expert', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存为多种格式
    figure_paths = []
    base_name = os.path.join(output_dir, f'layer{layer_idx}_transition_matrix{title_suffix.replace(" ", "_")}')
    for fmt in save_formats:
        file_path = f'{base_name}.{fmt}'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        figure_paths.append(file_path)
    
    plt.close()
    return figure_paths


def visualize_self_loop_ratio(
    layer_statistics: Dict[int, Dict[str, Any]],
    output_dir: str,
    save_formats: List[str] = None
) -> List[str]:
    """
    可视化各层的重复路由比例
    
    Args:
        layer_statistics: 层统计信息
        output_dir: 输出目录
        save_formats: 保存格式列表
    
    Returns:
        保存的文件路径列表
    """
    if save_formats is None:
        save_formats = ['png', 'pdf']
    
    os.makedirs(output_dir, exist_ok=True)
    
    layers = sorted(layer_statistics.keys())
    ratios = [layer_statistics[layer]['self_loop_ratio'] for layer in layers]
    total_transitions = [layer_statistics[layer]['total_transitions'] for layer in layers]
    self_loop_transitions = [layer_statistics[layer]['self_loop_transitions'] for layer in layers]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: 重复路由比例
    ax1.bar(layers, ratios, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Self-Loop Ratio', fontsize=12)
    ax1.set_title('Expert Self-Loop Ratio by Layer', fontsize=14)
    ax1.set_xticks(layers)
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱状图上标注数值
    for layer, ratio in zip(layers, ratios):
        ax1.text(layer, ratio, f'{ratio:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 子图2: 路由数量
    x = np.arange(len(layers))
    width = 0.35
    ax2.bar(x - width/2, total_transitions, width, label='Total Transitions', color='coral', alpha=0.7)
    ax2.bar(x + width/2, self_loop_transitions, width, label='Self-Loop Transitions', color='steelblue', alpha=0.7)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Number of Transitions', fontsize=12)
    ax2.set_title('Transition Counts by Layer', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    figure_paths = []
    base_name = os.path.join(output_dir, 'self_loop_ratio_summary')
    for fmt in save_formats:
        file_path = f'{base_name}.{fmt}'
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        figure_paths.append(file_path)
    
    plt.close()
    return figure_paths


def save_statistics(
    layer_statistics: Dict[int, Dict[str, Any]],
    n: int,
    output_dir: str
) -> str:
    """
    保存统计信息到JSON文件
    
    Args:
        layer_statistics: 层统计信息
        n: n-gram的n值
        output_dir: 输出目录
    
    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为可序列化格式
    stats = {
        'n': n,
        'summary': {
            str(layer_idx): {
                'total_ngram_paths': int(info['total_ngram_paths']),
                'unique_ngram_paths': int(info['unique_ngram_paths']),
                'total_transitions': int(info['total_transitions']),
                'unique_transitions': int(info['unique_transitions']),
                'self_loop_transitions': int(info['self_loop_transitions']),
                'self_loop_ratio': float(info['self_loop_ratio']),
            }
            for layer_idx, info in layer_statistics.items()
        },
        'transition_matrices': {
            str(layer_idx): info['transition_matrix'].tolist()
            for layer_idx, info in layer_statistics.items()
        },
        'transition_counts': {
            str(layer_idx): {f"{source}-{target}": int(count) for (source, target), count in info['transition_counts'].items()}
            for layer_idx, info in layer_statistics.items()
        }
    }
    
    output_file = os.path.join(output_dir, f'n_gram_statistics_n{n}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n保存统计信息: {output_file}")
    return output_file


def save_statistics_csv(
    layer_statistics: Dict[int, Dict[str, Any]],
    n: int,
    output_dir: str
) -> str:
    """
    保存统计信息到CSV文件
    
    Args:
        layer_statistics: 层统计信息
        n: n-gram的n值
        output_dir: 输出目录
    
    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    import csv
    
    output_file = os.path.join(output_dir, f'n_gram_statistics_n{n}.csv')
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer', 'Total N-gram Paths', 'Unique N-gram Paths', 
                        'Total Transitions', 'Unique Transitions', 
                        'Self-Loop Transitions', 'Self-Loop Ratio'])
        
        for layer_idx in sorted(layer_statistics.keys()):
            info = layer_statistics[layer_idx]
            writer.writerow([
                layer_idx,
                info['total_ngram_paths'],
                info['unique_ngram_paths'],
                info['total_transitions'],
                info['unique_transitions'],
                info['self_loop_transitions'],
                f"{info['self_loop_ratio']:.6f}"
            ])
    
    print(f"保存CSV统计信息: {output_file}")
    return output_file


def save_transition_matrices_csv(
    layer_statistics: Dict[int, Dict[str, Any]],
    n: int,
    output_dir: str
):
    """
    为每一层保存转移矩阵为CSV文件
    
    Args:
        layer_statistics: 层统计信息
        n: n-gram的n值
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_idx, info in layer_statistics.items():
        matrix = info['transition_matrix']
        output_file = os.path.join(output_dir, f'layer{layer_idx}_transition_matrix_n{n}.csv')
        np.savetxt(output_file, matrix, delimiter=',', fmt='%d')
        print(f"保存转移矩阵: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='n-gram路径统计脚本')
    parser.add_argument('--input_file', type=str,
                       default='results_all.jsonl',
                       help='输入的JSONL文件路径')
    parser.add_argument('--n', type=int, default=2,
                       help='n-gram的n值（默认: 2）')
    parser.add_argument('--output_dir', type=str,
                       default='results_deepseek/expert_statistics/n_gram_stats',
                       help='输出目录')
    parser.add_argument('--save_formats', type=str, default='png,pdf',
                       help='图像保存格式，用逗号分隔（默认: png,pdf）')
    parser.add_argument('--save_transition_matrices', action='store_true',
                       help='是否保存转移矩阵图像（默认: False）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 步骤1: 加载数据
    print("=" * 80)
    print("步骤1: 加载数据")
    print("=" * 80)
    records = load_jsonl_data(args.input_file)
    
    # 步骤2: 提取专家序列
    print("\n" + "=" * 80)
    print("步骤2: 提取专家序列")
    print("=" * 80)
    request_sequences = extract_expert_sequences_by_request(records)
    print(f"提取了 {len(request_sequences)} 个请求的专家序列")
    
    # 步骤3: 计算n-gram统计
    print("\n" + "=" * 80)
    print(f"步骤3: 计算 {args.n}-gram 统计")
    print("=" * 80)
    layer_statistics = calculate_ngram_statistics(request_sequences, args.n)
    
    # 步骤4: 保存统计信息
    print("\n" + "=" * 80)
    print("步骤4: 保存统计信息")
    print("=" * 80)
    save_statistics(layer_statistics, args.n, args.output_dir)
    save_statistics_csv(layer_statistics, args.n, args.output_dir)
    save_transition_matrices_csv(layer_statistics, args.n, args.output_dir)
    
    # 步骤5: 生成可视化
    print("\n" + "=" * 80)
    print("步骤5: 生成可视化")
    print("=" * 80)
    save_formats = [fmt.strip() for fmt in args.save_formats.split(',')]
    
    # 生成重复路由比例汇总图
    visualize_self_loop_ratio(layer_statistics, args.output_dir, save_formats)
    
    # 生成每一层的转移矩阵（如果启用）
    if args.save_transition_matrices:
        print("\n生成转移矩阵图像...")
        for layer_idx, info in tqdm(layer_statistics.items(), desc="生成转移矩阵"):
            visualize_transition_matrix(
                info['transition_matrix'],
                layer_idx,
                args.output_dir,
                save_formats,
                title_suffix=f" (n={args.n})"
            )
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print(f"所有结果保存在: {args.output_dir}")
    print("=" * 80)
    
    # 打印汇总信息
    print("\n汇总信息:")
    print(f"{'Layer':<10} {'N-gram Paths':<20} {'Unique N-gram':<20} {'Total Transitions':<20} {'Unique Transitions':<20} {'Self-Loop':<15} {'Ratio':<10}")
    print("-" * 120)
    for layer_idx in sorted(layer_statistics.keys()):
        info = layer_statistics[layer_idx]
        print(f"{layer_idx:<10} {info['total_ngram_paths']:<20} {info['unique_ngram_paths']:<20} "
              f"{info['total_transitions']:<20} {info['unique_transitions']:<20} "
              f"{info['self_loop_transitions']:<15} {info['self_loop_ratio']:.6f}")


if __name__ == "__main__":
    main()

