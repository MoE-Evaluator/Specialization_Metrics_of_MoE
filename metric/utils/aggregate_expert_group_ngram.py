#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总 expert_group_ngram 结果

功能：
1. 读取每个 INPUT_DIR 下的 group_experts/expert_group_ngram_n{2,5,10,20}.json 文件
2. 对每个 n 值，计算所有层的 group_ngram_ratio 均值
3. 生成 CSV 汇总文件

用法:
    python aggregate_expert_group_ngram.py --input_dirs <dir1> <dir2> ... --output_file summary.csv
"""

import json
import os
import argparse
import csv
import numpy as np
from typing import List, Dict, Any, Optional


def extract_dir_name(input_dir: str) -> str:
    """
    从完整路径中提取目录名称（用于 CSV 的行标签）
    
    Args:
        input_dir: 完整路径
    
    Returns:
        目录名称
    """
    dir_path = input_dir.rstrip('/')
    dir_name = os.path.basename(dir_path)
    return dir_name


def load_expert_group_ngram(input_dir: str, n: int) -> Optional[Dict[str, Any]]:
    """
    加载 expert_group_ngram 结果
    
    支持两种目录结构：
    1. 新结构: expert_group_routing/expert_group_ngram_n{n}.json
    2. 旧结构: group_experts/expert_group_ngram_n{n}.json
    
    Args:
        input_dir: 输入目录
        n: n-gram 的 n 值
    
    Returns:
        结果字典，如果文件不存在则返回 None
    """
    # 尝试新目录结构
    json_file_new = os.path.join(input_dir, 'expert_group_routing', f'expert_group_ngram_n{n}.json')
    
    # 尝试旧目录结构（向后兼容）
    json_file_old = os.path.join(input_dir, 'group_experts', f'expert_group_ngram_n{n}.json')
    
    json_file = None
    if os.path.exists(json_file_new):
        json_file = json_file_new
    elif os.path.exists(json_file_old):
        json_file = json_file_old
    else:
        print(f"警告: 文件不存在: {json_file_new} 或 {json_file_old}")
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"错误: 读取文件失败 {json_file}: {e}")
        return None


def calculate_group_ngram_mean_across_layers(stats: Dict[str, Any]) -> float:
    """
    计算所有层的 group_ngram_ratio 均值
    
    Args:
        stats: expert_group_ngram 统计结果
    
    Returns:
        所有层的 group_ngram_ratio 均值
    """
    layers = stats.get('layers', {})
    if not layers:
        return 0.0
    
    ratios = []
    for layer_key, layer_data in layers.items():
        ratio = layer_data.get('group_ngram_ratio', 0.0)
        ratios.append(ratio)
    
    if not ratios:
        return 0.0
    
    return float(np.mean(ratios))


def aggregate_results(input_dirs: List[str], output_file: str):
    """
    汇总所有结果并生成 CSV 文件
    
    Args:
        input_dirs: 输入目录列表
        output_file: 输出 CSV 文件路径
    """
    print("=" * 80)
    print("开始汇总 expert_group_ngram 结果")
    print("=" * 80)
    print(f"输入目录数量: {len(input_dirs)}")
    
    # n-gram 的 n 值列表
    n_values = [2, 5, 10, 20]
    
    # 存储所有结果
    all_results = []
    
    # 遍历每个输入目录
    for input_dir in input_dirs:
        print(f"\n处理目录: {input_dir}")
        
        if not os.path.isdir(input_dir):
            print(f"警告: 目录不存在，跳过: {input_dir}")
            continue
        
        dir_name = extract_dir_name(input_dir)
        result_row = {'directory': dir_name, 'full_path': input_dir}
        
        # 处理 expert_group_ngram 结果
        print("  读取 expert_group_ngram 统计结果...")
        for n in n_values:
            stats = load_expert_group_ngram(input_dir, n)
            if stats is None:
                result_row[f'group_ngram_n{n}_ratio_mean'] = None
                print(f"    警告: n={n} 的结果不存在")
                continue
            
            # 计算所有层的 group_ngram_ratio 均值
            mean_ratio = calculate_group_ngram_mean_across_layers(stats)
            result_row[f'group_ngram_n{n}_ratio_mean'] = mean_ratio
            print(f"    n={n}: group_ngram_ratio 均值 = {mean_ratio:.6f}")
        
        all_results.append(result_row)
    
    # 生成 CSV 文件
    print(f"\n生成 CSV 文件: {output_file}")
    
    # 定义列名
    columns = ['directory', 'full_path']
    columns.extend([f'group_ngram_n{n}_ratio_mean' for n in n_values])
    
    # 写入 CSV
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for result in all_results:
            writer.writerow(result)
    
    print(f"结果已保存到: {output_file}")
    print(f"共处理 {len(all_results)} 个目录")
    print("=" * 80)
    
    # 打印摘要
    print("\n结果摘要:")
    print(f"{'目录':<50} {'Group N-gram Ratio (n=2,5,10,20)':<40}")
    print("-" * 90)
    for result in all_results:
        ngram_str = ", ".join([
            f"{result.get(f'group_ngram_n{n}_ratio_mean', 0):.6f}" if result.get(f'group_ngram_n{n}_ratio_mean') is not None else "N/A"
            for n in n_values
        ])
        print(f"{result['directory']:<50} {ngram_str:<40}")


def main():
    parser = argparse.ArgumentParser(description='汇总 expert_group_ngram 结果')
    parser.add_argument('--input_dirs', type=str, nargs='+', required=True,
                       help='输入目录列表')
    parser.add_argument('--output_file', type=str, default='group_n_gram_summary.csv',
                       help='输出 CSV 文件路径（默认: group_n_gram_summary.csv）')
    
    args = parser.parse_args()
    
    if not args.input_dirs:
        print("错误: 没有找到输入目录")
        exit(1)
    
    # 汇总结果
    aggregate_results(args.input_dirs, args.output_file)


if __name__ == "__main__":
    main()

