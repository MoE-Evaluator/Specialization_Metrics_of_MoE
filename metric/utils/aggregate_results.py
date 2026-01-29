#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总 batch_call.sh 的计算结果

功能：
1. 读取每个 INPUT_DIR 下的 n_gram 统计结果（n=2,5,10,20）
2. 对每个 n 值，计算所有层的均值
3. 读取每个 INPUT_DIR 下的 Rademacher 复杂度结果（500,1000,2000）
4. 对每个值，计算所有专家的均值作为专家选择的复杂度
5. 生成 CSV 矩阵，每行是一个 INPUT_DIR，每列是一个指标

用法:
    python aggregate_results.py --input_dirs <dir1> <dir2> ... --output_file results_summary.csv
    或
    python aggregate_results.py --config_file batch_call.sh --output_file results_summary.csv
"""

import json
import os
import argparse
import csv
import re
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


def extract_dir_name(input_dir: str) -> str:
    """
    从完整路径中提取目录名称（用于 CSV 的行标签）
    
    Args:
        input_dir: 完整路径
    
    Returns:
        目录名称
    """
    # 移除末尾的斜杠
    dir_path = input_dir.rstrip('/')
    # 获取最后一部分
    dir_name = os.path.basename(dir_path)
    return dir_name


def load_ngram_statistics(input_dir: str, n: int) -> Optional[Dict[str, Any]]:
    """
    加载 n-gram 统计结果
    
    Args:
        input_dir: 输入目录
        n: n-gram 的 n 值
    
    Returns:
        统计结果字典，如果文件不存在则返回 None
    """
    json_file = os.path.join(input_dir, f'n_gram/n_gram_{n}', f'n_gram_statistics_n{n}.json')
    
    if not os.path.exists(json_file):
        print(f"警告: 文件不存在: {json_file}")
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"错误: 读取文件失败 {json_file}: {e}")
        return None


def calculate_ngram_mean_across_layers(stats: Dict[str, Any]) -> float:
    """
    计算所有层的 Self-Loop Ratio 均值
    
    Args:
        stats: n-gram 统计结果
    
    Returns:
        所有层的 Self-Loop Ratio 均值（和除以层数）
    """
    summary = stats.get('summary', {})
    if not summary:
        return 0.0
    
    self_loop_ratios = []
    
    for layer_id, layer_data in summary.items():
        self_loop_ratio = layer_data.get('self_loop_ratio', 0.0)
        self_loop_ratios.append(self_loop_ratio)
    
    if not self_loop_ratios:
        return 0.0
    
    # 计算均值（和除以行数）
    return float(np.mean(self_loop_ratios))


def load_rademacher_complexity(input_dir: str, value: int) -> Optional[Dict[str, Any]]:
    """
    加载 Rademacher 复杂度结果
    
    Args:
        input_dir: 输入目录
        value: num_samples/num_simulations 的值
    
    Returns:
        结果字典，如果文件不存在则返回 None
    """
    json_file = os.path.join(input_dir, f'rademacher/rademacher_complexity_{value}', 'rademacher_complexity.json')
    
    if not os.path.exists(json_file):
        print(f"警告: 文件不存在: {json_file}")
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"错误: 读取文件失败 {json_file}: {e}")
        return None


def calculate_rademacher_mean_across_experts(results: Dict[str, Any]) -> float:
    """
    计算所有专家的 Rademacher 复杂度均值
    
    Args:
        results: Rademacher 复杂度结果
    
    Returns:
        所有专家的均值
    """
    layer_results = results.get('layer_results', {})
    if not layer_results:
        return 0.0
    
    all_expert_complexities = []
    
    # 遍历所有层
    for layer_id, layer_data in layer_results.items():
        expert_rademacher = layer_data.get('expert_rademacher', {})
        if expert_rademacher:
            # 收集该层所有专家的复杂度值
            for expert_id, complexity in expert_rademacher.items():
                all_expert_complexities.append(complexity)
    
    if not all_expert_complexities:
        return 0.0
    
    return float(np.mean(all_expert_complexities))


def parse_input_dirs_from_batch_call(batch_call_file: str) -> List[str]:
    """
    从 batch_call.sh 文件中解析 INPUT_DIRS 数组
    
    Args:
        batch_call_file: batch_call.sh 文件路径
    
    Returns:
        输入目录列表
    """
    input_dirs = []
    
    if not os.path.exists(batch_call_file):
        print(f"警告: 文件不存在: {batch_call_file}")
        return input_dirs
    
    try:
        with open(batch_call_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式提取 INPUT_DIRS 数组内容
        # 匹配 INPUT_DIRS=( ... ) 模式
        pattern = r'INPUT_DIRS\s*=\s*\((.*?)\)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            array_content = match.group(1)
            # 按行分割
            for line in array_content.split('\n'):
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                
                # 移除引号和逗号
                path = line.strip().strip('"').strip("'").rstrip(',')
                if path:
                    input_dirs.append(path)
        else:
            print("警告: 未找到 INPUT_DIRS 数组")
    
    except Exception as e:
        print(f"错误: 解析 batch_call.sh 文件失败: {e}")
    
    return input_dirs


def aggregate_results(input_dirs: List[str], output_file: str):
    """
    汇总所有结果并生成 CSV 文件
    
    Args:
        input_dirs: 输入目录列表
        output_file: 输出 CSV 文件路径
    """
    print("=" * 80)
    print("开始汇总结果")
    print("=" * 80)
    print(f"输入目录数量: {len(input_dirs)}")
    
    # n-gram 的 n 值列表
    n_values = [2, 5, 10, 20]
    
    # Rademacher 复杂度的值列表
    rademacher_values = [500, 1000, 2000]
    
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
        
        # 处理 n-gram 统计结果
        print("  读取 n-gram 统计结果...")
        for n in n_values:
            stats = load_ngram_statistics(input_dir, n)
            if stats is None:
                result_row[f'ngram_{n}_ratio_mean'] = None
                print(f"    警告: n={n} 的结果不存在")
                continue
            
            # 计算所有层的 Self-Loop Ratio 均值（和除以层数）
            mean_ratio = calculate_ngram_mean_across_layers(stats)
            result_row[f'ngram_{n}_ratio_mean'] = mean_ratio
            print(f"    n={n}: Self-Loop Ratio 均值 = {mean_ratio:.6f}")
        
        # 处理 Rademacher 复杂度结果
        print("  读取 Rademacher 复杂度结果...")
        for value in rademacher_values:
            results = load_rademacher_complexity(input_dir, value)
            if results is None:
                result_row[f'rademacher_{value}_mean'] = None
                print(f"    警告: value={value} 的结果不存在")
                continue
            
            # 计算所有专家的均值
            mean_complexity = calculate_rademacher_mean_across_experts(results)
            result_row[f'rademacher_{value}_mean'] = mean_complexity
            print(f"    value={value}: 均值 = {mean_complexity:.6f}")
        
        all_results.append(result_row)
    
    # 生成 CSV 文件
    print(f"\n生成 CSV 文件: {output_file}")
    
    # 定义列名
    columns = ['directory', 'full_path']
    columns.extend([f'ngram_{n}_ratio_mean' for n in n_values])
    columns.extend([f'rademacher_{value}_mean' for value in rademacher_values])
    
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
    print(f"{'目录':<50} {'N-gram Self-Loop Ratio (2,5,10,20)':<40} {'Rademacher (500,1000,2000)':<30}")
    print("-" * 120)
    for result in all_results:
        ngram_str = ", ".join([
            f"{result.get(f'ngram_{n}_ratio_mean', 0):.6f}" if result.get(f'ngram_{n}_ratio_mean') is not None else "N/A"
            for n in n_values
        ])
        rademacher_str = ", ".join([
            f"{result.get(f'rademacher_{value}_mean', 0):.6f}" if result.get(f'rademacher_{value}_mean') is not None else "N/A"
            for value in rademacher_values
        ])
        print(f"{result['directory']:<50} {ngram_str:<40} {rademacher_str:<30}")


def main():
    parser = argparse.ArgumentParser(description='汇总 batch_call.sh 的计算结果')
    parser.add_argument('--input_dirs', type=str, nargs='+',
                       help='输入目录列表')
    parser.add_argument('--config_file', type=str,
                       help='batch_call.sh 配置文件路径（用于自动解析 INPUT_DIRS）')
    parser.add_argument('--output_file', type=str, default='results_summary.csv',
                       help='输出 CSV 文件路径（默认: results_summary.csv）')
    
    args = parser.parse_args()
    
    input_dirs = []
    
    # 从配置文件解析或从参数获取
    if args.config_file:
        input_dirs = parse_input_dirs_from_batch_call(args.config_file)
        print(f"从配置文件解析到 {len(input_dirs)} 个目录")
    elif args.input_dirs:
        input_dirs = args.input_dirs
    else:
        print("错误: 请提供 --input_dirs 或 --config_file 参数")
        parser.print_help()
        exit(1)
    
    if not input_dirs:
        print("错误: 没有找到输入目录")
        exit(1)
    
    # 汇总结果
    aggregate_results(input_dirs, args.output_file)


if __name__ == "__main__":
    main()

