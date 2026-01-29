#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并的专家组指标计算脚本

功能：
1. 从 results_all.jsonl 读取数据（只读取一次）
2. 计算专家组路由统计（expert_group_routing）
3. 计算专家组 n-gram 统计（expert_group_ngram），支持多个 n 值
4. 输出所有指标结果

用法：
    # 使用累计权重占比阈值模式
    python compute_expert_metrics.py \
        --input_file results_all.jsonl \
        --mode threshold \
        --threshold 0.85 \
        --n_values 2 5 10 20 \
        --output_dir output/

    # 使用 TopK 模式
    python compute_expert_metrics.py \
        --input_file results_all.jsonl \
        --mode topk \
        --topk 10 \
        --n_values 2 5 10 20 \
        --output_dir output/
"""

from __future__ import annotations

import argparse
import json
import os
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set, Tuple


def load_jsonl_data(input_file: str) -> List[Dict[str, Any]]:
    """加载 JSONL 数据（只读取一次）"""
    print(f"正在加载数据: {input_file}")
    records: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"警告: 第 {line_num} 行JSON解析失败: {exc}")
                continue
    print(f"加载完成: {len(records)} 条记录")
    return records


def compute_layer_expert_weights(
    records: List[Dict[str, Any]]
) -> Tuple[Dict[int, Counter], Dict[int, float]]:
    """计算每层专家的权重统计"""
    layer_expert_weights: Dict[int, Counter] = defaultdict(Counter)
    layer_total_weight: Dict[int, float] = defaultdict(float)

    for record in records:
        layers = record.get("layers", {})
        for layer_key, layer_data in layers.items():
            try:
                layer_idx = int(layer_key)
            except (ValueError, TypeError):
                continue
            expert_ids = layer_data.get("expert_ids", [])
            weights = layer_data.get("weights", [])

            if not expert_ids:
                continue

            if weights and len(weights) > 0:
                min_len = min(len(expert_ids), len(weights))
                for i in range(min_len):
                    expert_id = expert_ids[i]
                    weight = weights[i]
                    if not isinstance(weight, (int, float)):
                        continue
                    layer_expert_weights[layer_idx][expert_id] += float(weight)
                    layer_total_weight[layer_idx] += float(weight)
            else:
                # 若没有 weights，则使用计数作为权重
                for expert_id in expert_ids:
                    layer_expert_weights[layer_idx][expert_id] += 1.0
                    layer_total_weight[layer_idx] += 1.0

    return dict(layer_expert_weights), dict(layer_total_weight)


def select_top_experts_by_threshold(
    expert_weights: Counter,
    total_weight: float,
    threshold: float
) -> Tuple[List[int], Dict[int, float], float]:
    """按累计权重占比阈值选择专家组"""
    if total_weight <= 0:
        return [], {}, 0.0

    sorted_items = expert_weights.most_common()
    expert_shares: Dict[int, float] = {
        expert_id: weight / total_weight for expert_id, weight in sorted_items
    }

    selected: List[int] = []
    cumulative = 0.0
    for expert_id, _ in sorted_items:
        share = expert_shares[expert_id]
        selected.append(expert_id)
        cumulative += share
        if cumulative >= threshold:
            break

    return selected, expert_shares, cumulative


def select_top_experts_by_topk(
    expert_weights: Counter,
    total_weight: float,
    topk: int
) -> Tuple[List[int], Dict[int, float], float]:
    """按 TopK 选择专家组"""
    if total_weight <= 0:
        return [], {}, 0.0

    sorted_items = expert_weights.most_common()
    expert_shares: Dict[int, float] = {
        expert_id: weight / total_weight for expert_id, weight in sorted_items
    }

    selected: List[int] = []
    cumulative = 0.0
    for i, (expert_id, _) in enumerate(sorted_items):
        if i >= topk:
            break
        share = expert_shares[expert_id]
        selected.append(expert_id)
        cumulative += share

    return selected, expert_shares, cumulative


def compute_routing_ratio_single_token(
    records: List[Dict[str, Any]],
    expert_groups: Dict[int, List[int]]
) -> Dict[int, Dict[str, float]]:
    """计算单token路由占比"""
    layer_stats: Dict[int, Dict[str, float]] = defaultdict(lambda: {
        "total_tokens": 0,
        "group_tokens": 0
    })

    for record in records:
        layers = record.get("layers", {})
        for layer_key, layer_data in layers.items():
            try:
                layer_idx = int(layer_key)
            except (ValueError, TypeError):
                continue
            expert_ids = layer_data.get("expert_ids", [])
            if not expert_ids:
                continue
            primary_expert = expert_ids[0]
            group = set(expert_groups.get(layer_idx, []))
            layer_stats[layer_idx]["total_tokens"] += 1
            if primary_expert in group:
                layer_stats[layer_idx]["group_tokens"] += 1

    return dict(layer_stats)


def extract_expert_sequences_by_request(
    records: List[Dict[str, Any]]
) -> Dict[int, Dict[int, List[int]]]:
    """
    按 request_id 或 prompt 分组，提取每一层的专家序列（使用每个 token 的第一个 expert_id）
    """
    request_sequences: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    
    print("\n正在提取专家序列（按文件原始顺序）...")
    total_records = len(records)
    records_with_request_id = 0
    records_with_prompt = 0
    records_with_layers = 0
    tokens_processed = 0
    
    # 如果没有 request_id，使用 prompt 作为标识符，需要创建 prompt 到 ID 的映射
    prompt_to_id: Dict[str, int] = {}
    next_prompt_id = 0
    use_prompt_as_id = False
    
    for record in records:
        # 首先尝试使用 request_id
        request_id = record.get("request_id")
        
        # 如果没有 request_id，使用 prompt 作为标识符
        if request_id is None:
            prompt = record.get("prompt")
            if prompt is None:
                continue
            records_with_prompt += 1
            
            # 为每个唯一的 prompt 分配一个 ID
            if prompt not in prompt_to_id:
                prompt_to_id[prompt] = next_prompt_id
                next_prompt_id += 1
            request_id = prompt_to_id[prompt]
            use_prompt_as_id = True
        else:
            records_with_request_id += 1
        
        # 两种格式都包含 layers 字段
        layers = record.get("layers", {})
        if not layers:
            continue
        records_with_layers += 1
        
        # 遍历每一层
        for layer_key, layer_data in layers.items():
            try:
                layer_idx = int(layer_key)
            except (ValueError, TypeError):
                continue
            
            # 两种格式的 layer_data 都包含 expert_ids 字段
            expert_ids = layer_data.get("expert_ids", [])
            if expert_ids and len(expert_ids) > 0:
                # 使用第一个 expert_id 作为该 token 在该层的主要专家
                primary_expert = expert_ids[0]
                request_sequences[request_id][layer_idx].append(primary_expert)
                tokens_processed += 1
    
    print(f"总记录数: {total_records}")
    print(f"包含 request_id 的记录数: {records_with_request_id}")
    print(f"包含 prompt 的记录数: {records_with_prompt}")
    print(f"包含 layers 的记录数: {records_with_layers}")
    print(f"处理的 token 数: {tokens_processed}")
    if use_prompt_as_id:
        print(f"使用 prompt 作为请求标识符，共 {len(prompt_to_id)} 个唯一的 prompt")
    print(f"提取了 {len(request_sequences)} 个请求的专家序列")
    
    return dict(request_sequences)


def calculate_group_ngram_statistics(
    request_sequences: Dict[int, Dict[int, List[int]]],
    expert_groups: Dict[int, Set[int]],
    n: int
) -> Dict[int, Dict[str, Any]]:
    """
    计算每一层的专家组 n-gram 统计信息
    
    只统计所有专家都落在专家组内的 n-gram 路径
    """
    print(f"\n正在计算 {n}-gram 统计信息（只统计全部落在专家组内的路径）...")
    
    # 从 request_sequences 中获取所有层
    all_layers_from_sequences = set()
    for request_data in request_sequences.values():
        all_layers_from_sequences.update(request_data.keys())
    
    # 从 expert_groups 中获取所有层（作为备用）
    all_layers_from_groups = set(expert_groups.keys())
    
    # 使用两者的并集，确保所有层都被处理
    all_layers = all_layers_from_sequences | all_layers_from_groups
    
    if not all_layers:
        print("警告: 没有检测到任何层，请检查输入数据")
        return {}
    
    num_layers = max(all_layers) + 1 if all_layers else 0
    print(f"从序列中检测到 {len(all_layers_from_sequences)} 层")
    print(f"从专家组定义中检测到 {len(all_layers_from_groups)} 层")
    print(f"总共需要处理 {num_layers} 层")
    
    layer_statistics = {}
    
    for layer_idx in range(num_layers):
        group_set = expert_groups.get(layer_idx, set())
        if not group_set:
            print(f"警告: 层 {layer_idx} 没有专家组定义，跳过")
            layer_statistics[layer_idx] = {
                'total_ngram_paths': 0,
                'group_ngram_paths': 0,
                'group_ngram_ratio': 0.0,
                'unique_group_ngram_paths': 0,
            }
            continue
        
        total_ngram_paths = 0
        group_ngram_paths = 0
        unique_group_ngram_paths_set = set()
        
        for request_id, layer_data in request_sequences.items():
            expert_seq = layer_data.get(layer_idx, [])
            
            if len(expert_seq) < n:
                continue
            
            # 滑动窗口提取 n-gram 路径
            for start in range(len(expert_seq) - n + 1):
                ngram = tuple(expert_seq[start:start+n])
                total_ngram_paths += 1
                
                # 检查 n-gram 中的所有专家是否都在专家组内
                if all(expert_id in group_set for expert_id in ngram):
                    group_ngram_paths += 1
                    unique_group_ngram_paths_set.add(ngram)
        
        group_ngram_ratio = (group_ngram_paths / total_ngram_paths) if total_ngram_paths > 0 else 0.0
        
        layer_statistics[layer_idx] = {
            'total_ngram_paths': total_ngram_paths,
            'group_ngram_paths': group_ngram_paths,
            'group_ngram_ratio': group_ngram_ratio,
            'unique_group_ngram_paths': len(unique_group_ngram_paths_set),
        }
        
        print(f"层 {layer_idx}: 总路径={total_ngram_paths}, "
              f"专家组路径={group_ngram_paths} (唯一={len(unique_group_ngram_paths_set)}), "
              f"占比={group_ngram_ratio:.4f}")
    
    return layer_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="合并的专家组指标计算")
    parser.add_argument("--input_file", required=True, help="results_all.jsonl 路径")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["threshold", "topk"],
        default="threshold",
        help="选择模式：threshold（累计权重占比）或 topk（Top K 专家）"
    )
    parser.add_argument("--threshold", type=float, default=0.85, help="累计权重占比阈值（mode=threshold 时使用）")
    parser.add_argument("--topk", type=int, default=None, help="Top K 专家数量（mode=topk 时使用）")
    parser.add_argument("--n_values", type=int, nargs="+", default=[2, 5, 10, 20], 
                       help="n-gram 的 n 值列表（默认: 2 5 10 20）")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    args = parser.parse_args()

    # 验证参数
    if args.mode == "topk" and args.topk is None:
        raise ValueError("使用 topk 模式时，必须指定 --topk 参数")

    print("=" * 80)
    print("步骤1: 加载数据（只读取一次）")
    print("=" * 80)
    records = load_jsonl_data(args.input_file)
    if not records:
        raise ValueError("没有读取到任何记录，请检查 input_file。")

    print("\n" + "=" * 80)
    print("步骤2: 计算专家组路由统计")
    print("=" * 80)
    layer_expert_weights, layer_total_weight = compute_layer_expert_weights(records)
    expert_groups: Dict[int, List[int]] = {}
    expert_shares_by_layer: Dict[int, Dict[int, float]] = {}
    group_share_by_layer: Dict[int, float] = {}

    for layer_idx, weights in layer_expert_weights.items():
        total_weight = layer_total_weight.get(layer_idx, 0.0)
        if args.mode == "threshold":
            selected, shares, cumulative = select_top_experts_by_threshold(
                weights, total_weight, args.threshold
            )
        else:  # mode == "topk"
            selected, shares, cumulative = select_top_experts_by_topk(
                weights, total_weight, args.topk
            )
        expert_groups[layer_idx] = selected
        expert_shares_by_layer[layer_idx] = shares
        group_share_by_layer[layer_idx] = cumulative

    routing_stats = compute_routing_ratio_single_token(records, expert_groups)

    # 保存专家组路由统计结果
    routing_output_payload = {
        "input_file": args.input_file,
        "mode": args.mode,
        "layers": {}
    }
    if args.mode == "threshold":
        routing_output_payload["threshold"] = args.threshold
    else:
        routing_output_payload["topk"] = args.topk

    for layer_idx in sorted(expert_groups.keys()):
        stats = routing_stats.get(layer_idx, {"total_tokens": 0, "group_tokens": 0})
        total_tokens = int(stats["total_tokens"])
        group_tokens = int(stats["group_tokens"])
        routing_ratio = (group_tokens / total_tokens) if total_tokens > 0 else 0.0

        routing_output_payload["layers"][str(layer_idx)] = {
            "group_experts": expert_groups[layer_idx],
            "group_weight_share": group_share_by_layer.get(layer_idx, 0.0),
            "expert_weight_shares": {
                str(expert_id): share for expert_id, share in expert_shares_by_layer[layer_idx].items()
            },
            "routing_ratio": routing_ratio,
            "total_tokens": total_tokens,
            "group_tokens": group_tokens
        }

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存专家组路由统计JSON
    routing_json_file = os.path.join(args.output_dir, "expert_group_routing.json")
    with open(routing_json_file, "w", encoding="utf-8") as f:
        json.dump(routing_output_payload, f, indent=2, ensure_ascii=False)
    print(f"\n保存专家组路由统计: {routing_json_file}")

    # 保存专家组路由统计CSV
    routing_csv_file = os.path.join(args.output_dir, "expert_group_routing.csv")
    with open(routing_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "group_experts",
            "group_weight_share",
            "routing_ratio",
            "total_tokens",
            "group_tokens"
        ])
        for layer_idx in sorted(expert_groups.keys()):
            layer_key = str(layer_idx)
            layer_data = routing_output_payload["layers"][layer_key]
            writer.writerow([
                layer_key,
                ",".join(str(e) for e in layer_data["group_experts"]),
                f"{layer_data['group_weight_share']:.6f}",
                f"{layer_data['routing_ratio']:.6f}",
                layer_data["total_tokens"],
                layer_data["group_tokens"]
            ])
    print(f"保存专家组路由统计CSV: {routing_csv_file}")

    print("\n" + "=" * 80)
    print("步骤3: 提取专家序列（用于n-gram计算）")
    print("=" * 80)
    request_sequences = extract_expert_sequences_by_request(records)

    print("\n" + "=" * 80)
    print("步骤4: 计算专家组 n-gram 统计")
    print("=" * 80)
    # 将 expert_groups 转换为 Set 格式供 n-gram 计算使用
    expert_groups_set: Dict[int, Set[int]] = {
        layer_idx: set(group) for layer_idx, group in expert_groups.items()
    }

    # 为每个 n 值计算 n-gram 统计
    for n in args.n_values:
        print(f"\n处理 n={n}...")
        layer_statistics = calculate_group_ngram_statistics(
            request_sequences, expert_groups_set, n
        )

        ngram_output_payload = {
            "input_file": args.input_file,
            "group_file": routing_json_file,
            "n": n,
            "layers": {}
        }

        for layer_idx in sorted(layer_statistics.keys()):
            stats = layer_statistics[layer_idx]
            ngram_output_payload["layers"][str(layer_idx)] = {
                "total_ngram_paths": int(stats["total_ngram_paths"]),
                "group_ngram_paths": int(stats["group_ngram_paths"]),
                "group_ngram_ratio": float(stats["group_ngram_ratio"]),
                "unique_group_ngram_paths": int(stats["unique_group_ngram_paths"]),
            }

        # 保存 n-gram 统计JSON
        ngram_json_file = os.path.join(args.output_dir, f"expert_group_ngram_n{n}.json")
        with open(ngram_json_file, "w", encoding="utf-8") as f:
            json.dump(ngram_output_payload, f, indent=2, ensure_ascii=False)
        print(f"保存 n-gram 统计 (n={n}): {ngram_json_file}")

        # 保存 n-gram 统计CSV
        ngram_csv_file = os.path.join(args.output_dir, f"expert_group_ngram_n{n}.csv")
        with open(ngram_csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "layer",
                "total_ngram_paths",
                "group_ngram_paths",
                "group_ngram_ratio",
                "unique_group_ngram_paths"
            ])
            for layer_idx in sorted(layer_statistics.keys()):
                layer_key = str(layer_idx)
                layer_data = ngram_output_payload["layers"][layer_key]
                writer.writerow([
                    layer_key,
                    layer_data["total_ngram_paths"],
                    layer_data["group_ngram_paths"],
                    f"{layer_data['group_ngram_ratio']:.6f}",
                    layer_data["unique_group_ngram_paths"]
                ])
        print(f"保存 n-gram 统计CSV (n={n}): {ngram_csv_file}")

    print("\n" + "=" * 80)
    print("所有指标计算完成！")
    print("=" * 80)
    print(f"输出目录: {args.output_dir}")
    print(f"生成的文件:")
    print(f"  - expert_group_routing.json")
    print(f"  - expert_group_routing.csv")
    for n in args.n_values:
        print(f"  - expert_group_ngram_n{n}.json")
        print(f"  - expert_group_ngram_n{n}.csv")


if __name__ == "__main__":
    main()

