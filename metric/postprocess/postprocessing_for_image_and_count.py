#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能：
1. 加载inference结果和专家分布数据
2. 对齐输出tokens与expert records，添加output_token_id和output_token
3. 保存对齐后的数据为JSONL格式
4. 分析每层的专家选择情况（计数和权重）
5. 生成热力图可视化（PNG和PDF格式）
6. 保存统计信息（JSON和CSV格式）

默认结果保存在: input_file_path/expert_statistics/ 目录

用法:
    python analyze_expert_distribution_complete.py \
        --inference_file merged_test_val.jsonl \
        --expert_data_file results_deepseek \
        --tokenizer_path deepseek_moe \
        --num_requests None
"""

import json
import os
import argparse
import glob
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_inference_results(inference_file: str, tokenizer=None) -> List[Dict]:
    """
    加载inference结果，提取输出token_ids和logprobs
    
    Args:
        inference_file: inference_results.jsonl文件路径
        tokenizer: tokenizer对象，如果为None则从deepseek_moe加载
    
    Returns:
        包含token_ids和logprobs的列表（每个元素是一个请求）
    """
    # 如果没有提供tokenizer，尝试加载
    if tokenizer is None:
        try:
            print("正在加载tokenizer: deepseek_moe")
            tokenizer = AutoTokenizer.from_pretrained("deepseek_moe", trust_remote_code=True)
            print("Tokenizer加载完成")
        except Exception as e:
            print(f"警告: 无法加载tokenizer: {e}")
            print("将尝试从数据中直接获取token_ids")
    
    results = []
    with open(inference_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行JSON解析失败: {e}")
                continue
            
            # 优先从full_response.choices[0]获取token_ids和logprobs
            token_ids = None
            logprobs_list = None
            full_response = data.get('full_response', {})
            if isinstance(full_response, dict):
                choices = full_response.get('choices', [])
                if choices and len(choices) > 0:
                    choice = choices[0]
                    token_ids = choice.get('token_ids', [])
                    logprobs_list = choice.get('logprobs', None)
            
            # 如果没有token_ids，尝试从generated_response文本编码
            if not token_ids:
                generated_text = data.get('generated_response', '')
                if not generated_text:
                    # 尝试从full_response.choices[0].message.content获取
                    if isinstance(full_response, dict):
                        choices = full_response.get('choices', [])
                        if choices and len(choices) > 0:
                            message = choices[0].get('message', {})
                            if isinstance(message, dict):
                                generated_text = message.get('content', '')
                
                if generated_text and tokenizer:
                    try:
                        # 使用tokenizer编码文本，不添加特殊token
                        token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
                        print(f"第 {line_num} 行: 从文本编码得到 {len(token_ids)} 个tokens")
                    except Exception as e:
                        print(f"警告: 第 {line_num} 行使用tokenizer编码失败: {e}")
                        token_ids = None
                elif generated_text and not tokenizer:
                    print(f"警告: 第 {line_num} 行有文本但无tokenizer，无法编码")
            
            if not token_ids:
                print(f"警告: 第 {line_num} 行无法获取token_ids，跳过")
                continue
            
            # 提取每个token的logprob（如果logprobs存在）
            token_logprobs = []
            if logprobs_list is not None:
                # logprobs通常是每个token的logprob列表
                if isinstance(logprobs_list, list):
                    token_logprobs = logprobs_list
                elif isinstance(logprobs_list, dict):
                    # 如果logprobs是字典，尝试获取token_logprobs字段
                    token_logprobs = logprobs_list.get('token_logprobs', [])
            
            results.append({
                'request_id': line_num - 1,  # 从0开始
                'token_ids': token_ids,
                'token_logprobs': token_logprobs,  # 添加logprobs
                'original_data': data.get('original_data', {})
            })
    
    return results


def load_expert_distribution_data(data_file_or_dir: str) -> Dict:
    """
    加载专家分布数据，支持单个文件或文件夹
    
    Args:
        data_file_or_dir: expert_distribution_data JSON文件路径或包含JSON文件的文件夹路径
    
    Returns:
        包含records的字典
    """
    # 检查是文件还是文件夹
    if os.path.isdir(data_file_or_dir):
        return _load_expert_distribution_data_from_folder(data_file_or_dir)
    else:
        return _load_expert_distribution_data_from_file(data_file_or_dir)


def _load_expert_distribution_data_from_file(data_file: str) -> Dict:
    """
    从单个文件加载专家分布数据
    
    Args:
        data_file: expert_distribution_data JSON文件路径
    
    Returns:
        包含records的字典
    """
    print(f"正在加载专家分布数据: {data_file}")
    print("文件较大，请稍候...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = data.get('records', [])
    print(f"加载完成: {len(records)} 条记录")
    
    return {
        'records': records,
        'last_physical_to_logical_map': data.get('last_physical_to_logical_map', [])
    }


def _load_expert_distribution_data_from_folder(data_dir: str) -> Dict:
    """
    从文件夹中加载所有JSON文件并合并数据
    
    Args:
        data_dir: 包含expert_distribution_data JSON文件的文件夹路径
    
    Returns:
        包含合并后records的字典
    """
    print(f"正在从文件夹加载专家分布数据: {data_dir}")
    
    # 查找所有JSON文件
    json_pattern = os.path.join(data_dir, '*.json')
    json_files = sorted(glob.glob(json_pattern))
    
    if not json_files:
        raise ValueError(f"在文件夹 {data_dir} 中未找到任何JSON文件")
    
    print(f"找到 {len(json_files)} 个JSON文件:")
    for f in json_files:
        print(f"  - {os.path.basename(f)}")
    
    # 合并所有文件的数据
    all_records = []
    last_physical_to_logical_map = None
    
    print("\n开始加载和合并文件...")
    for file_path in tqdm(json_files, desc="加载文件"):
        try:
            print(f"\n加载文件: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = data.get('records', [])
            all_records.extend(records)
            
            # 使用最后一个文件的physical_to_logical_map（通常所有文件应该相同）
            current_map = data.get('last_physical_to_logical_map', [])
            if current_map:
                last_physical_to_logical_map = current_map
            
            print(f"  已加载 {len(records)} 条记录，当前总计: {len(all_records)} 条")
            
        except Exception as e:
            print(f"警告: 加载文件 {file_path} 失败: {e}")
            continue
    
    print(f"\n合并完成: 共 {len(all_records)} 条记录")
    
    return {
        'records': all_records,
        'last_physical_to_logical_map': last_physical_to_logical_map or []
    }


def extract_prompt_from_original_data(original_data: Dict) -> str:
    """
    从original_data中提取prompt（user消息的内容）
    
    Args:
        original_data: original_data字典
    
    Returns:
        prompt字符串
    """
    messages = original_data.get('messages', [])
    for msg in messages:
        if msg.get('role') == 'user':
            return msg.get('content', '')
    return ''


def align_tokens_with_experts_streaming(inference_results: List[Dict], records: List[Dict], tokenizer, 
                                        start_token_index: int, request_boundaries: List, request_prompts: List,
                                        all_token_ids: List) -> List[Dict]:
    """
    流式对齐tokens与expert records（处理单个文件的records）
    
    Args:
        inference_results: inference结果列表
        records: 当前文件的expert records
        tokenizer: tokenizer对象（用于解码token）
        start_token_index: 当前处理的起始token索引
        request_boundaries: 请求边界列表
        request_prompts: 请求prompt列表
        all_token_ids: 所有token_ids列表
    
    Returns:
        对齐后的数据列表，每个元素对应一个token
    """
    aligned_data = []
    num_records = len(records)
    
    for i in range(num_records):
        global_token_index = start_token_index + i
        
        # 检查是否超出token范围
        if global_token_index >= len(all_token_ids):
            break
        
        token_id = all_token_ids[global_token_index]
        record = records[i].copy()  # 复制record，避免修改原始数据
        
        # 解码token_id为token文本
        output_token = None
        if tokenizer:
            try:
                output_token = tokenizer.decode([token_id])
            except Exception as e:
                print(f"警告: 解码token_id {token_id} 失败: {e}")
        
        # 添加output_token_id和output_token到record
        record['output_token_id'] = token_id
        record['output_token'] = output_token
        
        # 确定当前token属于哪个请求
        request_idx = None
        prompt = ''
        for req_idx, (start_idx, end_idx) in enumerate(request_boundaries):
            if start_idx <= global_token_index < end_idx:
                request_idx = req_idx
                prompt = request_prompts[req_idx] if req_idx < len(request_prompts) else ''
                break
        
        # 添加prompt信息
        record['prompt'] = prompt
        record['request_id'] = request_idx
        record['token_index'] = global_token_index
        
        aligned_data.append(record)
    
    return aligned_data


def prepare_token_info(inference_results: List[Dict], num_requests: int = None):
    """
    准备token信息（token_ids列表、请求边界、请求prompts）
    
    Args:
        inference_results: inference结果列表
        num_requests: 要处理的请求数量，如果为None则处理所有
    
    Returns:
        all_token_ids: 所有token_ids列表
        request_boundaries: 请求边界列表
        request_prompts: 请求prompt列表
    """
    # 限制请求数量
    if num_requests is None:
        num_requests = len(inference_results)
    num_requests = min(num_requests, len(inference_results))
    
    # 收集所有token_ids（按请求分组）
    all_token_ids = []
    request_boundaries = []  # 记录每个请求的token范围 [start_idx, end_idx)
    request_prompts = []  # 记录每个请求的prompt
    current_idx = 0
    
    for req_idx in range(num_requests):
        result = inference_results[req_idx]
        token_ids = result.get('token_ids', [])
        original_data = result.get('original_data', {})
        
        if not token_ids:
            request_boundaries.append((current_idx, current_idx))
            request_prompts.append('')
            continue
        
        # 提取prompt
        prompt = extract_prompt_from_original_data(original_data)
        request_prompts.append(prompt)
        
        start_idx = current_idx
        all_token_ids.extend(token_ids)
        current_idx += len(token_ids)
        end_idx = current_idx
        
        request_boundaries.append((start_idx, end_idx))
    
    return all_token_ids, request_boundaries, request_prompts


def align_tokens_with_experts(inference_results: List[Dict], expert_data: Dict, tokenizer, num_requests: int = None) -> List[Dict]:
    """
    按照位置索引对齐tokens与expert records，为每条record添加output_token信息
    
    Args:
        inference_results: inference结果列表
        expert_data: 专家分布数据
        tokenizer: tokenizer对象（用于解码token）
        num_requests: 要处理的请求数量，如果为None则处理所有
    
    Returns:
        对齐后的数据列表，每个元素对应一个token
    """
    records = expert_data['records']
    
    # 限制请求数量
    if num_requests is None:
        num_requests = len(inference_results)
    num_requests = min(num_requests, len(inference_results))
    print(f"\n处理前 {num_requests} 个请求")
    
    # 收集所有token_ids（按请求分组）
    all_token_ids = []
    request_boundaries = []  # 记录每个请求的token范围 [start_idx, end_idx)
    request_prompts = []  # 记录每个请求的prompt
    current_idx = 0
    
    for req_idx in range(num_requests):
        result = inference_results[req_idx]
        token_ids = result.get('token_ids', [])
        original_data = result.get('original_data', {})
        
        if not token_ids:
            print(f"警告: 请求 {req_idx} 没有token_ids，跳过")
            request_boundaries.append((current_idx, current_idx))
            request_prompts.append('')
            continue
        
        # 提取prompt
        prompt = extract_prompt_from_original_data(original_data)
        request_prompts.append(prompt)
        
        start_idx = current_idx
        all_token_ids.extend(token_ids)
        current_idx += len(token_ids)
        end_idx = current_idx
        
        request_boundaries.append((start_idx, end_idx))
        print(f"请求 {req_idx}: {len(token_ids)} 个tokens (索引 {start_idx} 到 {end_idx-1})")
    
    total_token_ids = len(all_token_ids)
    total_records = len(records)
    
    print(f"\n统计信息:")
    print(f"  总token_ids数量: {total_token_ids}")
    print(f"  总expert记录数量: {total_records}")
    print(f"  差异: {abs(total_token_ids - total_records)}")
    
    # 按位置对齐：匹配数据（取较小的数量）
    match_count = min(total_token_ids, total_records)
    aligned_data = []
    
    print(f"\n开始对齐（对齐 {match_count} 条记录）...")
    
    for i in tqdm(range(match_count), desc="对齐记录"):
        token_id = all_token_ids[i]
        record = records[i].copy()  # 复制record，避免修改原始数据
        
        # 解码token_id为token文本
        output_token = None
        if tokenizer:
            try:
                output_token = tokenizer.decode([token_id])
            except Exception as e:
                print(f"警告: 解码token_id {token_id} 失败: {e}")
        
        # 添加output_token_id和output_token到record
        record['output_token_id'] = token_id
        record['output_token'] = output_token
        
        # 确定当前token属于哪个请求
        request_idx = None
        prompt = ''
        for req_idx, (start_idx, end_idx) in enumerate(request_boundaries):
            if start_idx <= i < end_idx:
                request_idx = req_idx
                prompt = request_prompts[req_idx] if req_idx < len(request_prompts) else ''
                break
        
        # 添加prompt信息
        record['prompt'] = prompt
        record['request_id'] = request_idx
        record['token_index'] = i
        
        aligned_data.append(record)
    
    return aligned_data


def convert_record_to_output_format(record: Dict) -> Dict:
    """
    将record转换为输出格式
    
    转换前格式：
    - topk_ids_of_layer: [[[expert_ids]]] 形状是 [num_layers][batch_size][topk]
    - topk_weights_of_layer: [[[weights]]] 形状是 [num_layers][batch_size][topk]
    - router_logits_of_layer: [[[logits]]] 形状是 [num_layers][batch_size][num_experts]（可选）
    
    转换后格式：
    {
        "layers": {
            "0": {
                "expert_ids": [123, 69, 74, ...],
                "weights": [0.207, 0.151, ...],
                "logits": [...]（可选）
            },
            "1": {...}
        },
        "output_token_id": 123,
        "output_token": "...",
        ...
    }
    
    Args:
        record: 原始record字典
    
    Returns:
        转换后的record字典
    """
    topk_ids_of_layer = record.get('topk_ids_of_layer', [])
    topk_weights_of_layer = record.get('topk_weights_of_layer', [])
    router_logits_of_layer = record.get('router_logits_of_layer', [])
    
    layers_dict = {}
    
    # 遍历每一层
    num_layers = len(topk_ids_of_layer)
    for layer_idx in range(num_layers):
        layer_topk_ids = topk_ids_of_layer[layer_idx] if layer_idx < len(topk_ids_of_layer) else []
        layer_topk_weights = topk_weights_of_layer[layer_idx] if layer_idx < len(topk_weights_of_layer) else []
        layer_router_logits = router_logits_of_layer[layer_idx] if layer_idx < len(router_logits_of_layer) else []
        
        # 提取batch[0]的数据（通常batch_size=1）
        expert_ids = []
        weights = []
        logits = []
        
        if layer_topk_ids and len(layer_topk_ids) > 0:
            expert_ids = layer_topk_ids[0] if isinstance(layer_topk_ids[0], list) else []
        
        if layer_topk_weights and len(layer_topk_weights) > 0:
            weights = layer_topk_weights[0] if isinstance(layer_topk_weights[0], list) else []
        
        if layer_router_logits and len(layer_router_logits) > 0:
            logits = layer_router_logits[0] if isinstance(layer_router_logits[0], list) else []
        
        # 过滤掉-1的expert_ids以及对应的weights
        # 确保expert_ids和weights长度一致
        filtered_expert_ids = []
        filtered_weights = []
        filtered_logits = []
        next_expert_logit = None
        
        min_len = min(len(expert_ids), len(weights))
        for i in range(min_len):
            expert_id = expert_ids[i]
            weight = weights[i] if i < len(weights) else None
            # 过滤掉-1和无效的expert_id
            if expert_id is not None and expert_id >= 0 and expert_id < 256:
                filtered_expert_ids.append(expert_id)
                if weight is not None:
                    filtered_weights.append(float(weight))
                else:
                    filtered_weights.append(0.0)
                
                # 从router_logits中提取对应 expert_id 的 logit
                # 如果 logits 是完整的专家 logits 列表，则根据 expert_id 索引提取
                if logits and len(logits) > expert_id:
                    filtered_logits.append(float(logits[expert_id]))
                elif logits and i < len(logits):
                    # 如果 logits 长度与 expert_ids 相同，直接对应
                    filtered_logits.append(float(logits[i]))

        if logits and filtered_expert_ids:
            selected_id_set = set(filtered_expert_ids)
            candidate_unselected_logits = [
                float(logits[idx])
                for idx in range(len(logits))
                if idx not in selected_id_set
            ]
            if candidate_unselected_logits:
                next_expert_logit = max(candidate_unselected_logits)
        
        # 构建该层的数据
        layer_data = {
            "expert_ids": filtered_expert_ids,
            "weights": filtered_weights
        }
        
        # 如果存在logits，添加到layer_data中
        if filtered_logits:
            layer_data["logits"] = filtered_logits
        if next_expert_logit is not None:
            layer_data["next_expert_logit"] = float(next_expert_logit)
        
        # 只有当该层有数据时才添加
        if filtered_expert_ids:
            layers_dict[str(layer_idx)] = layer_data
    
    # 构建输出格式
    output_record = {
        "layers": layers_dict
    }
    
    # 保留其他字段
    if 'output_token_id' in record:
        output_record['output_token_id'] = record['output_token_id']
    if 'output_token' in record:
        output_record['output_token'] = record['output_token']
    if 'prompt' in record:
        output_record['prompt'] = record['prompt']
    if 'request_id' in record:
        output_record['request_id'] = record['request_id']
    if 'token_index' in record:
        output_record['token_index'] = record['token_index']
    if 'input_ids' in record:
        output_record['input_ids'] = record['input_ids']
    
    return output_record


def save_results_jsonl(aligned_data: List[Dict], output_file: str, append_mode: bool = False):
    """
    保存对齐后的数据为JSONL格式，每个token一行
    
    Args:
        aligned_data: 对齐后的数据列表
        output_file: 输出文件路径
        append_mode: 是否追加模式（True=追加，False=覆盖）
    """
    mode = 'a' if append_mode else 'w'
    action = "追加" if append_mode else "保存"
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, mode, encoding='utf-8') as f:
        for record in tqdm(aligned_data, desc=f"{action}记录"):
            # 转换为输出格式
            output_record = convert_record_to_output_format(record)
            f.write(json.dumps(output_record, ensure_ascii=False) + '\n')


def get_eos_token_ids(tokenizer) -> set:
    """
    从tokenizer获取EOS和PAD token IDs
    
    Args:
        tokenizer: tokenizer对象
    
    Returns:
        EOS和PAD token ID的集合
    """
    if tokenizer is None:
        return set()
    
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    result = set()
    if eos_token_id is not None:
        result.add(eos_token_id)
    if pad_token_id is not None:
        result.add(pad_token_id)
    
    return result


def analyze_expert_selection(records, eos_token_ids: set = None, 
                            layer_expert_counts: Dict = None, 
                            layer_expert_weights: Dict = None):
    """
    分析每层的专家选择情况，支持累加到已有统计字典
    
    Args:
        records: 记录列表（可以是原始records或对齐后的aligned_data）
        eos_token_ids: 需要过滤的token ID集合（EOS/PAD等）
        layer_expert_counts: 已有的专家选择计数字典（如果提供则累加，否则新建）
        layer_expert_weights: 已有的专家权重字典（如果提供则累加，否则新建）
    
    Returns:
        layer_expert_counts: 字典，格式为 {layer_id: {expert_id: count}}
        layer_expert_weights: 字典，格式为 {layer_id: {expert_id: sum_of_weights}}
        stats: 统计信息字典，包含processed和skipped数量
    """
    if eos_token_ids is None:
        eos_token_ids = set()
    
    # 如果提供了已有字典则使用，否则创建新字典
    if layer_expert_counts is None:
        layer_expert_counts = defaultdict(lambda: defaultdict(int))
    if layer_expert_weights is None:
        layer_expert_weights = defaultdict(lambda: defaultdict(float))
    
    total_processed = 0
    total_skipped = 0
    
    for record in records:
        # 支持对齐后的数据（有output_token_id）和原始数据（有input_ids）
        input_ids = record.get('input_ids', [])
        output_token_id = record.get('output_token_id', None)
        
        # 优先使用output_token_id，如果没有则使用input_ids
        token_id = None
        if output_token_id is not None:
            token_id = output_token_id
        elif input_ids and len(input_ids) > 0:
            token_id = input_ids[0]
        
        # 过滤EOS/PAD token
        if token_id is not None and token_id in eos_token_ids:
            total_skipped += 1
            continue
        
        topk_ids_of_layer = record.get('topk_ids_of_layer', [])
        topk_weights_of_layer = record.get('topk_weights_of_layer', [])
        
        # 如果没有topk数据，跳过
        if not topk_ids_of_layer or len(topk_ids_of_layer) == 0:
            total_skipped += 1
            continue
        
        # 遍历每一层
        num_layers = len(topk_ids_of_layer)
        for layer_idx in range(num_layers):
            layer_topk_ids = topk_ids_of_layer[layer_idx]
            
            # layer_topk_ids的形状是 [batch_size][topk]
            # 通常batch_size=1，所以layer_topk_ids = [[topk_ids]]
            if not layer_topk_ids or len(layer_topk_ids) == 0:
                continue
            
            # 处理每个batch（通常是1个）
            for batch_idx, batch_topk_ids in enumerate(layer_topk_ids):
                if not batch_topk_ids:
                    continue
                
                # batch_topk_ids是一个列表，包含topk个专家ID
                # 过滤-1和256（256可能是特殊值，需要检查）
                for expert_id in batch_topk_ids:
                    if expert_id >= 0 and expert_id < 256:  # 过滤-1和256
                        layer_expert_counts[layer_idx][expert_id] += 1
                
                # 处理权重（如果有）
                if topk_weights_of_layer and layer_idx < len(topk_weights_of_layer):
                    layer_topk_weights = topk_weights_of_layer[layer_idx]
                    if batch_idx < len(layer_topk_weights):
                        batch_topk_weights = layer_topk_weights[batch_idx]
                        
                        if batch_topk_weights and len(batch_topk_weights) == len(batch_topk_ids):
                            for i, weight in enumerate(batch_topk_weights):
                                expert_id = batch_topk_ids[i]
                                if expert_id >= 0 and expert_id < 256:
                                    layer_expert_weights[layer_idx][expert_id] += float(weight)
        
        total_processed += 1
    
    stats = {
        'processed': total_processed,
        'skipped': total_skipped
    }
    
    return layer_expert_counts, layer_expert_weights, stats


def process_files_streaming(expert_data_dir: str, inference_results: List[Dict], 
                           tokenizer, all_token_ids: List, request_boundaries: List, 
                           request_prompts: List, eos_token_ids: set,
                           aligned_data_file: str, num_requests: int = None):
    """
    流式处理文件夹中的expert数据文件，逐个文件处理以避免内存溢出
    
    Args:
        expert_data_dir: 包含expert_distribution_data JSON文件的文件夹路径
        inference_results: inference结果列表
        tokenizer: tokenizer对象
        all_token_ids: 所有token_ids列表
        request_boundaries: 请求边界列表
        request_prompts: 请求prompt列表
        eos_token_ids: 需要过滤的token ID集合
        aligned_data_file: 对齐后数据的输出文件路径
        num_requests: 要处理的请求数量
    
    Returns:
        layer_expert_counts: 累加后的专家选择计数字典
        layer_expert_weights: 累加后的专家权重字典
        total_stats: 总体统计信息
    """
    # 查找所有JSON文件
    json_pattern = os.path.join(expert_data_dir, '*.json')
    json_files = sorted(glob.glob(json_pattern))
    
    if not json_files:
        raise ValueError(f"在文件夹 {expert_data_dir} 中未找到任何JSON文件")
    
    print(f"找到 {len(json_files)} 个JSON文件，将逐个处理以避免内存溢出")
    
    # 初始化全局统计字典（内存中只保持这个）
    layer_expert_counts = defaultdict(lambda: defaultdict(int))
    layer_expert_weights = defaultdict(lambda: defaultdict(float))
    
    # 全局token索引（用于流式对齐）
    global_token_index = 0
    
    # 总体统计
    total_processed = 0
    total_skipped = 0
    
    # 确保输出目录存在
    output_dir = os.path.dirname(aligned_data_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 如果文件已存在，先删除（因为我们用追加模式，需要确保从头开始）
    if os.path.exists(aligned_data_file):
        os.remove(aligned_data_file)
        print(f"已删除旧的输出文件: {aligned_data_file}")
    
    # 逐个文件处理
    for file_idx, file_path in enumerate(tqdm(json_files, desc="处理文件")):
        try:
            print(f"\n[{file_idx + 1}/{len(json_files)}] 处理文件: {os.path.basename(file_path)}")
            
            # 加载当前文件的records
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = data.get('records', [])
            if not records:
                print(f"  跳过：文件为空")
                continue
            
            print(f"  加载了 {len(records)} 条记录")
            
            # 流式对齐当前文件的records
            aligned_data = align_tokens_with_experts_streaming(
                inference_results, 
                records, 
                tokenizer,
                start_token_index=global_token_index,
                request_boundaries=request_boundaries,
                request_prompts=request_prompts,
                all_token_ids=all_token_ids
            )
            
            if not aligned_data:
                print(f"  跳过：对齐后数据为空")
                # 更新全局token索引（即使没有对齐数据，也要跳过这些records）
                global_token_index += len(records)
                continue
            
            print(f"  对齐了 {len(aligned_data)} 条记录")
            
            # 立即分析并累加到全局统计字典
            layer_expert_counts, layer_expert_weights, stats = analyze_expert_selection(
                aligned_data,
                eos_token_ids=eos_token_ids,
                layer_expert_counts=layer_expert_counts,
                layer_expert_weights=layer_expert_weights
            )
            
            total_processed += stats['processed']
            total_skipped += stats['skipped']
            
            # 立即保存对齐后的数据（追加模式）
            save_results_jsonl(aligned_data, aligned_data_file, append_mode=True)
            
            # 更新全局token索引
            global_token_index += len(records)
            
            # 释放内存
            del records
            del aligned_data
            del data
            
            # 强制垃圾回收（有助于及时释放内存）
            gc.collect()
            
            print(f"  处理完成，累计处理 {total_processed} 条记录，跳过 {total_skipped} 条")
            
        except Exception as e:
            print(f"错误: 处理文件 {file_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            # 继续处理下一个文件
            continue
    
    total_stats = {
        'processed': total_processed,
        'skipped': total_skipped,
        'total_files': len(json_files)
    }
    
    print(f"\n流式处理完成:")
    print(f"  处理文件数: {len(json_files)}")
    print(f"  成功处理记录: {total_processed}")
    print(f"  跳过记录: {total_skipped}")
    print(f"  总层数: {len(layer_expert_counts)}")
    
    return layer_expert_counts, layer_expert_weights, total_stats


def create_heatmap_data(layer_expert_counts, num_layers=None, max_experts=None):
    """
    创建热力图数据矩阵
    
    Args:
        layer_expert_counts: 专家选择计数字典
        num_layers: 层数（如果为None则自动推断）
        max_experts: 最大专家数（如果为None则自动推断）
    
    Returns:
        heatmap_matrix: numpy数组，形状为 (num_layers, max_experts)
    """
    if num_layers is None:
        num_layers = max(layer_expert_counts.keys()) + 1 if layer_expert_counts else 1
    
    if max_experts is None:
        max_expert_id = 0
        for layer_data in layer_expert_counts.values():
            if layer_data:
                max_expert_id = max(max_expert_id, max(layer_data.keys()))
        max_experts = max_expert_id + 1
    
    # 创建矩阵
    heatmap_matrix = np.zeros((num_layers, max_experts))
    
    for layer_idx in range(num_layers):
        if layer_idx in layer_expert_counts:
            for expert_id, count in layer_expert_counts[layer_idx].items():
                if 0 <= expert_id < max_experts:
                    heatmap_matrix[layer_idx, expert_id] = count
    
    return heatmap_matrix


def visualize_expert_selection(heatmap_matrix, output_dir: str, save_formats: list = None):
    """
    可视化专家选择热力图
    
    Args:
        heatmap_matrix: 热力图数据矩阵，形状为 (num_layers, num_experts)
        output_dir: 输出目录
        save_formats: 保存格式列表，如 ['png', 'pdf']
    """
    if save_formats is None:
        save_formats = ['png', 'pdf']
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_layers, num_experts = heatmap_matrix.shape
    
    print(f"\n生成热力图: {num_layers} 层 x {num_experts} 专家")
    
    # 创建热力图
    plt.figure(figsize=(max(12, num_experts * 0.3), max(6, num_layers * 0.3)))
    
    sns.heatmap(
        heatmap_matrix,
        cmap='YlOrRd',
        xticklabels=[f'Expert {i}' for i in range(num_experts)],
        yticklabels=[f'Layer {i}' for i in range(num_layers)],
        cbar=True,
        cbar_kws={'label': 'Selection Count'},
        fmt='.0f' if heatmap_matrix.max() < 1000 else '.0e'
    )
    
    plt.title('Expert Selection Count by Layer and Expert', fontsize=14, pad=20)
    plt.xlabel('Experts', fontsize=12)
    plt.ylabel('Layers', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存为多种格式
    for fmt in save_formats:
        output_path = os.path.join(output_dir, f'expert_selection_heatmap.{fmt}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"保存热力图: {output_path}")
    
    plt.close()
    
    # 保存数据为CSV
    csv_path = os.path.join(output_dir, 'expert_selection_counts.csv')
    np.savetxt(csv_path, heatmap_matrix, delimiter=',', fmt='%.0f')
    print(f"保存CSV数据: {csv_path}")


def save_statistics(layer_expert_counts, layer_expert_weights, output_dir: str):
    """
    保存统计信息到JSON文件
    
    Args:
        layer_expert_counts: 专家选择计数字典
        layer_expert_weights: 专家权重字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为可序列化格式
    stats = {
        'layer_expert_counts': {
            str(layer): {str(expert): count for expert, count in expert_data.items()}
            for layer, expert_data in layer_expert_counts.items()
        },
        'layer_expert_weights': {
            str(layer): {str(expert): float(weight) for expert, weight in expert_data.items()}
            for layer, expert_data in layer_expert_weights.items()
        }
    }
    
    stats_path = os.path.join(output_dir, 'expert_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"保存统计信息: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='完整分析：对齐tokens与experts，并进行统计分析')
    parser.add_argument('--inference_file', type=str, 
                       default=r'merged_test_val.jsonl',
                       help='inference_results.jsonl文件路径')
    parser.add_argument('--expert_data_file', type=str, 
                       default=r'expert_distribution_data_1767219917.json',
                       help='expert_distribution_data JSON文件路径或包含JSON文件的文件夹路径（如果是文件夹，会合并所有JSON文件的数据）')
    parser.add_argument('--tokenizer_path', type=str, default='deepseek_moe',
                       help='tokenizer路径，用于编码文本为token_ids')
    parser.add_argument('--num_requests', type=int, default=None,
                       help='要处理的请求数量，如果为None则处理所有')
    parser.add_argument('--save_formats', type=str, default='png,pdf',
                       help='保存格式，用逗号分隔（如: png,pdf）')
    parser.add_argument('--num_layers', type=int, default=None,
                       help='层数（如果为None则自动推断）')
    parser.add_argument('--num_experts', type=int, default=None,
                       help='专家数（如果为None则自动推断）')
    parser.add_argument('--output_base_dir', type=str, default='./results_deepseek',
                       help='输出基础目录，结果将保存到该目录下的expert_statistics文件夹')
    
    args = parser.parse_args()
    
    # 确定输出目录
    output_dir = os.path.join(args.output_base_dir, 'expert_statistics')
    os.makedirs(output_dir, exist_ok=True)
    
    # 对齐后的数据保存路径
    aligned_data_file = os.path.join(output_dir, 'results_all.jsonl')
    
    # 加载tokenizer
    tokenizer = None
    try:
        print("=" * 80)
        print("步骤0: 加载tokenizer")
        print("=" * 80)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        print(f"Tokenizer加载成功: {args.tokenizer_path}")
    except Exception as e:
        print(f"警告: 无法加载tokenizer ({args.tokenizer_path}): {e}")
        print("将无法解码token_id为token文本")
    
    # 步骤1: 加载inference结果
    print("\n" + "=" * 80)
    print("步骤1: 加载inference结果")
    print("=" * 80)
    inference_results = load_inference_results(args.inference_file, tokenizer=tokenizer)
    print(f"加载了 {len(inference_results)} 个请求的结果")
    
    # 准备token信息（用于流式处理）
    print("\n" + "=" * 80)
    print("步骤1.5: 准备token信息")
    print("=" * 80)
    all_token_ids, request_boundaries, request_prompts = prepare_token_info(
        inference_results, 
        num_requests=args.num_requests
    )
    print(f"总token数量: {len(all_token_ids)}")
    print(f"请求数量: {len(request_boundaries)}")
    
    # 获取EOS token IDs
    eos_token_ids = get_eos_token_ids(tokenizer)
    
    # 步骤2-5: 根据expert_data_file是文件还是文件夹选择处理方式
    print("\n" + "=" * 80)
    print("步骤2-5: 处理专家分布数据")
    print("=" * 80)
    
    if os.path.isdir(args.expert_data_file):
        # 文件夹：使用流式处理
        print(f"检测到文件夹，使用流式处理模式（逐个文件处理以避免内存溢出）")
        layer_expert_counts, layer_expert_weights, total_stats = process_files_streaming(
            expert_data_dir=args.expert_data_file,
            inference_results=inference_results,
            tokenizer=tokenizer,
            all_token_ids=all_token_ids,
            request_boundaries=request_boundaries,
            request_prompts=request_prompts,
            eos_token_ids=eos_token_ids,
            aligned_data_file=aligned_data_file,
            num_requests=args.num_requests
        )
        print(f"\n流式处理统计:")
        print(f"  处理文件数: {total_stats['total_files']}")
        print(f"  成功处理记录: {total_stats['processed']}")
        print(f"  跳过记录: {total_stats['skipped']}")
    else:
        # 单个文件：使用原有方式（向后兼容）
        print(f"检测到单个文件，使用传统处理模式")
        expert_data = load_expert_distribution_data(args.expert_data_file)
        
        # 步骤3: 对齐tokens与experts
        print("\n" + "=" * 80)
        print("步骤3: 对齐tokens与experts")
        print("=" * 80)
        aligned_data = align_tokens_with_experts(
            inference_results, 
            expert_data, 
            tokenizer=tokenizer,
            num_requests=args.num_requests
        )
        
        # 步骤4: 保存对齐后的数据
        print("\n" + "=" * 80)
        print("步骤4: 保存对齐后的数据")
        print("=" * 80)
        save_results_jsonl(aligned_data, aligned_data_file)
        
        # 步骤5: 分析专家选择（基于对齐后的数据）
        print("\n" + "=" * 80)
        print("步骤5: 分析专家选择")
        print("=" * 80)
        layer_expert_counts, layer_expert_weights, stats = analyze_expert_selection(aligned_data, eos_token_ids)
    
    # 步骤6: 创建热力图数据
    print("\n" + "=" * 80)
    print("步骤6: 创建热力图数据")
    print("=" * 80)
    heatmap_matrix = create_heatmap_data(
        layer_expert_counts,
        num_layers=args.num_layers,
        max_experts=args.num_experts
    )
    
    # 步骤7: 生成热力图
    print("\n" + "=" * 80)
    print("步骤7: 生成热力图")
    print("=" * 80)
    save_formats = [fmt.strip() for fmt in args.save_formats.split(',')]
    visualize_expert_selection(heatmap_matrix, output_dir, save_formats)
    
    # 步骤8: 保存统计信息
    print("\n" + "=" * 80)
    print("步骤8: 保存统计信息")
    print("=" * 80)
    save_statistics(layer_expert_counts, layer_expert_weights, output_dir)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print(f"所有结果保存在: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
