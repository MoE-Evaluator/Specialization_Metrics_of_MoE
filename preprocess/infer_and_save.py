#!/usr/bin/env python3
import json
import argparse
import os
import time
import requests
from tqdm import tqdm
from typing import Dict, Any, List, Set
from collections import defaultdict
from infer_backend import MoEClient

# --- 全局配置 ---
DEFAULT_VALID_SOURCES = [
        "aime_2025_messages.jsonl",
        "yale-financemath/validation_messages.jsonl",
        'livecodebench_code_generation/test_messages.jsonl',
        'princeton_SWE_bench_Verified/data/test_set_messages.jsonl',
        "allenai_sciq/data/val_set_messages.jsonl",
        "cais_hle_messages.jsonl",
        "bigbio_medqa_dev_messages.jsonl",
        "bigbio_medqa_test_messages.jsonl",
        'nguha--legalbench/legalbench_messages.jsonl',
]

class MoEInferenceManager:
    def __init__(self, api_url: str, enable_expert_recording: bool = True):
        self.api_url = api_url.rstrip('/')
        self.api_v1 = f"{self.api_url}/v1" if not self.api_url.endswith('/v1') else self.api_url
        self.api_base = self.api_v1.replace('/v1', '')
        self.enable_expert_recording = enable_expert_recording
        self.client = MoEClient(base_url=self.api_v1)

    def _request(self, endpoint: str, method: str = "POST", data: Dict = None):
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        try:
            resp = requests.request(method, url, json=data, timeout=30)
            if resp.status_code == 404:
                return {"error": "404", "msg": "Endpoint not found. Ensure server started with --expert-distribution-recorder-mode per_token"}
            return resp.json() if resp.status_code == 200 else {"error": resp.status_code, "msg": resp.text}
        except Exception as e:
            return {"error": "exception", "msg": str(e)}

    def manage_recording(self, action: str):
        """Action: start, stop, dump"""
        if not self.enable_expert_recording: return True
        
        print(f"执行专家记录操作: {action}...")
        res = self._request(f"{action}_expert_distribution_record")
        
        if "error" in res:
            print(f"❌ 专家记录 {action} 失败: {res['msg']}")
            return False
        return True

    def dump_records(self, output_dir: str, source_name: str):
        if not self.enable_expert_recording: return
        res = self._request("dump_expert_distribution_record")
        if "error" not in res:
            log_dir = os.path.join(output_dir, "expert_logs")
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, f"dist_{source_name}_{int(time.time())}.json")
            with open(path, 'w') as f:
                json.dump(res, f, indent=2)
            return path
        return None

def get_prompt_hash(item):
    """提取唯一标识用于断点续传"""
    if 'messages' in item: return item['messages'][0].get('content', '')
    return item.get('question', '')

def process_single_source(manager, items, output_file, source_name):
    """处理单个数据集的推理逻辑"""
    processed_prompts = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    processed_prompts.add(get_prompt_hash(json.loads(line)['original_data']))
                except: continue

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 开启记录
    if not manager.manage_recording("start"):
        print(f"⚠️ 跳过 {source_name} 的专家记录")

    with open(output_file, 'a', encoding='utf-8') as f_out:
        pbar = tqdm(items, desc=f"Source: {source_name}")
        for idx, item in enumerate(pbar):
            p_hash = get_prompt_hash(item)
            if p_hash in processed_prompts:
                continue

            messages = item.get('messages', [{"role": "user", "content": item.get('question', '')}])
            
            try:
                start_t = time.time()
                response = manager.client.chat(messages=messages, temperature=0.7, max_tokens=1024)
                duration = time.time() - start_t

                if response and 'choices' in response:
                    res_item = {
                        "original_data": item,
                        "generated_response": response['choices'][0]['message']['content'],
                        "latency": duration,
                        "full_response": response
                    }
                    f_out.write(json.dumps(res_item, ensure_ascii=False) + "\n")
                    f_out.flush()

                # 每 20 条 Dump 一次数据
                if (idx + 1) % 20 == 0:
                    manager.dump_records(os.path.dirname(output_file), source_name)
                    
            except Exception as e:
                print(f"推理错误: {e}")

    # 结束收尾
    manager.dump_records(os.path.dirname(output_file), source_name)
    manager.manage_recording("stop")

def run_batch_inference(args):
    manager = MoEInferenceManager(args.api_url, not args.disable_expert_recording)
    
    # 1. 优化点：单次扫描大文件进行内存分发
    print(f"🔍 正在单次扫描并过滤大文件: {args.input_file}")
    source_buckets = defaultdict(list)
    target_sources = set(args.sources)
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                src = item.get('source')
                if src in target_sources:
                    source_buckets[src].append(item)
            except: continue

    # 2. 依次处理有数据的 source
    total_start = time.time()
    for source in args.sources:
        items = source_buckets.get(source, [])
        if not items:
            print(f"❓ Source {source} 在输入文件中未找到匹配数据，跳过。")
            continue

        print(f"\n任务开始: {source} (共 {len(items)} 条)")
        
        # 自动推导路径
        safe_name = source.replace('/', '_').replace('.jsonl', '')
        source_dir = os.path.join(args.root_dir or ".", f"{safe_name}_results")
        output_file = os.path.join(source_dir, f"{os.path.basename(source)}")
        
        process_single_source(manager, items, output_file, safe_name)

    print(f"\n✅ 所有批量任务完成，总耗时: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs='*', default=DEFAULT_VALID_SOURCES)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--api_url", type=str, default="http://localhost:8000")
    parser.add_argument("--root_dir", type=str, default="outputs")
    parser.add_argument("--disable-expert-recording", action="store_true")
    
    args = parser.parse_args()
    run_batch_inference(args)
   