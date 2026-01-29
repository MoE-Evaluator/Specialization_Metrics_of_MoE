#!/usr/bin/env python3
"""
Qwen3 MoE Expert Logger 演示脚本
"""

import os
import json
import requests
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

class MoEClient:
    """MoE API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url
        self.api_url = f"{base_url}/chat/completions"
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """发送聊天请求"""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "default",
            "messages": messages,
            "return_token_ids": True,
            **kwargs
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        print("results:", response.json())
        return response.json()
    
    def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        """流式聊天请求"""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "default",
            "messages": messages,
            "stream": True,
            "return_token_ids": True,
            **kwargs
        }
        
        response = requests.post(self.api_url, headers=headers, json=data, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str != '[DONE]':
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            pass

class ExpertLogAnalyzer:
    """专家日志分析器"""
    
    def __init__(self, log_dir: str = "./expert_logs"):
        self.log_dir = Path(log_dir)
    
    def list_log_files(self) -> List[Path]:
        """列出所有日志文件"""
        if not self.log_dir.exists():
            return []
        return sorted(self.log_dir.glob("*.jsonl"))
    
    def read_log_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """读取日志文件"""
        tokens = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    tokens.append(json.loads(line))
        return tokens
    
    def analyze_expert_usage(self, tokens: List[Dict[str, Any]]) -> Dict[int, Dict[int, int]]:
        """分析专家使用情况"""
        layer_expert_count = defaultdict(lambda: defaultdict(int))
        
        for token_data in tokens:
            for layer_info in token_data.get('expert_detail', []):
                layer_idx = layer_info['layer_idx']
                topk_experts = layer_info.get('topk_experts', [])
                
                for expert_idx in topk_experts:
                    layer_expert_count[layer_idx][expert_idx] += 1
        
        return dict(layer_expert_count)
    
    def print_statistics(self, tokens: List[Dict[str, Any]]):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print(f"Expert Routing Statistics")
        print(f"{'='*60}")
        print(f"Total tokens analyzed: {len(tokens)}")
        
        if not tokens:
            print("No tokens found in log file")
            return
        
        # 分析专家使用
        layer_expert_count = self.analyze_expert_usage(tokens)
        
        if not layer_expert_count:
            print("No expert routing data found")
            return
        
        # 打印每层的专家使用情况
        print(f"\nExpert Usage by Layer:")
        print(f"{'-'*60}")
        
        for layer_idx in sorted(layer_expert_count.keys()):
            expert_counts = layer_expert_count[layer_idx]
            total = sum(expert_counts.values())
            
            print(f"\nLayer {layer_idx} (Total selections: {total}):")
            
            # 按使用次数排序
            sorted_experts = sorted(
                expert_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for expert_idx, count in sorted_experts:
                percentage = (count / total) * 100
                bar_length = int(percentage / 2)  # 每 2% 一个字符
                bar = '█' * bar_length
                print(f"  Expert {expert_idx:2d}: {count:4d} times ({percentage:5.1f}%) {bar}")
        
        # 计算专家负载均衡度
        print(f"\n{'='*60}")
        print(f"Load Balancing Analysis:")
        print(f"{'-'*60}")
        
        for layer_idx in sorted(layer_expert_count.keys()):
            expert_counts = layer_expert_count[layer_idx]
            if not expert_counts:
                continue
            
            counts = list(expert_counts.values())
            num_experts = len(counts)
            avg_count = sum(counts) / num_experts
            
            # 计算方差（负载均衡度）
            variance = sum((c - avg_count) ** 2 for c in counts) / num_experts
            std_dev = variance ** 0.5
            cv = (std_dev / avg_count) * 100 if avg_count > 0 else 0  # 变异系数
            
            print(f"Layer {layer_idx}:")
            print(f"  Number of active experts: {num_experts}")
            print(f"  Average selections per expert: {avg_count:.1f}")
            print(f"  Standard deviation: {std_dev:.1f}")
            print(f"  Coefficient of variation: {cv:.1f}%")
            print(f"  {'Well balanced' if cv < 30 else 'Moderate balance' if cv < 50 else 'Poor balance'}")

def demo_basic_usage():
    """基本使用演示"""
    print("="*60)
    print("Qwen3 MoE Expert Logger Demo")
    print("="*60)
    
    # 初始化客户端
    client = MoEClient()
    
    # 发送请求
    print("\n1. Sending chat completion request...")
    messages = [
        {"role": "user", "content": "请简要解释一下混合专家模型（MoE）的工作原理"}
    ]
    
    try:
        response = client.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        print("\nResponse:")
        print(response['choices'][0]['message']['content'])
        print("全部结果:", response)
        
        # 等待日志写入
        print("\n2. Waiting for expert logs to be written...")
        time.sleep(3)
        
        # 分析日志
        print("\n3. Analyzing expert logs...")
        analyzer = ExpertLogAnalyzer()
        log_files = analyzer.list_log_files()
        
        if not log_files:
            print("No log files found. Please check:")
            print("  1. VLLM_ENABLE_EXPERT_LOGGER=1 environment variable is set")
            print("  2. OUTPUT_DIR environment variable is set correctly")
            print("  3. vLLM server is running with the modified code")
            print("  4. The model is Qwen3 MoE")
            return
        
        # 分析最新的日志文件
        latest_file = log_files[-1]
        print(f"Reading log file: {latest_file.name}")
        
        tokens = analyzer.read_log_file(latest_file)
        analyzer.print_statistics(tokens)
        
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to vLLM server.")
        print("Please make sure the server is running on http://localhost:8000")
        print("\nTo start the server with expert logging enabled, run:")
        print("  export VLLM_ENABLE_EXPERT_LOGGER=1")
        print("  export OUTPUT_DIR=./expert_logs")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("      --model Qwen/Qwen3-MoE-15B-A2B \\")
        print("      --host 0.0.0.0 --port 8000 --trust-remote-code")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

def demo_streaming():
    """流式生成演示"""
    print("\n" + "="*60)
    print("Streaming Generation Demo")
    print("="*60)
    
    client = Qwen3MoEClient()
    
    messages = [
        {"role": "user", "content": "解释一下注意力机制在 Transformer 中的作用"}
    ]
    
    print("\nStreaming response:")
    print("-" * 60)
    
    try:
        for chunk in client.chat_stream(messages, temperature=0.7, max_tokens=1024):
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    print(delta['content'], end='', flush=True)
        print("\n" + "-" * 60)
        
        # 等待日志写入
        time.sleep(2)
        
        # 分析日志
        analyzer = ExpertLogAnalyzer()
        log_files = analyzer.list_log_files()
        if log_files:
            latest_file = log_files[-1]
            tokens = analyzer.read_log_file(latest_file)
            print(f"\nLogged {len(tokens)} tokens to {latest_file.name}")
    
    except Exception as e:
        print(f"\nError: {e}")

def demo_multiple_requests():
    """多个请求演示"""
    print("\n" + "="*60)
    print("Multiple Requests Demo")
    print("="*60)
    
    client = Qwen3MoEClient()
    
    prompts = [
        "什么是深度学习？",
        "解释一下反向传播算法",
        "Transformer 模型的主要组件有哪些？"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Request: {prompt}")
        try:
            response = client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            print(f"   Response: {response['choices'][0]['message']['content']}")
            time.sleep(1)  # 短暂等待
        except Exception as e:
            print(f"   Error: {e}")
    
    # 分析所有日志
    print("\n" + "="*60)
    print("Analyzing all log files...")
    print("="*60)
    
    analyzer = ExpertLogAnalyzer()
    log_files = analyzer.list_log_files()
    
    if log_files:
        print(f"\nFound {len(log_files)} log file(s):")
        for log_file in log_files[-3:]:  # 显示最后 3 个文件
            tokens = analyzer.read_log_file(log_file)
            print(f"  {log_file.name}: {len(tokens)} tokens")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3 MoE Expert Logger Demo")
    parser.add_argument(
        "--mode",
        choices=["basic", "stream", "multi"],
        default="multi",
        help="Demo mode: basic, stream, or multi"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/v1",
        help="vLLM API server URL"
    )
    parser.add_argument(
        "--log-dir",
        # default="/data/oceanus_share/yangqianwen/expert_logs",
        default="/data/oceanus_share/wangjing/expert_logs",
        help="Expert log directory"
    )
    
    args = parser.parse_args()
    
    # 设置环境变量（如果未设置）
    if not os.getenv("VLLM_ENABLE_EXPERT_LOGGER"):
        print("Warning: VLLM_ENABLE_EXPERT_LOGGER is not set. Expert logging will be disabled.")
        print("Set VLLM_ENABLE_EXPERT_LOGGER=1 to enable expert logging.")
    if not os.getenv("OUTPUT_DIR"):
        os.environ["OUTPUT_DIR"] = args.log_dir
    
    demo_basic_usage()
    demo_streaming()
    # demo_multiple_requests()
