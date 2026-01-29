import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import os
from scipy.stats import entropy
import argparse


class MoEExpertAnalyzer:
    def __init__(self, model_name: str, domain_matrices: Dict[str, pd.DataFrame]):
        """
        初始化分析器
        :param model_name: 模型名称 (用于报告标识)
        :param domain_matrices: 字典，Key是领域名，Value是DataFrame(行=层, 列=专家)
        """
        self.model_name = model_name
        self.raw_data = domain_matrices
        self.normalized_data = {}
        
        # 存储结果的容器
        self.metrics_df = None      # 领域级详细指标
        self.similarity_df = None   # 领域间相似度矩阵
        self.model_score = None     # 模型级综合评分
        
        # 预处理：归一化
        self._normalize_data()

    def _normalize_data(self):
        """数据预处理：行归一化 (Row-wise Normalization)"""
        for domain, df in self.raw_data.items():
            matrix = df.values.astype(float)
            # 计算每层的总数
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            # 得到概率矩阵 P(Expert|Layer)
            print(row_sums.shape)
            self.normalized_data[domain] = matrix / row_sums

    def analyze_indicators(self):
        """
        计算领域级的详细指标，包括特化度、隔离度和多样性
        """
        results = []
        
        # 1. 先计算基础指标 (特化度、多样性)
        for domain, matrix in self.normalized_data.items():
            n_layers, n_experts = matrix.shape
            print(f"Domain: {domain}, Layers: {n_layers}, Experts: {n_experts}")
            
            # --- A. Routing Specialization (专注度/特化度) ---
            # 使用 KL 散度衡量
            uniform_dist = np.full(n_experts, 1.0 / n_experts)
            epsilon = 1e-10
            kl_scores = []
            for row in matrix:
                p_safe = row + epsilon
                p_safe = p_safe / np.sum(p_safe)
                kl = np.sum(p_safe * np.log(p_safe / uniform_dist))
                kl_scores.append(kl)
            routing_spec = np.mean(kl_scores)

            # --- B. Expert Diversity/Efficiency (内部多样性) ---
            # 使用 归一化有效秩 (Normalized Effective Rank), 衡量在选定的活跃专家中，是否各司其职
            U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
            effective_rank = (np.sum(s) ** 2) / np.sum(s ** 2)
            norm_diversity = effective_rank / min(n_layers, n_experts)
            norm_diversity = min(norm_diversity, 1.0)
            
            # 条件数
            cond_number = s[0] / (s[-1] + 1e-10)

            results.append({
                'Domain': domain,
                'Routing_Specialization': routing_spec,  # 核心指标 1: KL
                'Internal_Diversity': norm_diversity,    # Norm Rank 
                'Condition_Number': cond_number,         # 参考
                'Effective_Rank': effective_rank,        # 参考
                # 'Active_Experts': n_active               # 参考
            })
        
        self.metrics_df = pd.DataFrame(results).set_index('Domain')

        # 2. 计算相似度矩阵
        self._compute_domain_similarity()

        # 3. 计算 Domain Isolation (隔离度) 并合并到 metrics_df
        # Isolation = 1 - Avg(Cosine Similarity with others)
        isolation_scores = []
        domains = self.metrics_df.index.tolist()
        
        if len(domains) > 1:
            for domain in domains:
                # 获取该领域与其他领域的相似度 (排除自己)
                other_sims = self.similarity_df.loc[domain].drop(domain)
                avg_sim = other_sims.mean()
                isolation = 1.0 - avg_sim
                isolation_scores.append(isolation)
        else:
            isolation_scores = [0.0] * len(domains) # 只有一个领域
            
        self.metrics_df['Domain_Isolation'] = isolation_scores 

        return self.metrics_df

    def _compute_domain_similarity(self):
        """内部辅助函数：计算相似度矩阵"""
        domains = list(self.normalized_data.keys())
        vectors = np.array([self.normalized_data[d].flatten() for d in domains])
        sim_matrix = cosine_similarity(vectors)
        self.similarity_df = pd.DataFrame(sim_matrix, index=domains, columns=domains)

    def calculate_model_score(self):
        """
        计算模型级的综合评分 (ESI - Expert Specialization Index)
        """
        if self.metrics_df is None:
            self.analyze_indicators()
            
        # 获取各维度的平均值
        avg_spec = self.metrics_df['NRS'].mean()
        avg_iso = self.metrics_df['Domain_Isolation'].mean()
        # avg_div = self.metrics_df['Internal_Diversity'].mean()
        
        # ESI 综合计算逻辑 (加权)
        # 归一化说明: KL散度通常在0.5-2.0之间，除以2使其落入0-1区间以便加权
        # 权重: 特化(50%) + 隔离(50%)
        esi_score = (0.5 * (avg_spec / 2.0)) + (0.5 * avg_iso)
        
        self.model_score = pd.DataFrame([{
            'Model_Name': self.model_name,
            'ESI_Total_Score': round(esi_score, 4),
            'Avg_Specialization (KL)': round(avg_spec, 4),
            'Avg_Isolation (1-Sim)': round(avg_iso, 4),
            # 'Avg_Diversity (NormRank)': round(avg_div, 4)
        }])
        
        return self.model_score

    def save_analysis_report(self, output_dir: str):
        """保存包含三个 Sheet 的完整 Excel 报告"""
        # 确保计算已完成
        if self.metrics_df is None:
            self.analyze_indicators()
        if self.model_score is None:
            self.calculate_model_score()
            
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{self.model_name}_moe_analysis_report_new.xlsx')
        
        with pd.ExcelWriter(file_path) as writer:
            # Sheet 1: 模型综合评分 (最重要)
            self.model_score.to_excel(writer, sheet_name='Model_Summary', index=False)
            # Sheet 2: 领域详细指标
            self.metrics_df.to_excel(writer, sheet_name='Domain_Metrics')
            # Sheet 3: 相似度矩阵
            self.similarity_df.to_excel(writer, sheet_name='Similarity_Matrix')
            
        print(f"[成功] 完整报告已保存至: {file_path}")
        print("Model Summary Preview:")
        print(self.model_score.to_string(index=False))

    def analyze_pca_clustering(self):
        """PCA 分析"""
        domains = list(self.normalized_data.keys())
        flattened_vectors = [self.normalized_data[d].flatten() for d in domains]
        X = np.array(flattened_vectors)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        return X_pca, domains, pca.explained_variance_ratio_

    def plot_pca(self, X_pca, domains, explained_variance, output_dir):
        """绘制 PCA 散点图"""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], s=120, hue=domains, style=domains)
        for i, domain in enumerate(domains):
            plt.text(X_pca[i, 0]+0.02, X_pca[i, 1]+0.02, domain, fontsize=9)
        plt.title(f'MoE Domain Routing Map ({self.model_name})\n(PC1: {explained_variance[0]:.1%}, PC2: {explained_variance[1]:.1%})')
        plt.xlabel('PC1 (Main Routing Difference)')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.model_name}_pca_map.png'), dpi=300)
        print(f"[成功] PCA 图已保存")

    def analyze_umap_clustering(self, n_neighbors=15, min_dist=0.1, metric='cosine'):
        """对各领域的专家路由分布做 UMAP 降维"""
        domains = list(self.normalized_data.keys())
        flattened_vectors = [self.normalized_data[d].flatten() for d in domains]
        X = np.array(flattened_vectors)

        umap_model = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        X_umap = umap_model.fit_transform(X)

        return X_umap, domains

    def plot_umap(self, X_umap, domains, output_dir):
        """绘制 UMAP 2D 可视化"""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], s=120, hue=domains, style=domains)
        for i, domain in enumerate(domains):
            plt.text(X_umap[i, 0] + 0.02, X_umap[i, 1] + 0.02, domain, fontsize=9)
        plt.title(f'MoE Domain Routing Map (UMAP) - {self.model_name}')
        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.model_name}_umap_map.png'), dpi=300)
        print(f"[成功] UMAP 图已保存")


class MoELayerAnalyzer:
    def __init__(self, model_name: str, domain_matrices: Dict[str, pd.DataFrame]):
        """
        MoE 层级动态分析器
        """
        self.model_name = model_name
        self.raw_data = domain_matrices
        self.normalized_data = {}
        self.n_layers = 0
        
        # 结果容器
        self.layer_spec_df = None  # 层级特化度 (Rows=Layers, Cols=Domains)
        self.layer_iso_df = None   # 层级隔离度 (Rows=Layers, Cols=Domains)
        self.layer_corr_matrices = {} # 层间相关性矩阵字典
        
        self._normalize_data()

    def _normalize_data(self):
        """预处理：行归一化"""
        for domain, df in self.raw_data.items():
            matrix = df.values.astype(float)
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.normalized_data[domain] = matrix / row_sums
            
            # 记录层数 (假设所有领域层数一致)
            if self.n_layers == 0:
                self.n_layers = matrix.shape[0]

    def analyze_layer_trajectories(self):
        """
        计算各领域的层级演化轨迹：
        1. Routing Specialization (Layer-wise)
        2. Domain Isolation (Layer-wise)
        """
        spec_data = {}
        iso_data = {}
        
        domains = list(self.normalized_data.keys())
        
        # --- 1. 计算 Layer-wise Specialization ---
        for domain, matrix in self.normalized_data.items():
            n_experts = matrix.shape[1]
            uniform_dist = np.full(n_experts, 1.0 / n_experts)
            epsilon = 1e-10
            
            # 对每一层分别计算 KL 散度
            layer_kls = []
            for row in matrix:
                p_safe = row + epsilon
                p_safe = p_safe / np.sum(p_safe)
                kl = np.sum(p_safe * np.log(p_safe / uniform_dist))
                layer_kls.append(kl)
            
            spec_data[domain] = layer_kls

        self.layer_spec_df = pd.DataFrame(spec_data)
        self.layer_spec_df.index.name = 'Layer_Depth'

        # --- 2. 计算 Layer-wise Isolation ---
        # 对于每一层，计算该领域向量与其他领域向量的平均距离 (1 - Cosine Sim)
        for target_domain in domains:
            layer_isos = []
            for l in range(self.n_layers):
                # 提取第 l 层的所有领域向量
                target_vec = self.normalized_data[target_domain][l].reshape(1, -1)
                
                other_sims = []
                for other_domain in domains:
                    if other_domain == target_domain:
                        continue
                    other_vec = self.normalized_data[other_domain][l].reshape(1, -1)
                    sim = cosine_similarity(target_vec, other_vec)[0][0]
                    other_sims.append(sim)
                
                # Isolation = 1 - Avg Similarity
                avg_sim = np.mean(other_sims) if other_sims else 1.0
                layer_isos.append(1.0 - avg_sim)
                
            iso_data[target_domain] = layer_isos
            
        self.layer_iso_df = pd.DataFrame(iso_data)
        self.layer_iso_df.index.name = 'Layer_Depth'

    def analyze_layer_correlations(self, target_domains: List[str] = None):
        """
        计算层间相关性矩阵 (Layer-to-Layer Similarity)
        分析第 i 层的专家分布与第 j 层的专家分布是否相似
        :param target_domains: 指定分析哪些领域，None则分析所有
        """
        if target_domains is None:
            target_domains = list(self.normalized_data.keys())
            
        for domain in target_domains:
            if domain not in self.normalized_data:
                continue
                
            matrix = self.normalized_data[domain]
            # 计算余弦相似度矩阵 (Layer x Layer)
            sim_matrix = cosine_similarity(matrix)
            self.layer_corr_matrices[domain] = pd.DataFrame(sim_matrix)

    def analyze_specific_layer_similarity(self, layer_indices: List[int], output_dir: str):
        """
        计算特定层在不同领域之间的相似度矩阵，并绘制热力图
        :param layer_indices: 要分析的层索引列表，如 [0, 48, 93]
        :param output_dir: 保存路径
        """
        domains = list(self.normalized_data.keys())
        if not domains:
            return

        # 确保层索引有效
        valid_indices = [idx for idx in layer_indices if 0 <= idx < self.n_layers]
        
        for layer_idx in valid_indices:
            # 构建该层的领域特征矩阵 (Rows=Domains, Cols=Experts)
            layer_domain_matrix = []
            for domain in domains:
                # 获取该领域在第 layer_idx 层的专家分布向量
                vec = self.normalized_data[domain][layer_idx].flatten()
                layer_domain_matrix.append(vec)
            
            layer_domain_matrix = np.array(layer_domain_matrix)
            
            # 计算领域间相似度 (Domains x Domains)
            sim_matrix = cosine_similarity(layer_domain_matrix)
            sim_df = pd.DataFrame(sim_matrix, index=domains, columns=domains)
            
            # 绘图
            plt.figure(figsize=(10, 8))
            sns.heatmap(sim_df, cmap='OrRd', annot=True, fmt=".2f", square=True, vmin=0, vmax=1)
            
            # 标题
            if layer_idx == 0:
                pos_str = "First Layer"
            elif layer_idx == self.n_layers - 1:
                pos_str = "Last Layer"
            else:
                pos_str = f"Layer {layer_idx}"
                
            plt.title(f'Domain Similarity at {pos_str} (ID: {layer_idx}) - {self.model_name}')
            plt.tight_layout()
            
            save_path = os.path.join(output_dir, f'{self.model_name}_layer_{layer_idx}_domain_sim.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"[成功] 第 {layer_idx} 层领域相似度热力图已保存: {save_path}")

    def save_results(self, output_dir: str):
        """保存层级分析结果到 Excel"""
        if self.layer_spec_df is None:
            self.analyze_layer_trajectories()
            
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f'{self.model_name}_layerwise_analysis.xlsx')
        
        with pd.ExcelWriter(file_path) as writer:
            self.layer_spec_df.to_excel(writer, sheet_name='Layer_Specialization')
            self.layer_iso_df.to_excel(writer, sheet_name='Layer_Isolation')
            
            # 保存前几个领域的层间相关性矩阵 (避免文件过大)
            for domain, df in list(self.layer_corr_matrices.items())[:3]:
                df.to_excel(writer, sheet_name=f'Corr_{domain[:10]}')
                
        print(f"[成功] 层级分析数据已保存至: {file_path}")

    def plot_trajectories(self, output_dir: str, spec_ylim: tuple = (0, 2.0), iso_ylim: tuple = (0, 1.05)):
        """绘制层级演化曲线图"""
        if self.layer_spec_df is None:
            return

        # 1. 绘制 Specialization 曲线
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.layer_spec_df, dashes=False, palette="tab10", linewidth=2.5)
        plt.title(f'Layer-wise Routing Specialization ({self.model_name})', fontsize=14)
        plt.xlabel('Layer Depth (0 = Input, N = Output)', fontsize=12)
        plt.ylabel('Specialization (KL Divergence)', fontsize=12)

        if spec_ylim:
            plt.ylim(spec_ylim)

        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.model_name}_layer_spec_curve.png'), dpi=300)
        plt.close()

        # 2. 绘制 Isolation 曲线
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.layer_iso_df, dashes=False, palette="tab10", linewidth=2.5)
        plt.title(f'Layer-wise Domain Isolation ({self.model_name})', fontsize=14)
        plt.xlabel('Layer Depth', fontsize=12)
        plt.ylabel('Isolation (1 - Cosine Sim)', fontsize=12)

        if iso_ylim:
            plt.ylim(iso_ylim)

        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.model_name}_layer_iso_curve.png'), dpi=300)
        plt.close()
        
        print(f"[成功] 层级演化曲线图已保存")

    def plot_layer_heatmap(self, domain: str, output_dir: str):
        """绘制特定领域的层间相关性热力图"""
        if domain not in self.layer_corr_matrices:
            return
            
        matrix = self.layer_corr_matrices[domain]
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, cmap='viridis', square=True)
        plt.title(f'Layer-to-Layer Correlation: {domain} ({self.model_name})')
        plt.xlabel('Layer ID')
        plt.ylabel('Layer ID')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.model_name}_{domain}_layer_corr.png'), dpi=300)
        plt.close()
        print(f"[成功] {domain} 层间相关性热力图已保存")


def main():
    parser = argparse.ArgumentParser(description='MoE 模型专家分布分析脚本')
    parser.add_argument('--result_save_dir', type=str, default=None, help='结果保存路径 (如果不填，默认与输入路径相同)')
    parser.add_argument('--model_name', type=str, required=True, help='模型名称 (例如: qwen_235B)')

    # 使用字典形式的领域参数
    parser.add_argument('--domain_config', type=str, default=None,
                       help='JSON格式的领域配置')
  
    args = parser.parse_args()

    result_save_dir = args.result_save_dir
    model_name = args.model_name

    domain_data = {}
    if args.domain_config:
        import json
        config = json.loads(args.domain_config)
        for domain_name, domain_csv_path in config.items():
            file_path = os.path.join(domain_csv_path)
            if os.path.exists(file_path):
                domain_data[domain_name] = pd.read_csv(file_path, header=None)
            else:
                print(f"警告: 文件不存在: {file_path}")

    # 3. 初始化分析器
    analyzer = MoEExpertAnalyzer(model_name, domain_data)

    # 4. 计算指标并保存
    print(f"正在分析模型: {model_name} ...")
    analyzer.save_analysis_report(output_dir=result_save_dir)

    # 5. PCA 画图
    X_pca, domains, variance = analyzer.analyze_pca_clustering()
    analyzer.plot_pca(X_pca, domains, variance, output_dir=result_save_dir)

    # 6. 层级动态分析 (Layer-wise Analysis)
    print("\n正在进行层级动态分析 (Layer-wise Analysis) ...")
    layer_analyzer = MoELayerAnalyzer(model_name, domain_data)
    
    # 计算并保存数据
    layer_analyzer.analyze_layer_trajectories()
    
    # 选取领域分析层间相关性
    representative_domains = ['Math', 'Medical', 'Knowledge', 'Science'] 
    valid_domains = [d for d in representative_domains if d in domain_data]
    layer_analyzer.analyze_layer_correlations(target_domains=representative_domains)
    
    layer_analyzer.save_results(output_dir=result_save_dir)
    
    # 绘图
    layer_analyzer.plot_trajectories(output_dir=result_save_dir)
    
    # 为代表性领域画层间热力图
    for domain in valid_domains:
        layer_analyzer.plot_layer_heatmap(domain, output_dir=result_save_dir)
     
    # 7. 对特定层进行分析
    n_layers = list(domain_data.values())[0].shape[0]
    
    # 定义要分析的关键层：第一层、中间层、最后一层
    target_layers = [0, n_layers // 2, n_layers - 1]
    
    print(f"\n正在分析特定层 {target_layers} 的领域相似度 ...")
    layer_analyzer.analyze_specific_layer_similarity(target_layers, output_dir=result_save_dir)


if __name__ == '__main__':
    main()

