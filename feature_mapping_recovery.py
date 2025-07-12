import pandas as pd
import torch
import json
import numpy as np

# 定义映射关系
GENE_MAP = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
DURATION_MAP = [24, 48, 72, 96]

# 创建反向映射
REVERSE_GENE_MAP = {str(v): k for k, v in GENE_MAP.items()}

def decode_gene_features(gene_vector_str):
    """将基因向量字符串转换回原始基因序列"""
    try:
        # 解析JSON字符串
        gene_matrix = json.loads(gene_vector_str)
        
        # 如果是嵌套列表，需要转置
        if isinstance(gene_matrix[0], list):
            gene_matrix = list(map(list, zip(*gene_matrix)))
        
        # 将每个one-hot向量转换回字符
        gene_sequence = ''
        for vector in gene_matrix:
            vector_str = str(vector)
            if vector_str in REVERSE_GENE_MAP:
                gene_sequence += REVERSE_GENE_MAP[vector_str]
            else:
                gene_sequence += 'N'  # 未知字符
        
        return gene_sequence
    except:
        return 'UNKNOWN'

def decode_duration_features(duration_vector_str):
    """将持续时间向量转换回原始数值"""
    try:
        duration_vector = json.loads(duration_vector_str)
        
        # 找到值为1的索引
        if isinstance(duration_vector, list):
            for i, val in enumerate(duration_vector):
                if val == 1 and i < len(DURATION_MAP):
                    return DURATION_MAP[i]
        
        return 'UNKNOWN'
    except:
        return 'UNKNOWN'

def decode_taxonomy_features(taxonomy_vector_str):
    """将分类向量转换为简化的字符串表示"""
    try:
        taxonomy_vector = json.loads(taxonomy_vector_str)
        
        # 计算向量的统计信息作为简化表示
        if isinstance(taxonomy_vector, list) and len(taxonomy_vector) > 0:
            mean_val = np.mean(taxonomy_vector)
            max_val = np.max(taxonomy_vector)
            min_val = np.min(taxonomy_vector)
            return f"mean:{mean_val:.3f}_max:{max_val:.3f}_min:{min_val:.3f}"
        
        return 'UNKNOWN'
    except:
        return 'UNKNOWN'

def recover_original_features(input_csv, output_csv):
    """恢复原始特征格式"""
    print(f"正在处理文件: {input_csv}")
    
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 处理gene列
    if 'gene' in df.columns:
        print("正在解码gene特征...")
        df['gene_decoded'] = df['gene'].apply(decode_gene_features)
    
    # 处理duration列
    if 'duration' in df.columns:
        print("正在解码duration特征...")
        df['duration_decoded'] = df['duration'].apply(decode_duration_features)
    
    # 处理taxonomy列
    if 'taxonomy' in df.columns:
        print("正在解码taxonomy特征...")
        df['taxonomy_decoded'] = df['taxonomy'].apply(decode_taxonomy_features)
    
    # 保存结果
    df.to_csv(output_csv, index=False)
    print(f"结果已保存到: {output_csv}")
    
    # 显示前几行结果
    print("\n前3行解码结果:")
    cols_to_show = ['sample_index', 'smiles']
    if 'gene_decoded' in df.columns:
        cols_to_show.append('gene_decoded')
    if 'duration_decoded' in df.columns:
        cols_to_show.append('duration_decoded')
    if 'taxonomy_decoded' in df.columns:
        cols_to_show.append('taxonomy_decoded')
    
    print(df[cols_to_show].head(3))

if __name__ == '__main__':
    # 使用示例
    input_file = '14744.csv'
    output_file = '14744_out.csv'
    
    recover_original_features(input_file, output_file)