#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV to PKL Converter for YYD Project
将CSV文件转换为与use_yyd_3.py兼容的PKL格式

Author: Generated based on CustomDataset.py and export_test_dataset_from_train_yyd.py
Date: 2024
"""

import pandas as pd
import torch
import pickle
import logging
import os
import sys
from datetime import datetime
import traceback
from torch_geometric.utils import from_smiles
from torch_geometric.data import Data
from pre_transform import GenFeatures

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_to_pkl_conversion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Debug mode flag
DEBUG_MODE = True

def debug_print(message):
    """Print debug information if debug mode is enabled"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")
        logger.debug(message)

class CSVToPKLConverter:
    """CSV到PKL转换器，与CustomDataset保持一致的数据处理逻辑"""
    
    # gene映射（与CustomDataset保持一致）
    GENE_MAP = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    # duration映射（与CustomDataset保持一致）
    DURATION_MAP = [24, 48, 72, 96]
    
    def __init__(self, mode="gene+taxonomy"):
        """
        初始化转换器
        
        Args:
            mode: 处理模式，可选 'gene', 'taxonomy', 'gene+taxonomy', 'noclass'
        """
        self.mode = mode
        self.validate_mode()
        self.pre_transform = GenFeatures()  # 使用与训练时相同的预处理
        
    def validate_mode(self):
        """验证模式参数的有效性"""
        valid_modes = ['gene', 'taxonomy', 'gene+taxonomy', 'noclass']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Mode must be one of: {valid_modes}")
    
    def _process_gene_data(self, gene_data: str) -> torch.Tensor:
        """处理基因数据，转换为张量（与CustomDataset保持一致）"""
        encoded_gene_data = [self.GENE_MAP[char] for char in gene_data]
        return torch.tensor(encoded_gene_data, dtype=torch.float).transpose(0, 1).unsqueeze(0)
    
    def process_csv_row(self, row, idx):
        """
        处理CSV文件的单行数据，转换为PyTorch Geometric Data对象
        
        Args:
            row: pandas DataFrame的一行
            idx: 行索引
            
        Returns:
            torch_geometric.data.Data: 处理后的数据对象
        """
        try:
            # 从DataFrame行中提取数据（与CustomDataset.process保持一致）
            organism = row.iloc[0]
            chemical_name = row.iloc[1] 
            duration = int(row.iloc[2])
            smiles = row.iloc[3]
            mg_perl = float(row.iloc[4])
            gene_data = row.iloc[5]
            
            # 动态确定分类数据的范围
            available_cols = len(row)
            if available_cols < 756:
                debug_print(f"Warning: Expected 756 columns but found {available_cols}")
                # 获取从第6列到最后一列的所有分类数据
                taxonomy_data = row.iloc[6:].values.astype(float)
            else:
                # 获取从第6列到第755列的分类数据（750个特征）
                taxonomy_data = row.iloc[6:756].values.astype(float)
            
            # 从SMILES生成图结构
            data = from_smiles(smiles)
            if data is None:
                logger.warning(f"Could not generate graph from SMILES for row {idx}")
                return None

            # 添加基本属性
            data.organism = organism
            data.chemical_name = chemical_name
            data.smiles = smiles  # 保存SMILES用于use_yyd_3.py
            
            # 处理duration为one-hot编码
            duration_tensor = torch.zeros(len(self.DURATION_MAP))
            if duration in self.DURATION_MAP:
                duration_tensor[self.DURATION_MAP.index(duration)] = 1.0
            data.duration = duration_tensor
            
            data.y = torch.tensor([mg_perl], dtype=torch.float)

            # 添加基因特征
            if self.mode in ['gene', 'gene+taxonomy']:
                data.gene = self._process_gene_data(gene_data)

            # 添加分类特征
            if self.mode in ['taxonomy', 'gene+taxonomy']:
                data.taxonomy = torch.tensor(taxonomy_data, dtype=torch.float)

            # 应用预转换（与训练时保持一致）
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            return data

        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            return None
    
    def convert_csv_to_pkl(self, csv_filename, pkl_filename=None, encoding='utf-8'):
        """
        将CSV文件转换为PKL格式，与use_yyd_3.py兼容
        
        Args:
            csv_filename (str): 输入CSV文件路径
            pkl_filename (str, optional): 输出PKL文件路径
            encoding (str): CSV文件编码
        
        Returns:
            str: 生成的PKL文件路径
        """
        start_time = datetime.now()
        logger.info(f"Starting CSV to PKL conversion: {csv_filename}")
        debug_print(f"Processing mode: {self.mode}")
        
        try:
            # 验证输入文件
            if not os.path.exists(csv_filename):
                raise FileNotFoundError(f"CSV file not found: {csv_filename}")
            
            file_size = os.path.getsize(csv_filename)
            debug_print(f"Input file size: {file_size / 1024:.2f} KB")
            
            # 生成输出文件名
            if pkl_filename is None:
                base_name = os.path.splitext(os.path.basename(csv_filename))[0]
                pkl_filename = f"{base_name}_converted.pkl"
            
            if not pkl_filename.endswith('.pkl'):
                pkl_filename += '.pkl'
            
            debug_print(f"Output file: {pkl_filename}")
            
            # 加载CSV文件
            logger.info(f"Loading CSV file: {csv_filename}")
            encodings_to_try = [encoding, 'utf-8', 'gbk', 'gb2312', 'latin-1']
            df = None
            
            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(csv_filename, encoding=enc)
                    logger.info(f"Successfully loaded CSV with encoding: {enc}")
                    break
                except UnicodeDecodeError:
                    debug_print(f"Failed to load with encoding: {enc}")
                    continue
            
            if df is None:
                raise ValueError(f"Could not load CSV file with any supported encoding")
            
            logger.info(f"CSV loaded successfully - Shape: {df.shape}")
            debug_print(f"Columns: {list(df.columns)}")
            
            # 处理数据
            logger.info(f"Processing data in {self.mode} mode...")
            data_list = []
            
            for idx, row in df.iterrows():
                data = self.process_csv_row(row, idx)
                if data is not None:
                    data_list.append(data)
                
                if (idx + 1) % 100 == 0:
                    debug_print(f"Processed {idx + 1}/{len(df)} rows")
            
            logger.info(f"Successfully processed {len(data_list)} samples out of {len(df)} total rows")
            
            if len(data_list) == 0:
                raise ValueError("No valid samples were processed. Please check your CSV file format.")
            
            # 保存为PKL文件
            logger.info(f"Saving data to PKL file: {pkl_filename}")
            with open(pkl_filename, 'wb') as f:
                pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 验证保存的文件
            pkl_size = os.path.getsize(pkl_filename)
            logger.info(f"PKL file saved successfully - Size: {pkl_size / 1024:.2f} KB")
            
            # 验证文件内容
            logger.info("Verifying saved PKL file...")
            with open(pkl_filename, 'rb') as f:
                loaded_data = pickle.load(f)
            
            debug_print(f"Verification: Loaded list with {len(loaded_data)} Data objects")
            if len(loaded_data) > 0:
                sample_data = loaded_data[0]
                debug_print(f"Sample data attributes: {dir(sample_data)}")
                if hasattr(sample_data, 'x'):
                    debug_print(f"Node features shape: {sample_data.x.shape}")
                if hasattr(sample_data, 'edge_index'):
                    debug_print(f"Edge index shape: {sample_data.edge_index.shape}")
            
            logger.info("PKL file verification successful")
            
            # 性能统计
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            logger.info(f"Conversion completed in {processing_time:.2f} seconds")
            
            return pkl_filename
            
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            traceback.print_exc()
            raise

def save_csv_to_pkl(csv_filename, pkl_filename=None, encoding='utf-8', mode="gene+taxonomy"):
    """
    便捷函数：将CSV文件转换为PKL格式
    
    Args:
        csv_filename (str): 输入CSV文件路径
        pkl_filename (str, optional): 输出PKL文件路径
        encoding (str): CSV文件编码
        mode (str): 处理模式
    
    Returns:
        str: 生成的PKL文件路径
    """
    converter = CSVToPKLConverter(mode=mode)
    return converter.convert_csv_to_pkl(csv_filename, pkl_filename, encoding)

def main():
    """主函数示例"""
    # 示例用法
    csv_file = "6ppd.csv"  # 替换为你的CSV文件路径
    pkl_file = "6ppd.pkl"  # 输出PKL文件路径
    
    try:
        result_file = save_csv_to_pkl(
            csv_filename=csv_file,
            pkl_filename=pkl_file,
            encoding='utf-8',
            mode="gene+taxonomy"  # 与训练时保持一致
        )
        print(f"\n转换成功！")
        print(f"输出文件: {result_file}")
        print(f"现在可以在use_yyd_3.py中使用此PKL文件进行预测")
        
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    main()