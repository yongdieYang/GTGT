import re
import math
import os  # 添加缺失的import
from typing import List, Optional, Callable, Dict, Any

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_smiles


class ToxicDataset(InMemoryDataset):
    """处理毒性数据的数据集类，支持多种模式的特征处理"""

    # gene映射
    GENE_MAP = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    # 添加 DURATION_MAP
    DURATION_MAP = [24, 48, 72, 96]  # 根据你的实际数据调整这些值

    def __init__(
            self,
            root: str,
            filenames: str,
            mode: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
    ):
        """
        初始化毒性数据集

        Args:
            root: 数据根目录
            filenames: 原始文件名
            mode: 处理模式，可选 'gene', 'taxonomy', 'gene+taxonomy', 'noclass'
            transform: 数据转换函数
            pre_transform: 数据预转换函数
            pre_filter: 数据预过滤函数
        """
        self.filenames = filenames
        self.mode = mode
        self.validate_mode()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def validate_mode(self):
        """验证模式参数的有效性"""
        valid_modes = ['gene', 'taxonomy', 'gene+taxonomy', 'noclass']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Mode must be one of: {valid_modes}")

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, f'{self.mode}_processed')

    @property
    def raw_file_names(self) -> str:
        """原始文件的文件名，如果存在则不会触发download"""
        return self.filenames

    @property
    def processed_file_names(self) -> str:
        """处理后的文件名，如果在 processed_dir 中找到则跳过 process"""
        return 'data.pt'

    def process(self):
        import pandas as pd
        from torch_geometric.utils import from_smiles
        
        print("Processing dataset in {} mode...".format(self.mode))
        data_list = []
        
        # 使用pandas读取CSV文件 - 修复：使用 self.raw_paths[0] 而不是 self.raw_file_paths[0]
        df = pd.read_csv(self.raw_paths[0])
        print(f"CSV file has {len(df.columns)} columns and {len(df)} rows")
        
        for idx, row in df.iterrows():
            try:
                # 直接从DataFrame行中提取数据
                organism = row.iloc[0]
                chemical_name = row.iloc[1] 
                duration = int(row.iloc[2])
                smiles = row.iloc[3]
                mg_perl = float(row.iloc[4])
                gene_data = row.iloc[5]
                
                # 动态确定分类数据的范围
                available_cols = len(row)
                if available_cols < 756:
                    print(f"Warning: Expected 756 columns but found {available_cols}")
                    # 获取从第6列到最后一列的所有分类数据
                    taxonomy_data = row.iloc[6:].values.astype(float)
                else:
                    # 获取从第6列到第755列的分类数据（750个特征）
                    taxonomy_data = row.iloc[6:756].values.astype(float)
                
                # 从SMILES生成图结构
                data = from_smiles(smiles)
                if data is None:
                    print(f"Warning: Could not generate graph from SMILES for row {idx}")
                    continue

                # 添加基本属性
                data.organism = organism
                data.chemical_name = chemical_name
                
                # 处理duration为one-hot编码
                duration_tensor = torch.zeros(len(self.DURATION_MAP))
                if duration in self.DURATION_MAP:
                    duration_tensor[self.DURATION_MAP.index(duration)] = 1.0  # 修复：使用 index() 而不是直接索引
                data.duration = duration_tensor
                
                data.y = torch.tensor([mg_perl], dtype=torch.float)

                # 添加基因特征
                if self.mode in ['gene', 'gene+taxonomy']:
                    data.gene = self._process_gene_data(gene_data)

                # 添加分类特征
                if self.mode in ['taxonomy', 'gene+taxonomy']:
                    data.taxonomy = torch.tensor(taxonomy_data, dtype=torch.float)

                # 应用预过滤和预转换
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        print(f"Successfully processed {len(data_list)} samples")
        
        if len(data_list) == 0:
            raise ValueError("No valid samples were processed. Please check your CSV file format.")
        
        torch.save(self.collate(data_list), self.processed_paths[0])

    def _process_gene_data(self, gene_data: str) -> torch.Tensor:
        """处理基因数据，转换为张量"""
        encoded_gene_data = [self.GENE_MAP[char] for char in gene_data]
        return torch.tensor(encoded_gene_data, dtype=torch.float).transpose(0, 1).unsqueeze(0)