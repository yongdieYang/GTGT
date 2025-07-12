import torch
import random
import numpy as np
import pickle
import pandas as pd
import os
from pre_transform import GenFeatures
from CustomDataset import ToxicDataset

# ----------------------
# 配置块（与 train_yyd_modified.py 完全相同）
# ----------------------
SEED = 42
NUM_FOLDS = 10

# 创建输出目录
DATASET_OUTPUT_DIR = 'exported_datasets-1'
os.makedirs(DATASET_OUTPUT_DIR, exist_ok=True)

# 固定随机种子（与 train_yyd_modified.py 完全相同）
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

# ----------------------
# 数据集准备（与 train_yyd_modified.py 完全相同）
# ----------------------
def prepare_datasets(root, filename, mode, pre_transform):
    """与 train_yyd_modified.py 完全相同的数据集准备函数"""
    dataset = ToxicDataset(
        root=root,
        filenames=filename,
        mode=mode,
        pre_transform=pre_transform
    ).shuffle()
    
    n = len(dataset) // NUM_FOLDS
    val_dataset = dataset[:n]
    test_dataset = dataset[n:2 * n]
    train_dataset = dataset[2 * n:]
    
    return train_dataset, val_dataset, test_dataset

def export_dataset_to_csv(dataset, csv_path, dataset_name):
    """将数据集导出为CSV文件"""
    data_list = []
    
    for i, data in enumerate(dataset):
        # 提取基本信息
        row = {
            'index': i,
            'smiles': data.smiles if hasattr(data, 'smiles') else '',
            'gene': data.gene.item() if hasattr(data, 'gene') and data.gene.numel() == 1 else str(data.gene.tolist()) if hasattr(data, 'gene') else '',
            'taxonomy': data.taxonomy.item() if hasattr(data, 'taxonomy') and data.taxonomy.numel() == 1 else str(data.taxonomy.tolist()) if hasattr(data, 'taxonomy') else '',
            'duration': data.duration.item() if hasattr(data, 'duration') and data.duration.numel() == 1 else str(data.duration.tolist()) if hasattr(data, 'duration') else '',
            'y': data.y.item() if data.y.numel() == 1 else str(data.y.tolist())
        }
        
        # 添加节点特征信息（可选）
        if hasattr(data, 'x') and data.x is not None:
            row['num_nodes'] = data.x.shape[0]
            row['node_features_dim'] = data.x.shape[1]
        
        # 添加边信息（可选）
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            row['num_edges'] = data.edge_index.shape[1]
        
        data_list.append(row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data_list)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"{dataset_name}已导出到CSV: {csv_path}")

def export_dataset_files(dataset, dataset_name):
    """导出单个数据集的pkl和csv文件"""
    # 导出PKL文件
    pkl_path = os.path.join(DATASET_OUTPUT_DIR, f'{dataset_name}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"{dataset_name}已导出到PKL: {pkl_path}")
    
    # 导出CSV文件
    csv_path = os.path.join(DATASET_OUTPUT_DIR, f'{dataset_name}.csv')
    export_dataset_to_csv(dataset, csv_path, dataset_name)
    
    return pkl_path, csv_path

def main():
    print("开始导出与 train_yyd_modified.py 完全相同的数据集...")
    
    # 使用与 train_yyd_modified.py 完全相同的数据集准备逻辑
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        root='.',
        filename='yyd_data_train.csv',
        mode="gene+taxonomy",
        pre_transform=GenFeatures()
    )
    
    print(f"\n=== 开始导出数据集 ===")
    
    # 导出训练集
    print("\n导出训练集...")
    train_pkl, train_csv = export_dataset_files(train_dataset, 'train')
    
    # 导出验证集
    print("\n导出验证集...")
    val_pkl, val_csv = export_dataset_files(val_dataset, 'val')
    
    # 导出测试集
    print("\n导出测试集...")
    test_pkl, test_csv = export_dataset_files(test_dataset, 'test')
    
    print(f"\n=== 导出完成 ===")
    print(f"输出目录: {DATASET_OUTPUT_DIR}")
    print(f"\n文件列表:")
    print(f"训练集 - PKL: {train_pkl}")
    print(f"训练集 - CSV: {train_csv}")
    print(f"验证集 - PKL: {val_pkl}")
    print(f"验证集 - CSV: {val_csv}")
    print(f"测试集 - PKL: {test_pkl}")
    print(f"测试集 - CSV: {test_csv}")
    
    print(f"\n=== 数据集统计 ===")
    print(f"训练集大小: {len(train_dataset)} 个样本")
    print(f"验证集大小: {len(val_dataset)} 个样本")
    print(f"测试集大小: {len(test_dataset)} 个样本")
    
    # 验证数据集划分
    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(f"总样本数: {total_samples}")
    print(f"训练集比例: {len(train_dataset)/total_samples:.2%}")
    print(f"验证集比例: {len(val_dataset)/total_samples:.2%}")
    print(f"测试集比例: {len(test_dataset)/total_samples:.2%}")

if __name__ == "__main__":
    main()