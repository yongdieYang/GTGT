import torch
import random
import numpy as np
import pickle
import pandas as pd
import os
from pre_transform import GenFeatures
from CustomDataset import ToxicDataset

# ----------------------
# 配置块
# ----------------------
SEED = 42
NUM_FOLDS = 10
DATASET_OUTPUT_DIR = 'exported_14754_datasets_complete'
os.makedirs(DATASET_OUTPUT_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

# ----------------------
# 数据准备函数
# ----------------------
def prepare_datasets(root, filename, mode, pre_transform):
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

# ----------------------
# 原始CSV加载与复合键映射
# ----------------------
def load_original_csv_data(csv_path):
    try:
        original_df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"原始CSV文件加载成功，包含 {len(original_df)} 行数据")
        print(f"原始CSV文件列名: {list(original_df.columns)}")
        return original_df
    except Exception as e:
        print(f"加载原始CSV文件失败: {e}")
        return None

def create_multi_key_mapping(original_df):
    if original_df is None:
        return {}

    mapping = {}
    for idx, row in original_df.iterrows():
        key = (
            str(row.get('organism', '')).strip(),
            str(row.get('smiles', '')).strip(),
            str(row.get('gene', '')).strip(),
            str(row.get('duration', '')).strip()
        )
        if key not in mapping:
            mapping[key] = row.to_dict()

    print(f"创建了 {len(mapping)} 个复合键映射")
    return mapping

# ----------------------
# 数据导出函数
# ----------------------
def export_dataset_to_csv_complete(dataset, csv_path, dataset_name, multi_key_mapping):
    data_list = []
    for i, data in enumerate(dataset):
        organism = getattr(data, 'organism', '')
        smiles = getattr(data, 'smiles', '')
        gene = str(data.gene.item()) if hasattr(data, 'gene') and data.gene.numel() == 1 else ''
        duration = str(data.duration.item()) if hasattr(data, 'duration') and data.duration.numel() == 1 else ''

        key = (organism, smiles, gene, duration)
        original_data = multi_key_mapping.get(key, {})

        row = {
            'dataset_index': i,
            'dataset_type': dataset_name,
            'processed_organism': organism,
            'processed_smiles': smiles,
            'processed_gene': gene,
            'processed_duration': duration,
            'processed_y': data.y.item() if data.y.numel() == 1 else str(data.y.tolist())
        }

        for col, value in original_data.items():
            row[f'original_{col}'] = value

        if hasattr(data, 'x') and data.x is not None:
            row['graph_num_nodes'] = data.x.shape[0]
            row['graph_node_features_dim'] = data.x.shape[1]

        if hasattr(data, 'edge_index') and data.edge_index is not None:
            row['graph_num_edges'] = data.edge_index.shape[1]

        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            row['graph_edge_features_dim'] = data.edge_attr.shape[1] if len(data.edge_attr.shape) > 1 else 1

        data_list.append(row)

    df = pd.DataFrame(data_list)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"{dataset_name}已导出到CSV: {csv_path}")
    print(f"导出的CSV包含 {len(df.columns)} 列，{len(df)} 行数据")

def export_dataset_files_complete(dataset, dataset_name, mapping, use_14755_naming=False):
    base_name = f'14755_{dataset_name}' if use_14755_naming else dataset_name

    pkl_path = os.path.join(DATASET_OUTPUT_DIR, f'{base_name}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"{dataset_name}已导出到PKL: {pkl_path}")

    csv_path = os.path.join(DATASET_OUTPUT_DIR, f'{base_name}_complete.csv')
    export_dataset_to_csv_complete(dataset, csv_path, dataset_name, mapping)
    return pkl_path, csv_path

def export_combined_dataset_to_csv(train_dataset, val_dataset, test_dataset, mapping):
    all_data = []
    for dataset, dataset_type in [(train_dataset, 'train'), (val_dataset, 'val'), (test_dataset, 'test')]:
        for i, data in enumerate(dataset):
            organism = getattr(data, 'organism', '')
            smiles = getattr(data, 'smiles', '')
            gene = str(data.gene.item()) if hasattr(data, 'gene') and data.gene.numel() == 1 else ''
            duration = str(data.duration.item()) if hasattr(data, 'duration') and data.duration.numel() == 1 else ''

            key = (organism, smiles, gene, duration)
            original_data = mapping.get(key, {})

            row = {
                'dataset_type': dataset_type,
                'dataset_index': i,
                'processed_organism': organism,
                'processed_smiles': smiles,
                'processed_gene': gene,
                'processed_duration': duration,
                'processed_y': data.y.item() if data.y.numel() == 1 else str(data.y.tolist())
            }

            for col, value in original_data.items():
                row[f'original_{col}'] = value

            if hasattr(data, 'x') and data.x is not None:
                row['graph_num_nodes'] = data.x.shape[0]
                row['graph_node_features_dim'] = data.x.shape[1]

            if hasattr(data, 'edge_index') and data.edge_index is not None:
                row['graph_num_edges'] = data.edge_index.shape[1]

            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                row['graph_edge_features_dim'] = data.edge_attr.shape[1] if len(data.edge_attr.shape) > 1 else 1

            all_data.append(row)

    df = pd.DataFrame(all_data)
    csv_path = os.path.join(DATASET_OUTPUT_DIR, '14754_complete.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"合并数据集导出至: {csv_path}, 共 {len(df)} 行")
    return csv_path

# ----------------------
# 主函数
# ----------------------
def main():
    print("开始处理...")
    original_csv_path = os.path.join('raw', 'yyd_data_train.csv')
    original_df = load_original_csv_data(original_csv_path)
    mapping = create_multi_key_mapping(original_df)

    train_dataset, val_dataset, test_dataset = prepare_datasets(
        root='.',
        filename='yyd_data_train.csv',
        mode="gene+taxonomy",
        pre_transform=GenFeatures()
    )

    print("\n导出训练集...")
    train_pkl, train_csv = export_dataset_files_complete(train_dataset, 'train', mapping, use_14755_naming=True)

    print("\n导出验证集...")
    val_pkl, val_csv = export_dataset_files_complete(val_dataset, 'val', mapping, use_14755_naming=True)

    print("\n导出测试集...")
    test_pkl, test_csv = export_dataset_files_complete(test_dataset, 'test', mapping, use_14755_naming=True)

    print("\n导出合并数据集...")
    combined_csv = export_combined_dataset_to_csv(train_dataset, val_dataset, test_dataset, mapping)

    print("\n导出完成。")
    print(f"输出目录: {DATASET_OUTPUT_DIR}")
    print(f"训练集: {train_pkl}, {train_csv}")
    print(f"验证集: {val_pkl}, {val_csv}")
    print(f"测试集: {test_pkl}, {test_csv}")
    print(f"合并集: {combined_csv}")

if __name__ == "__main__":
    main()
