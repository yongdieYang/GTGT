import pandas as pd
import torch
import json
import pickle # 新增导入 pickle 模块
import random  # 添加这行来修复 NameError
import numpy as np

# 设置随机种子，与train_yyd_modified.py保持一致
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 在main部分开始前设置种子
SEED = 42
set_seed(SEED)

def save_dataset_to_pkl(dataset, filename):
    """
    Saves the entire dataset object to a .pkl file.

    Args:
        dataset: The dataset object to save (e.g., a PyTorch Dataset instance).
        filename: The name of the .pkl file to save the dataset to.
                  It's good practice to ensure filename ends with '.pkl'.
    """
    # 确保文件名以 .pkl 结尾，如果用户没有提供，则添加
    if not filename.endswith('.pkl'):
        filename += '.pkl'
        
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {filename}")


def save_dataset_to_csv(dataset, filename):
    data_to_save = []
    for data_item in dataset:
        # 从 CustomDataset.py 的 _extract_base_features 和 process 方法中获取原始SMILES等信息的方式可能需要调整
        # 这里我们假设 data_item 直接包含SMILES字符串，或者可以从某个属性获取
        # 对于 gene, taxonomy, duration, y，它们是tensor，需要转换
        smiles = data_item.smiles if hasattr(data_item, 'smiles') else "N/A" # 需要确认SMILES的存储方式
        
        gene_features = "N/A"
        if hasattr(data_item, 'gene') and data_item.gene is not None:
            gene_features = data_item.gene.tolist()

        taxonomy_features = "N/A"
        if hasattr(data_item, 'taxonomy') and data_item.taxonomy is not None:
            taxonomy_features = data_item.taxonomy.tolist()
            
        duration_features = "N/A"
        if hasattr(data_item, 'duration') and data_item.duration is not None:
            # duration 是 one-hot tensor, e.g., [0,1,0,0] for 48
            # 我们可以找到值为1的索引，然后映射回原始值
            duration_idx = torch.argmax(data_item.duration).item()
            if duration_idx < len(ToxicDataset.DURATION_MAP):
                 duration_features = ToxicDataset.DURATION_MAP[duration_idx]
            else:
                duration_features = "Invalid_duration_idx"

        target_value = "N/A"
        if hasattr(data_item, 'y') and data_item.y is not None:
            target_value = data_item.y.item() # y 是 [[value]]

        data_to_save.append({
            'SMILES': smiles,
            'gene': json.dumps(gene_features),  # <--- 修改这里: 使用 json.dumps()
            'taxonomy': json.dumps(taxonomy_features), # <--- 修改这里: 使用 json.dumps()
            'duration': duration_features,
            'target': target_value
        })
    
    df = pd.DataFrame(data_to_save)
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")


if __name__ == '__main__':
    # 从你的训练脚本中导入必要的类
    from CustomDataset import ToxicDataset # 确保 CustomDataset.py 在PYTHONPATH中
    from pre_transform import GenFeatures   # 确保 pre_transform.py 在PYTHONPATH中

    # 数据集准备参数 (与 train_yyd.py 类似)
    root_dir = '.'  # 或者您的数据根目录
    csv_filename = 'yyd_data_train.csv' # 与 train_yyd.py 中使用的文件名一致
    mode = "gene+taxonomy" # 与 train_yyd.py 中使用的模式一致
    num_folds = 10 # 与 train_yyd.py 中定义的一致

    print(f"Loading dataset from {csv_filename} with mode '{mode}'...")
    try:
        # 加载完整数据集
        full_dataset = ToxicDataset(
            root=root_dir,
            filenames=csv_filename,
            mode=mode,
            pre_transform=GenFeatures()
        ).shuffle() # 与训练脚本一致，先打乱

        # 划分数据集 (与 train_yyd.py 中的 prepare_datasets 逻辑一致)
        n = len(full_dataset) // num_folds
        val_dataset = full_dataset[:n]
        test_dataset = full_dataset[n:2 * n]
        train_dataset = full_dataset[2 * n:]

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        # 定义输出文件名
        train_pkl_filename = 'exported_train_dataset.pkl'
        val_pkl_filename = 'exported_val_dataset.pkl'
        test_pkl_filename = 'exported_test_dataset.pkl'
        
        # 定义 CSV 输出文件名
        train_csv_filename = 'exported_train_dataset.csv'
        val_csv_filename = 'exported_val_dataset.csv'
        test_csv_filename = 'exported_test_dataset.csv'

        # 保存数据集为 PKL 文件
        print(f"\nSaving datasets to PKL files...")
        save_dataset_to_pkl(train_dataset, train_pkl_filename)
        save_dataset_to_pkl(val_dataset, val_pkl_filename)
        save_dataset_to_pkl(test_dataset, test_pkl_filename)
        
        # 保存数据集为 CSV 文件
        print(f"\nSaving datasets to CSV files...")
        save_dataset_to_csv(train_dataset, train_csv_filename)
        save_dataset_to_csv(val_dataset, val_csv_filename)
        save_dataset_to_csv(test_dataset, test_csv_filename)

        print("\nDatasets exported successfully to both PKL and CSV formats.")

        # (可选) 示例：加载保存的 PKL 文件进行验证
        print(f"\nExample of loading the PKL file: {train_pkl_filename}")
        try:
            with open(train_pkl_filename, 'rb') as f:
                loaded_train_dataset = pickle.load(f)
            print(f"Successfully loaded dataset from {train_pkl_filename}")
            if len(loaded_train_dataset) > 0:
                # 可以打印一些样本信息来验证
                sample_item = loaded_train_dataset[0]
                print(f"First item from loaded train dataset: {sample_item}")
            else:
                print("Loaded train dataset is empty.")
        except Exception as e:
            print(f"Error loading PKL file {train_pkl_filename}: {e}")

    except FileNotFoundError:
        print(f"Error: The CSV file {csv_filename} was not found in root '{root_dir}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred during dataset processing or export: {e}")
        import traceback
        traceback.print_exc()