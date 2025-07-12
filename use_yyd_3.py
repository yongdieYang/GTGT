import torch
import pandas as pd
import pickle
from torch_geometric.data import Data

# --- 配置 --- #
MODEL_PATH = 'final_best_model_clean.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 简化的模型加载函数 --- #
def load_model(model_path):
    """直接加载完整的模型文件"""
    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # 如果是完整的模型信息字典
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model = checkpoint['model']
        print(f"模型信息:")
        print(f"  试验编号: {checkpoint.get('trial_number', 'N/A')}")
        print(f"  最佳轮数: {checkpoint.get('best_epoch', 'N/A')}")
        if 'best_metrics' in checkpoint:
            metrics = checkpoint['best_metrics']
            print(f"  测试RMSE: {metrics.get('test_rmse', 'N/A'):.6f}")
            print(f"  测试R²: {metrics.get('test_r2', 'N/A'):.6f}")
    else:
        # 兼容旧格式
        model = checkpoint
    
    model = model.to(DEVICE)
    model.eval()
    print("模型加载成功！")
    return model

# --- 预测函数 --- #
def predict(model, data_input):
    """使用加载的模型进行预测"""
    data_input = data_input.to(DEVICE)
    with torch.no_grad():
        batch = torch.zeros(data_input.x.size(0), dtype=torch.long, device=DEVICE)
        
        # 确保gene、taxonomy和duration张量具有正确的维度
        gene = data_input.gene
        if gene.dim() == 1:
            gene = gene.unsqueeze(0)
        
        taxonomy = data_input.taxonomy
        if taxonomy.dim() == 1:
            taxonomy = taxonomy.unsqueeze(0)
        
        duration = data_input.duration
        if duration.dim() == 1:
            duration = duration.unsqueeze(0)
        
        prediction = model(
            data_input.x,
            data_input.edge_index,
            data_input.edge_attr,
            batch,
            gene,
            taxonomy,
            duration
        )
    return prediction.item()

# --- 主程序 --- #
if __name__ == '__main__':
    # 1. 直接加载模型 - 只需要一个参数！
    loaded_model = load_model(MODEL_PATH)
    
    # 2. 读取PKL文件
    input_file = '6ppd.pkl'
    output_file = 'result/6ppd.csv'
    
    try:
        with open(input_file, 'rb') as f:
            test_dataset = pickle.load(f)
        print(f"\n成功加载数据集: {input_file}，共 {len(test_dataset)} 个样本")
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        exit()
    except Exception as e:
        print(f"读取文件错误 {input_file}: {e}")
        exit()
    
    # 3. 进行预测
    print("\n开始预测...")
    predictions = []
    smiles_list = []
    
    for i, data in enumerate(test_dataset):
        try:
            # 直接使用PKL文件中的数据进行预测
            prediction_result = predict(loaded_model, data)
            predictions.append(prediction_result)
            
            # 提取SMILES用于结果保存
            smiles = data.smiles if hasattr(data, 'smiles') else f"sample_{i}"
            smiles_list.append(smiles)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"已完成 {i+1}/{len(test_dataset)} 个样本的预测")
                
        except Exception as e:
            print(f"样本 {i} 预测出错: {e}")
            predictions.append(None)
            smiles_list.append(f"sample_{i}_error")
    
    # 4. 保存结果
    results_df = pd.DataFrame({
        'smiles': smiles_list,
        'predictions': predictions
    })
    
    try:
        results_df.to_csv(output_file, index=False)
        print(f"\n预测结果已保存到: {output_file}")
        print(f"总样本数: {len(predictions)}")
        print(f"成功预测数: {sum(1 for p in predictions if p is not None)}")
        
        # 显示前几个预测结果
        print("\n前5个预测结果:")
        for i in range(min(5, len(predictions))):
            if predictions[i] is not None:
                print(f"  样本 {i+1}: {predictions[i]:.6f}")
                
    except Exception as e:
        print(f"保存结果出错: {e}")
    
    print("\n预测完成！")