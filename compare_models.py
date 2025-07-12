import torch
import numpy as np
from collections import OrderedDict
import os

def compare_model_files(model_path1, model_path2, tolerance=1e-8):
    """
    对比两个PyTorch模型文件是否完全一致
    
    Args:
        model_path1 (str): 第一个模型文件路径
        model_path2 (str): 第二个模型文件路径
        tolerance (float): 浮点数比较的容差值
    
    Returns:
        dict: 包含详细比较结果的字典
    """
    
    # 检查文件是否存在
    if not os.path.exists(model_path1):
        return {"error": f"模型文件不存在: {model_path1}"}
    if not os.path.exists(model_path2):
        return {"error": f"模型文件不存在: {model_path2}"}
    
    try:
        # 加载模型
        print(f"正在加载模型: {model_path1}")
        model1 = torch.load(model_path1, map_location='cpu')
        print(f"正在加载模型: {model_path2}")
        model2 = torch.load(model_path2, map_location='cpu')
        
        result = {
            "files_identical": True,
            "details": {},
            "differences": []
        }
        
        # 1. 比较文件大小
        size1 = os.path.getsize(model_path1)
        size2 = os.path.getsize(model_path2)
        result["details"]["file_sizes"] = {"model1": size1, "model2": size2}
        
        if size1 != size2:
            result["files_identical"] = False
            result["differences"].append(f"文件大小不同: {size1} vs {size2} bytes")
        
        # 2. 比较顶层键
        keys1 = set(model1.keys()) if isinstance(model1, dict) else set()
        keys2 = set(model2.keys()) if isinstance(model2, dict) else set()
        
        result["details"]["keys"] = {
            "model1_keys": sorted(list(keys1)),
            "model2_keys": sorted(list(keys2))
        }
        
        if keys1 != keys2:
            result["files_identical"] = False
            missing_in_2 = keys1 - keys2
            missing_in_1 = keys2 - keys1
            if missing_in_2:
                result["differences"].append(f"模型2中缺少的键: {missing_in_2}")
            if missing_in_1:
                result["differences"].append(f"模型1中缺少的键: {missing_in_1}")
        
        # 3. 比较共同的键
        common_keys = keys1 & keys2
        for key in common_keys:
            print(f"正在比较键: {key}")
            
            val1 = model1[key]
            val2 = model2[key]
            
            # 比较张量
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                if not compare_tensors(val1, val2, tolerance):
                    result["files_identical"] = False
                    result["differences"].append(f"张量 '{key}' 不同")
                    result["details"][f"{key}_shape"] = {"model1": val1.shape, "model2": val2.shape}
                    if val1.shape == val2.shape:
                        diff = torch.abs(val1 - val2)
                        max_diff = torch.max(diff).item()
                        result["details"][f"{key}_max_diff"] = max_diff
            
            # 比较字典（如state_dict）
            elif isinstance(val1, dict) and isinstance(val2, dict):
                dict_result = compare_state_dicts(val1, val2, tolerance)
                if not dict_result["identical"]:
                    result["files_identical"] = False
                    result["differences"].extend([f"在 '{key}' 中: {diff}" for diff in dict_result["differences"]])
                    result["details"][f"{key}_comparison"] = dict_result
            
            # 比较其他类型
            else:
                if val1 != val2:
                    result["files_identical"] = False
                    result["differences"].append(f"键 '{key}' 的值不同: {val1} vs {val2}")
        
        return result
        
    except Exception as e:
        return {"error": f"加载或比较模型时出错: {str(e)}"}

def compare_tensors(tensor1, tensor2, tolerance=1e-8):
    """
    比较两个张量是否相等
    """
    if tensor1.shape != tensor2.shape:
        return False
    
    if tensor1.dtype != tensor2.dtype:
        return False
    
    # 使用torch.allclose进行数值比较
    return torch.allclose(tensor1, tensor2, atol=tolerance, rtol=tolerance)

def compare_state_dicts(state_dict1, state_dict2, tolerance=1e-8):
    """
    比较两个state_dict
    """
    result = {
        "identical": True,
        "differences": [],
        "parameter_count": {"model1": len(state_dict1), "model2": len(state_dict2)}
    }
    
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    if keys1 != keys2:
        result["identical"] = False
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        if missing_in_2:
            result["differences"].append(f"参数缺失在模型2: {missing_in_2}")
        if missing_in_1:
            result["differences"].append(f"参数缺失在模型1: {missing_in_1}")
    
    # 比较共同参数
    common_keys = keys1 & keys2
    total_params = 0
    different_params = 0
    
    for key in common_keys:
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        if isinstance(param1, torch.Tensor) and isinstance(param2, torch.Tensor):
            total_params += param1.numel()
            if not compare_tensors(param1, param2, tolerance):
                result["identical"] = False
                different_params += param1.numel()
                result["differences"].append(f"参数 '{key}' 不同 (shape: {param1.shape})")
        else:
            if param1 != param2:
                result["identical"] = False
                result["differences"].append(f"非张量参数 '{key}' 不同")
    
    result["parameter_stats"] = {
        "total_parameters": total_params,
        "different_parameters": different_params,
        "similarity_percentage": (total_params - different_params) / total_params * 100 if total_params > 0 else 100
    }
    
    return result

def print_comparison_report(result):
    """
    打印详细的比较报告
    """
    if "error" in result:
        print(f"❌ 错误: {result['error']}")
        return
    
    print("\n" + "="*60)
    print("           PyTorch 模型比较报告")
    print("="*60)
    
    if result["files_identical"]:
        print("✅ 结果: 两个模型文件完全一致!")
    else:
        print("❌ 结果: 两个模型文件存在差异!")
    
    print(f"\n📊 文件大小:")
    sizes = result["details"].get("file_sizes", {})
    print(f"   模型1: {sizes.get('model1', 'N/A')} bytes")
    print(f"   模型2: {sizes.get('model2', 'N/A')} bytes")
    
    if "keys" in result["details"]:
        keys_info = result["details"]["keys"]
        print(f"\n🔑 顶层键:")
        print(f"   模型1: {keys_info['model1_keys']}")
        print(f"   模型2: {keys_info['model2_keys']}")
    
    if result["differences"]:
        print(f"\n⚠️  发现的差异 ({len(result['differences'])} 项):")
        for i, diff in enumerate(result["differences"], 1):
            print(f"   {i}. {diff}")
    
    # 显示参数统计信息
    for key, value in result["details"].items():
        if key.endswith("_comparison") and isinstance(value, dict):
            if "parameter_stats" in value:
                stats = value["parameter_stats"]
                print(f"\n📈 {key.replace('_comparison', '')} 参数统计:")
                print(f"   总参数数: {stats['total_parameters']:,}")
                print(f"   不同参数数: {stats['different_parameters']:,}")
                print(f"   相似度: {stats['similarity_percentage']:.2f}%")
    
    print("\n" + "="*60)

def main():
    """
    主函数 - 示例用法
    """
    # 示例：比较两个模型文件
    model1_path = "final_best_model_clean.pth"
    model2_path = "model_186.pth"
    
    print("PyTorch 模型比较工具")
    print(f"比较模型: {model1_path} vs {model2_path}")
    
    # 执行比较
    result = compare_model_files(model1_path, model2_path, tolerance=1e-8)
    
    # 打印报告
    print_comparison_report(result)
    
    # 可选：保存详细结果到JSON文件
    import json
    with open("model_comparison_result.json", "w", encoding="utf-8") as f:
        # 处理不能序列化的对象
        serializable_result = {}
        for k, v in result.items():
            if isinstance(v, (str, int, float, bool, list)):
                serializable_result[k] = v
            elif isinstance(v, dict):
                serializable_result[k] = {}
                for k2, v2 in v.items():
                    if isinstance(v2, (str, int, float, bool, list, dict)):
                        serializable_result[k][k2] = v2
                    else:
                        serializable_result[k][k2] = str(v2)
            else:
                serializable_result[k] = str(v)
        
        json.dump(serializable_result, f, ensure_ascii=False, indent=2)
    
    print("\n详细结果已保存到: model_comparison_result.json")

if __name__ == "__main__":
    main()