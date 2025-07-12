import torch
import pandas as pd
import pickle

MODEL_PATH = ' '    
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    """Load the complete model file"""
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model = checkpoint['model']
        print(f"Model information:")
        print(f"  Trial number: {checkpoint.get('trial_number', 'N/A')}")
        print(f"  Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
        if 'best_metrics' in checkpoint:
            metrics = checkpoint['best_metrics']
            print(f"  Test RMSE: {metrics.get('test_rmse', 'N/A'):.6f}")
            print(f"  Test R²: {metrics.get('test_r2', 'N/A'):.6f}")
    else:
        model = checkpoint
    
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    return model

def predict(model, data_input):
    """Use the loaded model for prediction"""
    data_input = data_input.to(DEVICE)
    with torch.no_grad():
        batch = torch.zeros(data_input.x.size(0), dtype=torch.long, device=DEVICE)
        
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

if __name__ == '__main__':
    loaded_model = load_model(MODEL_PATH)
    input_file = ''
    output_file = ''
    
    try:
        with open(input_file, 'rb') as f:
            test_dataset = pickle.load(f)
        print(f"\nSuccessfully loaded dataset: {input_file}, total {len(test_dataset)} samples")
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        exit()
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        exit()
    
    # 3. Make predictions
    print("\nStarting prediction...")
    predictions = []
    smiles_list = []
    
    for i, data in enumerate(test_dataset):
        try:
            prediction_result = predict(loaded_model, data)
            predictions.append(prediction_result)
            
            smiles = data.smiles if hasattr(data, 'smiles') else f"sample_{i}"
            smiles_list.append(smiles)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Completed prediction for {i+1}/{len(test_dataset)} samples")
                
        except Exception as e:
            print(f"Error predicting sample {i}: {e}")
            predictions.append(None)
            smiles_list.append(f"sample_{i}_error")
    
    # 4. Save results
    results_df = pd.DataFrame({
        'smiles': smiles_list,
        'predictions': predictions
    })
    
    try:
        results_df.to_csv(output_file, index=False)
        print(f"\nPrediction results saved to: {output_file}")
        print(f"Total samples: {len(predictions)}")
        print(f"Successful predictions: {sum(1 for p in predictions if p is not None)}")
        
        print("\nFirst 5 prediction results:")
        for i in range(min(5, len(predictions))):
            if predictions[i] is not None:
                print(f"  Sample {i+1}: {predictions[i]:.6f}")
                
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\nPrediction completed!")