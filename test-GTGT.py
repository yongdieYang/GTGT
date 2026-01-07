import torch
import random
import numpy as np
import os
import copy
from torch_geometric.loader import DataLoader 
import torch.nn.functional as F
import optuna
import wandb
from pre_transform import GenFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from CustomDataset_gene import ToxicDataset
from CMOS import GATGENETAXONOMY
import pandas as pd
from tqdm import tqdm

# ----------------------
# Configuration Block
# Please adjust the following parameters according to your actual situation.
# ----------------------
SEED = X
NUM_FOLDS = 10
MAX_EPOCHS = X
PATIENCE = X
BEST_MODEL_PATH = 'XXXX.pth'
PROJECT_NAME = 'XXXX'
N_TRIALS_OPTUNA = X
FINAL_PATIENCE = X
IN_CHANNELS = X 
EDGE_DIM = X
TAXONOMY_DIM = X
DATA_FILENAME = 'data.csv'
DATA_MODE = "gene+taxonomy"


# Create directories for saving models and hyperparameters
MODELS_DIR = f'saved_models_{PROJECT_NAME}'
os.makedirs(MODELS_DIR, exist_ok=True)

# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables
_cached_datasets = None

def prepare_datasets(root, filename, mode, pre_transform):
    global _cached_datasets
    
    if _cached_datasets is not None:
        return _cached_datasets
    
    dataset = ToxicDataset(
        root=root,
        filenames=filename,
        mode=mode,
        pre_transform=pre_transform
    )
    
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    dataset = dataset.shuffle()
    
    n = len(dataset) // NUM_FOLDS
    val_dataset = dataset[:n]
    test_dataset = dataset[n:2 * n]
    train_dataset = dataset[2 * n:]
    
    _cached_datasets = (train_dataset, val_dataset, test_dataset)
    return _cached_datasets

# Data export function
def export_datasets_to_csv(train_dataset, val_dataset, test_dataset, prefix="", output_dir="."):
    """Simplified dataset export function - ensures consistency with training data"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    def _decode_duration(duration_tensor):
        """Restore duration value from one-hot encoding"""
        if duration_tensor is None or duration_tensor.sum() == 0:
            return None
        duration_map = [24, 48, 72, 96]
        idx = duration_tensor.argmax().item()
        return duration_map[idx] if idx < len(duration_map) else None
    
    def extract_basic_info(dataset, dataset_name):
        """Extract basic information, ensuring consistency with the data structure during training"""
        data_list = []
        
        print(f"Exporting {dataset_name}...")
        for i, data in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
            row = {
                'sample_id': i,
                'y_value': data.y.item() if hasattr(data, 'y') else None,
                'smiles': getattr(data, 'smiles', 'N/A'),
                'duration': _decode_duration(data.duration) if hasattr(data, 'duration') else None,
                'gene_shape': tuple(data.gene.shape) if hasattr(data, 'gene') else None,
                'taxonomy_shape': tuple(data.taxonomy.shape) if hasattr(data, 'taxonomy') else None,
                'x_shape': tuple(data.x.shape) if hasattr(data, 'x') else None,
                'edge_index_shape': tuple(data.edge_index.shape) if hasattr(data, 'edge_index') else None,
                'edge_attr_shape': tuple(data.edge_attr.shape) if hasattr(data, 'edge_attr') else None
            }
            data_list.append(row)
        
        return pd.DataFrame(data_list)
    
    # Export each dataset
    for dataset, name in [(train_dataset, 'train'), (val_dataset, 'val'), (test_dataset, 'test')]:
        df = extract_basic_info(dataset, name)
        output_path = os.path.join(output_dir, f'{prefix}{name}_dataset.csv')
        df.to_csv(output_path, index=False)
        print(f"{name} dataset exported to: {output_path}")

# ----------------------
# Model and Optimizer Creation
# ----------------------
def create_model(trial, in_channels, edge_dim, out_channels=1):
    if hasattr(trial, 'params') and 'hidden_channels' in trial.params:
        hidden_channels = trial.params['hidden_channels']
        num_layers = trial.params['num_layers']
        num_timesteps = trial.params['num_timesteps']
        dropout = trial.params['dropout']
    else:
        hidden_channels = trial.suggest_int('hidden_channels', 64, 512)
        num_layers = trial.suggest_int('num_layers', 1, 5)
        num_timesteps = trial.suggest_int('num_timesteps', 1, 5)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Ensure hidden_channels is divisible by 4 (for num_heads=4)
    hidden_channels = ((hidden_channels + 3) // 4) * 4
    
    # Use the new enhanced model, add taxonomy_dim parameter
    model = GATGENETAXONOMY(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        edge_dim=edge_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
        taxonomy_dim=TAXONOMY_DIM
    ).to(device)
    return model

def create_optimizer(trial, model):
    if hasattr(trial, 'params') and 'lr' in trial.params:
        lr = trial.params['lr']
        weight_decay = trial.params['weight_decay']
    else:
        lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    
    # Use AdamW optimizer, which often performs better in Transformer-like models
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

# ----------------------
# Training and Evaluation Functions
# ----------------------
def train_epoch(train_loader, model, optimizer, device):
    model.train()
    y_true_list = []
    y_pred_list = []
    total_loss = 0
    total_samples = 0
    
    train_pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for data in train_pbar:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(
            data.x, data.edge_index, data.edge_attr,
            data.batch, data.gene, data.taxonomy, data.duration
        )
        data_y_reshaped = data.y.view(-1, 1)
        loss = F.mse_loss(out, data_y_reshaped)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.y.size(0)
        total_samples += data.y.size(0)
        
        y_true_list.extend(data.y.cpu().numpy())
        y_pred_list.extend(out.detach().cpu().numpy().flatten())
        
        train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    train_pbar.close()
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    train_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    train_r2 = r2_score(y_true, y_pred)
    train_mae = mean_absolute_error(y_true, y_pred)
    
    return train_rmse, train_r2, train_mae

def evaluate(loader, model, device):
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(
                data.x, data.edge_index, data.edge_attr,
                data.batch, data.gene, data.taxonomy, data.duration
            )
            y_true_list.extend(data.y.cpu().numpy())
            y_pred_list.extend(out.cpu().numpy().flatten())
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    return rmse, r2, mae

def objective(trial):
    # Initialize WandB
    batch_size = trial.suggest_int('batch_size', 8, 512)
    wandb.init(
        project=PROJECT_NAME,
        name=f'Trial_{trial.number}',
        mode='offline',
        config={
            'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
            'batch_size': batch_size,  
            #Adjust according to actual circumstances
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'num_layers': trial.suggest_int('num_layers', 1, 5),
            'num_timesteps': trial.suggest_int('num_timesteps', 1, 5),
            'hidden_channels': trial.suggest_int('hidden_channels', 64, 512),
        }
    )

    # Prepare data and model
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        root='.',
        filename=DATA_FILENAME,
        mode=DATA_MODE,
        pre_transform=GenFeatures()
    )
    
    # Export datasets from the Optuna phase
    if trial.number == 0:  # Export only during the first trial to avoid repetition
        print("\n=== Exporting Optuna Phase Datasets ===")
        export_datasets_to_csv(train_dataset, val_dataset, test_dataset, prefix="optuna_", output_dir=f"optuna_dataset_{PROJECT_NAME}")
        print("Optuna phase datasets exported successfully!")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(trial, in_channels=IN_CHANNELS, edge_dim=EDGE_DIM)
    optimizer = create_optimizer(trial, model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    best_val_rmse = float('inf')
    early_stopping_counter = 0
    best_epoch = 0
    best_model_state = None

    # Training loop - add epoch progress bar
    epoch_pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f"Trial {trial.number}", position=1)
    
    for epoch in epoch_pbar:
        train_rmse, train_r2, train_mae = train_epoch(train_loader, model, optimizer, device)
        val_rmse, val_r2, val_mae = evaluate(val_loader, model, device)
        test_rmse, test_r2, test_mae = evaluate(test_loader, model, device)

        # Update epoch progress bar to show current metrics
        epoch_pbar.set_postfix({
            'Val_RMSE': f'{val_rmse:.4f}',
            'Val_R2': f'{val_r2:.4f}',
            'Best_RMSE': f'{best_val_rmse:.4f}'
        })

        # Log to WandB
        wandb.log({
            'epoch': epoch,
            'train/rmse': train_rmse,
            'train/r2': train_r2,
            'train/mae': train_mae,
            'val/rmse': val_rmse,
            'val/r2': val_r2,
            'val/mae': val_mae,
            'test/rmse': test_rmse,
            'test/r2': test_r2,
            'test/mae': test_mae,
        })

        # Early stopping logic
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            # Save the best model state
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= PATIENCE:
                epoch_pbar.set_description(f"Trial {trial.number} - Early Stopped")
                break

        # Report to Optuna
        trial.report(val_rmse, epoch)
        if trial.should_prune():
            epoch_pbar.set_description(f"Trial {trial.number} - Pruned")
            raise optuna.TrialPruned()

    epoch_pbar.close()
    
    # Save the best model and hyperparameters
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model_save_path = os.path.join(MODELS_DIR, f'model_{trial.number}.pth')
        torch.save(model, model_save_path)
        
        print(f"\\nTrial {trial.number} completed:")
        print(f"  Best validation RMSE: {best_val_rmse:.6f} (at epoch {best_epoch})")
        print(f"  Model saved to: {model_save_path}")
    
    wandb.finish()
    return best_val_rmse


# Main function
def main():
    wandb.init(project=PROJECT_NAME, name=f'{PROJECT_NAME}-Optuna Hyperparameter Tuning', mode='offline')
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    )
    print("Starting Optuna hyperparameter optimization...")
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA)

    print(f'\\nBest Trial: RMSE={study.best_value:.4f}')
    print('Hyperparameters:')
    for param, value in study.best_params.items():
        print(f'  {param}: {value}')
    wandb.finish()
    print("\\nStarting final model training...")

    # Reinitialize wandb for final training
    wandb.init(project=PROJECT_NAME, name=f'{PROJECT_NAME}-Final Training', mode='offline')
    best_trial = study.best_trial
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        root='.',
        filename=DATA_FILENAME,
        mode=DATA_MODE,
        pre_transform=GenFeatures()
    )

    # Export datasets from the final training phase
    print("\\n=== Exporting Final Training Phase Datasets ===")
    export_datasets_to_csv(train_dataset, val_dataset, test_dataset, prefix="final_", output_dir=f"final_dataset_{PROJECT_NAME}")
    print("Final training phase datasets exported successfully!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(best_trial, in_channels=IN_CHANNELS, edge_dim=EDGE_DIM)
    optimizer = create_optimizer(best_trial, model)

    # Create data loaders - keep train, validation, and test sets separate
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_trial.params['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=best_trial.params['batch_size'],
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=best_trial.params['batch_size'],
        shuffle=False
    )

    # Initialize variables for tracking the best model
    final_best_val_rmse = float('inf')
    final_best_epoch = 0
    final_best_model_state = None
    final_best_train_rmse = float('inf')
    final_best_train_r2 = 0.0
    final_best_train_mae = float('inf')
    final_best_val_r2 = 0.0
    final_best_val_mae = float('inf')
    final_best_test_rmse = float('inf')
    final_best_test_r2 = 0.0
    final_best_test_mae = float('inf')

    final_patience_counter = 0
    final_min_delta = 1e-6

    final_pbar = tqdm(range(1, MAX_EPOCHS + 1), desc="Final Training")

    for epoch in final_pbar:
        # Training phase
        final_train_rmse, final_train_r2, final_train_mae = train_epoch(
            train_loader, model, optimizer, device
        )
        # Validation phase - for model selection
        final_val_rmse, final_val_r2, final_val_mae = evaluate(
            val_loader, model, device
        )
        # Test phase - for monitoring only, not involved in model selection
        final_test_rmse, final_test_r2, final_test_mae = evaluate(
            test_loader, model, device
        )
    
        if final_val_rmse < final_best_val_rmse - final_min_delta:
            final_best_val_rmse = final_val_rmse
            final_best_epoch = epoch
            final_best_model_state = model.state_dict().copy()
            final_best_model_object = copy.deepcopy(model)
            final_best_train_rmse = final_train_rmse
            final_best_train_r2 = final_train_r2
            final_best_train_mae = final_train_mae
            final_best_val_r2 = final_val_r2
            final_best_val_mae = final_val_mae
            final_best_test_rmse = final_test_rmse
            final_best_test_r2 = final_test_r2
            final_best_test_mae = final_test_mae
            
            final_patience_counter = 0
        else:
            final_patience_counter += 1
    
        # Update progress bar
        final_pbar.set_postfix({
            'Train_R2': f'{final_train_r2:.4f}',
            'Val_RMSE': f'{final_val_rmse:.4f}',
            'Val_R2': f'{final_val_r2:.4f}',
            'Test_R2': f'{final_test_r2:.4f}',
            'Best_Val_RMSE': f'{final_best_val_rmse:.4f}',
            'Best_Epoch': final_best_epoch,
            'Patience': f'{final_patience_counter}/{FINAL_PATIENCE}'
        })

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'final_train/rmse': final_train_rmse,
            'final_train/r2': final_train_r2,
            'final_train/mae': final_train_mae,
            'final_val/rmse': final_val_rmse,
            'final_val/r2': final_val_r2,
            'final_val/mae': final_val_mae,
            'final_test/rmse': final_test_rmse,
            'final_test/r2': final_test_r2,
            'final_test/mae': final_test_mae,
            'best_val_rmse': final_best_val_rmse,
            'patience_counter': final_patience_counter
        })
    
        # Early stopping check
        if final_patience_counter >= FINAL_PATIENCE:
            print(f"\\nEarly stopping triggered! Stopping training at epoch {epoch}")
            print(f"Best validation RMSE at epoch {final_best_epoch}: {final_best_val_rmse:.6f}")
            break

    final_pbar.close()

    if final_best_model_object is not None:
        model_save_path = f'best_{PROJECT_NAME}.pth'
        torch.save(final_best_model_object, model_save_path)
        print(f"\\nComplete model object saved to: {model_save_path}")

    if final_best_model_state is not None:
        model.load_state_dict(final_best_model_state)
        torch.save(model, f'final_{PROJECT_NAME}.pth')
    
        # Log performance metrics of the best model to wandb
        wandb.log({
            'final_best_final_train/rmse': final_best_train_rmse,
            'final_best_final_train/r2': final_best_train_r2,
            'final_best_final_train/mae': final_best_train_mae,
            'final_best_final_val/rmse': final_best_val_rmse,
            'final_best_final_val/r2': final_best_val_r2,
            'final_best_final_val/mae': final_best_val_mae,
            'final_best_final_test/rmse': final_best_test_rmse,
            'final_best_final_test/r2': final_best_test_r2,
            'final_best_final_test/mae': final_best_test_mae,
            'final_best_epoch': final_best_epoch
        })
    
        print(f"\\n=== Final Training Completed! ===")
        print(f"Best model from epoch {final_best_epoch}")
        print(f"\\n=== Training Set Performance ===")
        print(f"RMSE: {final_best_train_rmse:.6f}")
        print(f"R²: {final_best_train_r2:.6f}")
        print(f"MAE: {final_best_train_mae:.6f}")
        print(f"\\n=== Validation Set Performance (Model Selection Criterion) ===")
        print(f"RMSE: {final_best_val_rmse:.6f}")
        print(f"R²: {final_best_val_r2:.6f}")
        print(f"MAE: {final_best_val_mae:.6f}")
        print(f"\\n=== Test Set Performance (Final Evaluation) ===")
        print(f"RMSE: {final_best_test_rmse:.6f}")
        print(f"R²: {final_best_test_r2:.6f}")
        print(f"MAE: {final_best_test_mae:.6f}")
    else:
        print("\\nWarning: Best model state not found, saving the current model")
        torch.save(model, f'fallback_{PROJECT_NAME}.pth')

    wandb.finish()

if __name__ == "__main__":
    main()
