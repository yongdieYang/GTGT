import torch
import random
import numpy as np
import json
import os
from torch_geometric.loader import DataLoader 
import torch.nn.functional as F
import optuna
import wandb
from tqdm import tqdm  
from pre_transform import GenFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from CustomDataset import ToxicDataset
from CMOS import GATGENETAXONOMY

# Configuration block
SEED = 42
NUM_FOLDS = 10  
MAX_EPOCHS = 100-300  #Modified by the user
PATIENCE = 30
BEST_MODEL_PATH = '.pth'
LOG_FILE = '.txt'
PROJECT_NAME = ''

MODELS_DIR = ''
HYPERPARAMS_DIR = ''
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HYPERPARAMS_DIR, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  


set_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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



def create_model(trial, in_channels, edge_dim, out_channels=1):
    hidden_channels = trial.suggest_int('hidden_channels', 64, 512)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_timesteps = trial.suggest_int('num_timesteps', 1, 5)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)  

    model = GATGENETAXONOMY(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        edge_dim=edge_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout
    ).to(device)
    return model


def create_optimizer(trial, model):
    lr = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def save_hyperparameters(trial, best_val_rmse, best_epoch, trial_number):
    hyperparams = {
        'trial_number': trial_number,
        'best_val_rmse': best_val_rmse,
        'best_epoch': best_epoch,
        'hyperparameters': trial.params,
        'model_architecture': {
            'in_channels': 139,
            'edge_dim': 10,
            'out_channels': 1
        }
    }
    
    hyperparams_file = os.path.join(HYPERPARAMS_DIR, f'hyperparameters_{trial_number}.json')
    with open(hyperparams_file, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)
    
    hyperparams_txt_file = os.path.join(HYPERPARAMS_DIR, f'hyperparameters_{trial_number}.txt')
    with open(hyperparams_txt_file, 'w', encoding='utf-8') as f:
        f.write(f'Trial {trial_number} Hyperparameters\n')
        f.write('=' * 50 + '\n')
        f.write(f'Best Validation RMSE: {best_val_rmse:.6f}\n')
        f.write(f'Best Epoch: {best_epoch}\n')
        f.write('\nHyperparameters:\n')
        for param, value in trial.params.items():
            f.write(f'  {param}: {value}\n')
        f.write('\nModel Architecture:\n')
        f.write(f'  in_channels: 139\n')
        f.write(f'  edge_dim: 10\n')
        f.write(f'  out_channels: 1\n')
    
    return hyperparams_file, hyperparams_txt_file


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
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        y_true_list.append(data_y_reshaped.cpu())  
        y_pred_list.append(out.cpu())
        

        train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    

    y_true = torch.cat(y_true_list, dim=0).detach().numpy()
    y_pred = torch.cat(y_pred_list, dim=0).detach().numpy()
    
    train_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    train_r2 = r2_score(y_true, y_pred)
    train_mae = mean_absolute_error(y_true, y_pred)
    
    return train_rmse, train_r2, train_mae


def evaluate(loader, model, device):
    model.eval()
    y_true_list = []
    y_pred_list = []
    

    eval_pbar = tqdm(loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for data in eval_pbar:
            data = data.to(device)
            out = model(
                data.x, data.edge_index, data.edge_attr,
                data.batch, data.gene, data.taxonomy, data.duration
            )

            data_y_reshaped = data.y.view(-1, 1)
            y_true_list.append(data_y_reshaped.cpu())
            y_pred_list.append(out.cpu())

    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, r2, mae


def objective(trial):

    batch_size = trial.suggest_int('batch_size', 8, 512)  
    
    wandb.init(
        project=PROJECT_NAME,
        name=f'Trial_{trial.number}',
        mode='offline',  #Adjust according to actual circumstances
        config={
            'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
            'batch_size': batch_size,  
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'num_layers': trial.suggest_int('num_layers', 1, 5),
            'num_timesteps': trial.suggest_int('num_timesteps', 1, 5),
            'hidden_channels': trial.suggest_int('hidden_channels', 64, 512),
        }
    )

    train_dataset, val_dataset, test_dataset = prepare_datasets(
        root='.',
        filename='yyd_data_train.csv',
        mode="gene+taxonomy",
        pre_transform=GenFeatures()
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(trial, in_channels=139, edge_dim=10)
    optimizer = create_optimizer(trial, model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    best_val_rmse = float('inf')
    early_stopping_counter = 0
    best_epoch = 0
    best_model_state = None

    epoch_pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f"Trial {trial.number}", position=1)
    
    for epoch in epoch_pbar:
        train_rmse, train_r2, train_mae = train_epoch(train_loader, model, optimizer, device)
        val_rmse, val_r2, val_mae = evaluate(val_loader, model, device)
        test_rmse, test_r2, test_mae = evaluate(test_loader, model, device)


        epoch_pbar.set_postfix({
            'Val_RMSE': f'{val_rmse:.4f}',
            'Val_R2': f'{val_r2:.4f}',
            'Best_RMSE': f'{best_val_rmse:.4f}'
        })


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

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch

            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= PATIENCE:
                epoch_pbar.set_description(f"Trial {trial.number} - Early Stopped")
                break


        trial.report(val_rmse, epoch)
        if trial.should_prune():
            epoch_pbar.set_description(f"Trial {trial.number} - Pruned")
            raise optuna.TrialPruned()

    epoch_pbar.close()
    

    if best_model_state is not None:

        model_file = os.path.join(MODELS_DIR, f'model_{trial.number}.pth')

        model.load_state_dict(best_model_state)
        torch.save(model, model_file) 
        

        hyperparams_json, hyperparams_txt = save_hyperparameters(trial, best_val_rmse, best_epoch, trial.number)
        
        print(f"\nTrial {trial.number} 完成:")
        print(f"  模型保存至: {model_file}")
        print(f"  超参数保存至: {hyperparams_json} 和 {hyperparams_txt}")
        print(f"  最佳验证RMSE: {best_val_rmse:.6f}")
    

    with open(LOG_FILE, 'a') as f:
        f.write(f'Trial {trial.number}: Best Val RMSE={best_val_rmse:.4f} at epoch {best_epoch}, '
                f'params={trial.params}\n')

    wandb.finish()
    return best_val_rmse


def load_model_with_hyperparams(trial_number, device='cpu'):

    hyperparams_file = os.path.join(HYPERPARAMS_DIR, f'hyperparameters_{trial_number}.json')
    if not os.path.exists(hyperparams_file):
        raise FileNotFoundError(f"The hyperparameter file does not exist: {hyperparams_file}")
    
    with open(hyperparams_file, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)
    
    model_file = os.path.join(MODELS_DIR, f'model_{trial_number}.pth')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model does not exist: {model_file}")
    
    model = torch.load(model_file, map_location=device)  
    
    return model, hyperparams


def list_saved_models():
    if not os.path.exists(MODELS_DIR):
        print("No saved model directory found")
        return []
    
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith('model_') and f.endswith('.pth')]
    trial_numbers = []
    
    for model_file in model_files:
        try:
            trial_num = int(model_file.split('_')[1].split('.')[0])
            trial_numbers.append(trial_num)
        except:
            continue
    
    trial_numbers.sort()
    
    print(f"Found {len(trial_numbers)} saved models:")
    for trial_num in trial_numbers:
        hyperparams_file = os.path.join(HYPERPARAMS_DIR, f'hyperparameters_{trial_num}.json')
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, 'r', encoding='utf-8') as f:
                hyperparams = json.load(f)
            print(f"  Trial {trial_num}: RMSE={hyperparams['best_val_rmse']:.6f}")
        else:
            print(f"  Trial {trial_num}: Hyperparameter file missing")
    
    return trial_numbers


def main():
    wandb.init(project=PROJECT_NAME, name='Optuna Hyperparameter Tuning',mode='offline')

    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50)
    )

    study.optimize(objective, n_trials=200)

    print(f'\nBest Trial: RMSE={study.best_value:.4f}')
    print('Hyperparameters:')
    for param, value in study.best_params.items():
        print(f'  {param}: {value}')

    best_trial = study.best_trial
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        root='.',
        filename='yyd_data_train.csv',
        mode="gene+taxonomy",
        pre_transform=GenFeatures()
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(best_trial, in_channels=139, edge_dim=10)
    optimizer = create_optimizer(best_trial, model)


    test_loader = DataLoader(
        test_dataset,
        batch_size=best_trial.params['batch_size'],
        shuffle=False  
    )

    full_train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    full_train_loader = DataLoader(
        full_train_dataset,
        batch_size=best_trial.params['batch_size'],
        shuffle=True
    )

    final_pbar = tqdm(range(1, MAX_EPOCHS + 1), desc="Final Training")
    
    for epoch in final_pbar:
        train_rmse, train_r2, train_mae = train_epoch(full_train_loader, model, optimizer, device)

        test_rmse, test_r2, test_mae = evaluate(test_loader, model, device)
        
        final_pbar.set_postfix({
            'Train_R2': f'{train_r2:.4f}',
            'Test_RMSE': f'{test_rmse:.4f}',
            'Test_R2': f'{test_r2:.4f}'
        })

        wandb.log({
            'final_train/r2': train_r2,
            'final_train/rmse': train_rmse,
            'final_train_mae': train_mae,
            'final_test/rmse': test_rmse,
            'final_test/r2': test_r2,
            'final_test/mae': test_mae,
        })

    final_pbar.close()

    torch.save(model, 'final_best_model_clean.pth')
    print("\nThe training is done!")
    
    print("=== All saved models ===")
    list_saved_models()
    
    wandb.finish()


if __name__ == "__main__":
    main()