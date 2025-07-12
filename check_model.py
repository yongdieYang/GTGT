import torch

# Load the model
model = torch.load('final_best_model.pth', map_location='cpu')

print('Model type:', type(model))
print('\n' + '='*50)

if isinstance(model, dict):
    print('Model is a dictionary with keys:')
    for key in model.keys():
        print(f'  - {key}: {type(model[key])}')
        if hasattr(model[key], 'shape'):
            print(f'    Shape: {model[key].shape}')
        elif hasattr(model[key], '__len__'):
            try:
                print(f'    Length: {len(model[key])}')
            except:
                pass
    
    # Check if it contains model state_dict
    if 'model_state_dict' in model:
        print('\nModel state_dict keys:')
        state_dict = model['model_state_dict']
        for key in list(state_dict.keys())[:10]:  # Show first 10 keys
            print(f'  - {key}: {state_dict[key].shape}')
        if len(state_dict.keys()) > 10:
            print(f'  ... and {len(state_dict.keys()) - 10} more layers')
    
    # Check if it contains optimizer state
    if 'optimizer_state_dict' in model:
        print('\nOptimizer state_dict present')
    
    # Check for other common keys
    common_keys = ['epoch', 'loss', 'val_loss', 'best_val_loss', 'hyperparameters']
    for key in common_keys:
        if key in model:
            print(f'\n{key}: {model[key]}')
else:
    print('Model is not a dictionary, likely just the model state_dict')
    if hasattr(model, 'keys'):
        print('Model layers:')
        for key in list(model.keys())[:10]:  # Show first 10 keys
            print(f'  - {key}: {model[key].shape}')
        if len(model.keys()) > 10:
            print(f'  ... and {len(model.keys()) - 10} more layers')