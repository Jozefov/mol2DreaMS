import os
import yaml
from datetime import datetime
import importlib
import torch
from torch import nn
import torch.optim as optim
from mol2dreams.model.mol2dreams import Mol2DreaMS
from mol2dreams.trainer.trainer import Trainer

def build_model_from_config(config):
    def get_class(module_name, class_name):
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(f"Module {module_name} not found.") from e
        except AttributeError as e:
            raise AttributeError(f"Class {class_name} not found in {module_name}.") from e

    # Define module paths for each layer type
    layer_modules = {
        'input_layers': 'mol2dreams.model.InputLayer',
        'body_layers': 'mol2dreams.model.BodyLayer',
        'head_layers': 'mol2dreams.model.HeadLayer'
    }

    # Input Layer
    input_layer_type = config['input_layer']['type']
    input_layer_params = config['input_layer']['params']
    try:
        input_layer_class = get_class(layer_modules['input_layers'], input_layer_type)
        input_layer = input_layer_class(**input_layer_params)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Unsupported input layer type: {input_layer_type}") from e

    # Body Layer
    body_layer_type = config['body_layer']['type']
    body_layer_params = config['body_layer']['params']
    try:
        body_layer_class = get_class(layer_modules['body_layers'], body_layer_type)
        body_layer = body_layer_class(**body_layer_params)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Unsupported body layer type: {body_layer_type}") from e

    # Head Layer
    head_layer_type = config['head_layer']['type']
    head_layer_params = config['head_layer']['params']
    try:
        head_layer_class = get_class(layer_modules['head_layers'], head_layer_type)
        head_layer = head_layer_class(**head_layer_params)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Unsupported head layer type: {head_layer_type}") from e

    # Assemble model
    model = Mol2DreaMS(input_layer, body_layer, head_layer)
    return model

def get_class(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Module {module_name} not found.") from e
    except AttributeError as e:
        raise AttributeError(f"Class {class_name} not found in {module_name}.") from e

def build_loss_from_config(training_config):
    loss_config = training_config['loss_function']
    loss_type = loss_config['type']
    loss_params = loss_config.get('params', {})

    # Try to get the loss function from mol2dreams.trainer.loss
    try:
        loss_class = get_class('mol2dreams.trainer.loss', loss_type)
    except (ImportError, AttributeError):
        # If not found, try to get it from torch.nn
        try:
            loss_class = getattr(nn, loss_type)
        except AttributeError as e:
            raise ValueError(f"Loss function '{loss_type}' not found in mol2dreams.trainer.loss or torch.nn") from e

    loss_fn = loss_class(**loss_params)
    return loss_fn

def build_optimizer_from_config(training_config, model_parameters):
    optimizer_config = training_config['optimizer']
    optimizer_type = optimizer_config['type']
    optimizer_params = optimizer_config.get('params', {})

    # Try to get the optimizer from torch.optim
    try:
        optimizer_class = getattr(optim, optimizer_type)
    except AttributeError as e:
        raise ValueError(f"Optimizer '{optimizer_type}' not found in torch.optim") from e

    optimizer = optimizer_class(model_parameters, **optimizer_params)
    return optimizer

def build_data_loaders_from_config(training_config):
    def load_data_loader(path):
        try:
            data_loader = torch.load(path)
            return data_loader
        except Exception as e:
            raise ValueError(f"Could not load DataLoader from path {path}: {str(e)}")

    train_loader_path = training_config['train_loader']['path']
    val_loader_path = training_config.get('val_loader', {}).get('path')
    test_loader_path = training_config.get('test_loader', {}).get('path')

    train_loader = load_data_loader(train_loader_path)
    val_loader = load_data_loader(val_loader_path) if val_loader_path else None
    test_loader = load_data_loader(test_loader_path) if test_loader_path else None

    return train_loader, val_loader, test_loader

def build_trainer_from_config(config):
    # Build model
    model_config = config['model']
    model = build_model_from_config(model_config)

    # Device
    device = torch.device(config['training'].get('device', 'cpu'))
    model.to(device)

    # Build loss function
    loss_fn = build_loss_from_config(config['training'])

    # Build optimizer
    optimizer = build_optimizer_from_config(config['training'], model.parameters())

    # Load data loaders
    train_loader, val_loader, test_loader = build_data_loaders_from_config(config['training'])

    # Training parameters
    num_epochs = config['training'].get('num_epochs', 50)
    validate_every = config['training'].get('validate_every', 1)
    save_every = config['training'].get('save_every', 1)
    save_best_only = config['training'].get('save_best_only', True)

    # Log directory
    base_log_dir = config['training'].get('log_dir', './logs')
    # Create unique log dir with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_log_dir, f'{timestamp}_mol2dreams')
    os.makedirs(log_dir, exist_ok=True)

    # Save the configuration file in the log directory
    config_save_path = os.path.join(log_dir, 'config.yaml')
    with open(config_save_path, 'w') as file:
        yaml.dump(config, file)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        log_dir=log_dir,
        epochs=num_epochs,
        validate_every=validate_every,
        save_every=save_every,
        save_best_only=save_best_only
    )

    return trainer





