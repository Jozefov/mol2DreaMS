import os
import yaml
from datetime import datetime
import importlib
import torch
from torch import nn
import torch.optim as optim
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
from mol2dreams.model.mol2dreams import Mol2DreaMS
from mol2dreams.model.discriminator import Discriminator

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
        'global_input_layers': 'mol2dreams.model.GlobalInputLayer',
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

    # Global Input Layer (if specified)
    global_input_layer = None
    if 'global_input_layer' in config:
        global_input_layer_type = config['global_input_layer']['type']
        global_input_layer_params = config['global_input_layer']['params']
        try:
            global_input_layer_class = get_class(layer_modules['global_input_layers'], global_input_layer_type)
            global_input_layer = global_input_layer_class(**global_input_layer_params)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unsupported global input layer type: {global_input_layer_type}") from e

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

    # Assemble the model
    model = Mol2DreaMS(input_layer=input_layer, global_input_layer=global_input_layer, body_layer=body_layer, head_layer=head_layer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load pretrained weights if provided
    pretrained_weights_path = config.get('pretrained_weights', None)
    if pretrained_weights_path:
        try:
            model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
            print(f"Loaded pretrained weights from {pretrained_weights_path}")
        except Exception as e:
            raise ValueError(f"Failed to load pretrained weights from {pretrained_weights_path}: {e}")

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

def build_optimizer_from_config(optimizer_config,  model_parameters):
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
    def load_dataset(loader_config):
        path = loader_config['path']
        dataset_type = loader_config.get('dataset_type', None)

        if dataset_type:
            # Construct the module path
            module_path = f"mol2dreams.datasets.{dataset_type}"
            try:
                dataset_class = get_class(module_path, dataset_type)
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Could not import dataset class '{dataset_type}' from module '{module_path}': {e}")
        else:
            dataset_class = None

        try:
            dataset = torch.load(path)
        except Exception as e:
            raise ValueError(f"Could not load dataset from path {path}: {e}")

        # Handle the case where dataset is a list of Data objects
        if isinstance(dataset, list) and all(isinstance(data, Data) for data in dataset):
            if dataset_type == 'SimpleDataset':
                # Use the SimpleDataset class from mol2dreams.datasets.SimpleDataset
                module_path = "mol2dreams.datasets.SimpleDataset"
                dataset_class = get_class(module_path, 'SimpleDataset')
                dataset = dataset_class(dataset)
            else:
                raise ValueError("Dataset is a list of Data objects but 'dataset_type' is not 'SimpleDataset'.")
        elif dataset_class and not isinstance(dataset, Dataset):
            # If dataset is not an instance of Dataset but we have a dataset_class, reconstruct it
            dataset = dataset_class(**dataset)
        elif not isinstance(dataset, Dataset):
            raise ValueError(f"Loaded dataset is not an instance of torch.utils.data.Dataset or list of Data objects.")

        return dataset

    def create_data_loader(loader_config):
        dataset = load_dataset(loader_config)
        batch_size = loader_config.get('batch_size', 32)
        num_workers = loader_config.get('num_workers', 4)
        shuffle = loader_config.get('shuffle', False)
        collate_fn = getattr(dataset, 'collate_fn', None)

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        return data_loader

    train_loader_config = training_config['train_loader']
    val_loader_config = training_config.get('val_loader', {})
    test_loader_config = training_config.get('test_loader', {})

    train_loader = create_data_loader(train_loader_config)
    val_loader = create_data_loader(val_loader_config) if val_loader_config else None
    test_loader = create_data_loader(test_loader_config) if test_loader_config else None

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
    optimizer = build_optimizer_from_config(config['training'].get('optimizer', {}), model.parameters())

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

    # Get trainer type and parameters
    trainer_config = config['training'].get('trainer', {})
    trainer_type = trainer_config.get('type', 'Trainer')
    trainer_params = trainer_config.get('params', {})

    # Dynamically load the trainer class
    try:
        trainer_class = get_class('mol2dreams.trainer.trainer', trainer_type)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Unsupported trainer type: {trainer_type}") from e

    # Initialize Trainer
    if trainer_type == 'AdversarialTrainer':
        # Build the discriminator
        discriminator_config = trainer_params.get('discriminator', {})
        embedding_dim = discriminator_config.get('embedding_dim', 0)
        discriminator = Discriminator(embedding_dim=embedding_dim)
        discriminator.to(device)

        # Build discriminator optimizer
        discriminator_optimizer = build_optimizer_from_config(
            config['training'].get('discriminator_optimizer', {}),
            discriminator.parameters()
        )

        trainer_params.pop('discriminator', None)

        trainer = trainer_class(
            model=model,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
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
            save_best_only=save_best_only,
            **trainer_params
        )
    else:
        trainer_params.pop('discriminator', None)

        trainer = trainer_class(
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
            save_best_only=save_best_only,
            **trainer_params
        )

    return trainer





