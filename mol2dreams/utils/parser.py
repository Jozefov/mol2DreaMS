import importlib
from mol2dreams.model.mol2dreams import Mol2DreaMS

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