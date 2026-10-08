REGISTRY = {}


def register(name: str, input_type: str, framework: str = "torch"):
    """Make a model available to train.py / evaluate.py under `name`.

    input_type: "flat" (batch, n_objects * n_features) or "objects" (batch, n_objects, n_features)
    framework:  "torch" (trained with gradient descent in train.py) or "sklearn" (trained with .fit)
    """
    def decorator(cls):
        assert name not in REGISTRY, f"Model name '{name}' is already registered."
        REGISTRY[name] = {
            "class": cls,
            "input_type": input_type,
            "framework": framework,
        }
        return cls
    return decorator


def class_to_config(model_class):
    for name, cfg in REGISTRY.items():
        if cfg["class"] == model_class:
            return name, cfg
    raise ValueError(f"Model class {model_class} not found in registry.")
