REGISTRY = {}


def register(name: str, input_type: str, framework: str = "keras"):
    """Make a model available to train.py / evaluate.py under `name`.

    input_type: "flat" (batch, n_objects * n_features) or "objects" (batch, n_objects, n_features)
    framework:  "keras" (a function returning a Keras autoencoder, trained with model.fit in train.py)
                or "sklearn" (a SklearnModelWrapper class, trained with .fit)
    """
    def decorator(build):
        assert name not in REGISTRY, f"Model name '{name}' is already registered."
        REGISTRY[name] = {
            "build": build,
            "input_type": input_type,
            "framework": framework,
        }
        return build
    return decorator


def class_to_config(model_class):
    for name, cfg in REGISTRY.items():
        if cfg["build"] == model_class:
            return name, cfg
    raise ValueError(f"Model class {model_class} not found in registry.")
