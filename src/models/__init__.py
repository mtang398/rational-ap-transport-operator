from .fno import FNOTransport
from .deeponet import DeepONetTransport
from .ap_micromacro import APMicroMacroTransport
from .common import FourierFeatures, MLP, SphericalFourierFeatures


def get_model(name: str, **kwargs):
    """Factory function to get a model by name."""
    models = {
        "fno": FNOTransport,
        "deeponet": DeepONetTransport,
        "ap_micromacro": APMicroMacroTransport,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from: {list(models.keys())}")
    return models[name](**kwargs)
