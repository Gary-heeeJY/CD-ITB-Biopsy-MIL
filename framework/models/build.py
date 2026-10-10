import functools
import inspect
import os

import torch
from fvcore.common.config import CfgNode as _CfgNode
from fvcore.common.registry import Registry

from models.MambaMIL import MambaMIL


MODEL_FRAMEWORK_REGISTRY = Registry("FRAME")


def load_model_model(cfg, model, case_model):
    """Load a hierarchical checkpoint and return optimizer metadata."""
    if cfg.model_weights:
        model_path = os.path.join(
            cfg.model_dir,
            f"{cfg.Task_name}/fold-{cfg.DATA.fold_idx}/case_best_model.pt",
        )
        model_dict = model.state_dict()
        pretrained_state = torch.load(model_path)
        pretrained_dict = {
            key: value
            for key, value in pretrained_state["model"].items()
            if key in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        lr = pretrained_state["optimizer1"]["param_groups"][0]["lr"]
        print(f"Loading pre-train model from {model_path}")
        return model, {
            "params": model.parameters(),
            "case_params": case_model.parameters(),
            "lr": lr,
            "case_lr": lr,
            "epoch": 0,
        }

    print("Training from scratch..............")
    return model, {
        "params": model.parameters(),
        "case_params": case_model.parameters(),
        "lr": cfg.SOLVER.lr,
        "case_lr": cfg.SOLVER.case_lr,
        "epoch": 0,
    }


def build_model(cfg, case_model):
    """Build one of the models used by the manuscript experiments."""
    if cfg.model_type == "GatedAttention":
        from models.abmil import GatedAttention

        model = GatedAttention(
            hidden_dim=cfg.DATA.hidden_dim,
            n_classes=cfg.DATA.n_classes,
        )
    elif cfg.model_type == "MambaMIL":
        model = MambaMIL(
            in_dim=cfg.DATA.hidden_dim,
            n_classes=cfg.DATA.n_classes,
            dropout=cfg.MODEL.Mamba.dropout,
            act=cfg.MODEL.Mamba.act,
            layer=cfg.MODEL.Mamba.layer,
            rate=cfg.MODEL.Mamba.rate,
            type=cfg.MODEL.Mamba.type,
        )
    elif cfg.model_type == "TransMIL":
        from models.TransMIL import TransMIL

        model = TransMIL(n_classes=cfg.DATA.n_classes)
    else:
        model_class = MODEL_FRAMEWORK_REGISTRY.get(cfg.model_type)
        try:
            model = model_class(cfg.DATA)
        except TypeError:
            model = model_class(cfg.DATA.hidden_dim, cfg.DATA.n_classes)

    device = torch.device(
        f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    case_model.to(device)
    params = [
        {"params": model.parameters(), "lr": cfg.SOLVER.lr},
        {"params": case_model.parameters(), "lr": cfg.SOLVER.case_lr},
    ]
    try:
        print("model path", inspect.getfile(type(model)))
    except (OSError, TypeError):
        pass
    return model, params


def configurable(init_func=None, *, from_config=None):
    """Allow a function or initializer to accept a YACS config node."""
    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable."

        @functools.wraps(init_func)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as exc:
                raise AttributeError(
                    "Class with @configurable needs a from_config classmethod."
                ) from exc
            if not inspect.ismethod(from_config_func):
                raise TypeError("from_config must be a classmethod.")
            if _called_with_cfg(*args, **kwargs):
                init_func(self, **_get_args_from_config(from_config_func, *args, **kwargs))
            else:
                init_func(self, *args, **kwargs)

        return wrapped

    if from_config is None:
        return configurable
    assert inspect.isfunction(from_config), "from_config must be a function."

    def wrapper(orig_func):
        @functools.wraps(orig_func)
        def wrapped(*args, **kwargs):
            if _called_with_cfg(*args, **kwargs):
                return orig_func(**_get_args_from_config(from_config, *args, **kwargs))
            return orig_func(*args, **kwargs)

        wrapped.from_config = from_config
        return wrapped

    return wrapper


def _get_args_from_config(from_config_func, *args, **kwargs):
    signature = inspect.signature(from_config_func)
    if list(signature.parameters)[0] != "cfg":
        raise TypeError("from_config must take cfg as its first argument.")
    supports_var_args = any(
        parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    )
    if supports_var_args:
        return from_config_func(*args, **kwargs)
    supported = set(signature.parameters)
    extra_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key not in supported}
    result = from_config_func(*args, **kwargs)
    result.update(extra_kwargs)
    return result


def _called_with_cfg(*args, **kwargs):
    if args and isinstance(args[0], _CfgNode):
        return True
    return isinstance(kwargs.pop("cfg", None), _CfgNode)
