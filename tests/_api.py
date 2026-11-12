"""Collect the public API: names in each module's __all__ and their parameters (and public methods)."""
import importlib
import inspect

MODULES = ["papilon", "papilon.mmm", "papilon.ai", "papilon.io", "papilon.discovery", "papilon.persist",
           "papilon.report", "papilon.tracking", "papilon.loop"]


def params(obj):
    try:
        return [p for p in inspect.signature(obj).parameters if p not in ("self", "cls")]
    except (TypeError, ValueError):
        return None


def collect():
    out = {}
    for mod in MODULES:
        m = importlib.import_module(mod)
        for name in getattr(m, "__all__", []):
            obj = getattr(m, name)
            key = f"{mod}.{name}"
            out[key] = {"params": params(obj)}
            if inspect.isclass(obj):
                out[key]["methods"] = {k: params(v) for k, v in vars(obj).items()
                                       if not k.startswith("_") and callable(v)}
    return out
