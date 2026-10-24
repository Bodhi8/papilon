"""Marketing mix modeling with Papilon's gated importance sampling (RFC 001)."""

from .model import MMM, GateResult
from .transforms import delayed_adstock, geometric_adstock, hill

__all__ = ["MMM", "GateResult", "geometric_adstock", "delayed_adstock", "hill"]
