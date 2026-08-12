"""Companion code for "Learning When Not to Inspect: Risk-Calibrated Weak
Supervision for Efficient Industrial Visual Inspection".

Three stages: (1) weak scout trained from image-level OK/NOK labels only,
(2) risk-calibrated Fast-OK-Exit gate, (3) split-conformal
accept/review/reject decision layer. See README.md.

Exports are lazy (PEP 562) so ``python -m selective_inspection.<tool>``
does not pre-import sibling modules.
"""

from typing import Any

__version__ = "0.1.0"

_LAZY = {
    "WeakScout": ("selective_inspection.model", "WeakScout"),
    "WeakScoutLogits": ("selective_inspection.model", "WeakScoutLogits"),
    "load_checkpoint": ("selective_inspection.model", "load_checkpoint"),
    "save_checkpoint": ("selective_inspection.model", "save_checkpoint"),
    "weak_bce_loss": ("selective_inspection.model", "weak_bce_loss"),
    "SelectiveInspectionPipeline": ("selective_inspection.pipeline", "SelectiveInspectionPipeline"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
