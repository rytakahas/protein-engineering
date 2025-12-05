# src/resintnet/ingest/adapters/__init__.py
from .generic_csv import load_dms_csv

try:
    from .d3distal import load_d3distal
except Exception as e:
    # Optional adapter: only needed if you call it explicitly
    def load_d3distal(*args, **kwargs):
        raise ImportError(
            "load_d3distal() is optional and currently unavailable. "
            f"Reason: {e}"
        )

__all__ = ["load_dms_csv", "load_d3distal"]

