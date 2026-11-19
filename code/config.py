"""Project paths used by code/seg/collect_results.py and code/seg/prepare_kv_inputs.py (the other scripts take their paths from the command line).

  NDVI_GVI_ROOT   repository root                          (default: the directory above code/)
  NDVI_GVI_CACHE  large external and derived inputs        (default: <parent of ROOT>/cache; prepare_kv_inputs.py reads <cache>/us/kv)

Example:  export NDVI_GVI_CACHE=/data/ndvi_gvi_cache ; python code/seg/prepare_kv_inputs.py
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("NDVI_GVI_ROOT", Path(__file__).resolve().parents[1]))
CACHE = Path(os.environ.get("NDVI_GVI_CACHE", ROOT.parent / "cache"))
CACHE_US = CACHE / "us"
TABLES = ROOT / "results" / "tables"            # output of collect_results.py
TABLES.mkdir(parents=True, exist_ok=True)
