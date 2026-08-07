"""
Training launcher wrapper — sets UTF-8 encoding before importing Lightning
to avoid Windows cp1252 UnicodeEncodeError from emoji characters in Lightning output.
"""
import sys
import os

# Force UTF-8 before anything else loads
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Reconfigure stdout/stderr to UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Now import and run train
sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("train", "scripts/train.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
