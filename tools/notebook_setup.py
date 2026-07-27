"""Bootstrap helper for notebooks: adds the repo root to `sys.path`.

Run this with IPython's %run magic (works in both Jupyter and plain scripts):

    %run ../tools/notebook_setup.py

Adjust the relative path depending on where the notebook lives relative to the repo root.
"""
import sys
from pathlib import Path


def setup_repo_path() -> Path:
    """Locate the repo root (looks for pyproject.toml) and add it to sys.path."""
    here = Path(globals().get("__file__", Path.cwd())).resolve()
    repo_root = next(p for p in [here, *here.parents] if (p / "pyproject.toml").exists())
    
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        
    return repo_root


repo_root = setup_repo_path()
