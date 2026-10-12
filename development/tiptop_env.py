"""
Make the astro-tiptop checkouts importable without installing them into the conda environment.

The repositories are expected one level above the TipTorch repository, in ``<projects>/astro-tiptop/{TIPTOP,P3,MASTSEL,SYMAO,SEEING}``
(``git clone https://github.com/astro-tiptop/<name>.git``), and MAOPPY (used for the PSFAO Strehl metric) unpacked in
``<projects>/maoppy``. Import this module before anything from ``tiptop``, ``p3``, ``mastsel`` or ``maoppy``::

    import tiptop_env  # noqa: F401
    from tiptop.tiptop import overallSimulation

``tiptop.asterismSimulation`` imports ``torchmetrics`` (only needed for its NN asterism selector); when the package is missing, a stub
module is registered so that ``overallSimulation`` stays importable.
"""

import sys
import types
from pathlib import Path

PROJECTS_DIR  = Path(__file__).resolve().parents[2]
ASTRO_TIPTOP  = PROJECTS_DIR / 'astro-tiptop'
TIPTOP_REPOS  = ['TIPTOP', 'P3', 'MASTSEL', 'SYMAO', 'SEEING']
MAOPPY_DIR    = PROJECTS_DIR / 'maoppy'


def add_to_path():
    missing = [name for name in TIPTOP_REPOS if not (ASTRO_TIPTOP / name).is_dir()]
    if missing:
        raise ImportError(f"astro-tiptop checkouts not found in {ASTRO_TIPTOP}: {missing}. Clone them from https://github.com/astro-tiptop.")

    for name in TIPTOP_REPOS:
        path = str(ASTRO_TIPTOP / name)
        if path not in sys.path:
            sys.path.insert(0, path)

    if MAOPPY_DIR.is_dir() and str(MAOPPY_DIR) not in sys.path:
        sys.path.insert(0, str(MAOPPY_DIR))

    try:
        import torchmetrics  # noqa: F401
    except ImportError:
        stub = types.ModuleType('torchmetrics')
        regression = types.ModuleType('torchmetrics.regression')
        for cls_name in ('SymmetricMeanAbsolutePercentageError', 'MeanAbsolutePercentageError', 'MeanAbsoluteError'):
            setattr(regression, cls_name, type(cls_name, (), {}))
            
        stub.regression = regression
        sys.modules['torchmetrics'] = stub
        sys.modules['torchmetrics.regression'] = regression


add_to_path()
