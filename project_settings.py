#%%
"""
Backward-compatibility shim.

All configuration now lives in ``tiptorch._config``.  This file re-exports
everything so that existing research scripts using
``from project_settings import ...`` keep working unchanged.
"""
# Re-export everything from the installed package config
from tiptorch._config import *  # noqa: F401,F403
