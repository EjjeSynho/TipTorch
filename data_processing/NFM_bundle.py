"""
The MUSE NFM calibrator bundle (WEIGHTS_FOLDER/NFM_calibrator/NFM_calibrator_bundle.pth): the single file distributed to the users (resource
pack MUSE_NFM_cailbrator), sufficient to process the MUSE NFM data and to calibrate the PSF model. It is built up in stages, so that the data
loaders (ProcessMUSEcube, InitNFMConfig, reduce_dataframe) find what they need before any calibrator is trained:

    1. telemetry resources (create_STD_dataset.py, pack_telemetry_resources):
         'archive_telemetry'   LGS flux / LGS slopes / IRLOS regimes tables fetched from the ESO datalab (CSV text, time-indexed), the source of
                               this telemetry for the cubes whose headers miss it (GetLGSdata, GetIRLoopData)
         'median_Cn2_profile'  median AOF Cn2 profile (fallback and K-means reference of InitNFMConfig / reduce_dataframe)
         'Cn2_binning'         reduction of the AOF Cn2 profile ('kmeans' or 'grid', MUSE_data_utils.CN2_BINNING)
         'Cn2_grid'            fixed Cn2 grid (Cn2Grid.state_dict(), 'grid' binning only)
    2. telemetry preprocessor (create_STD_dataset.py): 'telemetry_preprocessor', the scaler + imputer of the calibrator features
    3. calibrator: network weights, output transforms and model metadata. NFMCalibratorTrainer.save_calibrator writes a run bundle on top of
       stages 1-2, install_calibrator copies it into the distributed bundle

The writable working copies of the dataset building stay in the STD folder (STD_RESOURCES_FOLDER): the archive tables the datalab fetches are
appended to and the Cn2 grid of store_Cn2_grid.
"""
import logging
import numpy as np
import torch
from pathlib import Path

from tiptorch._config import WEIGHTS_FOLDER, project_settings
from data_processing.Cn2_profile import Cn2Grid

logger = logging.getLogger(__name__)

BUNDLE_PATH = WEIGHTS_FOLDER / 'NFM_calibrator' / 'NFM_calibrator_bundle.pth'

# Working copies of the dataset building, None on the user side (no STD folder configured)
STD_RESOURCES_FOLDER = Path(project_settings['MUSE_STD_data_folder']) / 'telemetry_resources' if project_settings.get('MUSE_STD_data_folder') else None
CN2_GRID_FILE = 'AOF_Cn2_grid.json'

ARCHIVE_TABLES = { 'IRLOS_regimes': 'IRLOS_regimes_df.csv', 'LGS_flux': 'LGS_flux_df.csv', 'LGS_slopes': 'LGS_slopes_df.csv' }
RESOURCE_KEYS  = ('archive_telemetry', 'median_Cn2_profile', 'Cn2_binning', 'Cn2_grid', 'telemetry_preprocessor') # stages 1-2

_loaded = {} # path -> ((mtime, size), state)


def load_bundle(path=None) -> dict:
    """ State of the bundle (empty if it does not exist yet), cached until the file changes. Do not modify it in place """
    path = Path(path or BUNDLE_PATH)
    if not path.exists():
        return {}
    stat = path.stat()
    key  = (stat.st_mtime_ns, stat.st_size)
    if path not in _loaded or _loaded[path][0] != key:
        _loaded[path] = (key, torch.load(path, map_location='cpu', weights_only=False))
    return _loaded[path][1]


def update_bundle(entries: dict, path=None) -> Path:
    """ Adds / replaces entries of the bundle (created if needed) and keeps the others: the staged build-up of the bundle """
    path = Path(path or BUNDLE_PATH)
    state = {**load_bundle(path), **entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def telemetry_resources(path=None) -> dict:
    """ The entries of stages 1-2 present in the bundle """
    state = load_bundle(path)
    return {k: state[k] for k in RESOURCE_KEYS if k in state}


def _warn_if_calibrated(path, what):
    if 'net_state_dict' in load_bundle(path):
        logger.warning(f'{what} of {Path(path or BUNDLE_PATH).name} replaced under its installed calibrator, which was trained with the previous '
                       'ones: install a calibrator trained on the new dataset (install_calibrator)')


def pack_telemetry_resources(median_Cn2_profile, Cn2_binning, Cn2_grid=None, path=None) -> Path:
    """
    Stage 1: the archive tables (working copies of the STD folder; a table without a working copy keeps the one of the bundle), the median
    AOF Cn2 profile (median_Cn2_alts, median_Cn2_fracs), the Cn2 binning and its grid (Cn2Grid, 'grid' binning only).
    """
    tables = dict(load_bundle(path).get('archive_telemetry', {}))
    
    for name, file in ARCHIVE_TABLES.items():
        if STD_RESOURCES_FOLDER is not None and (STD_RESOURCES_FOLDER / file).exists():
            tables[name] = (STD_RESOURCES_FOLDER / file).read_text('utf-8')
        
        elif name not in tables:
            logger.warning(f'No {name} table in {STD_RESOURCES_FOLDER} nor in the bundle: the cubes without this telemetry in their headers get the default values')

    median_alts, median_fracs = median_Cn2_profile
   
    _warn_if_calibrated(path, 'The telemetry resources')
   
    return update_bundle({
        'archive_telemetry':  tables,
        'median_Cn2_profile': { 'median_Cn2_alts': np.asarray(median_alts, dtype=float).tolist(), 'median_Cn2_fracs': np.asarray(median_fracs, dtype=float).tolist() },
        'Cn2_binning':        Cn2_binning,
        'Cn2_grid':           Cn2_grid.state_dict() if Cn2_grid is not None else None,
    }, path)


def pack_telemetry_preprocessor(preprocessor_state: dict, path=None) -> Path:
    """ Stage 2: TelemetryPreprocessor.state_dict() of the calibrator features """
    _warn_if_calibrated(path, 'The telemetry preprocessor')
    return update_bundle({'telemetry_preprocessor': preprocessor_state}, path)


def install_calibrator(run_bundle_path, path=None) -> Path:
    """ Stage 3: copies a run bundle (NFMCalibratorTrainer.save_calibrator) into the distributed bundle """
    path = update_bundle(torch.load(run_bundle_path, map_location='cpu', weights_only=False), path)
    logger.info(f'Calibrator {Path(run_bundle_path).name} installed -> {path}')
    return path


def archive_table_csv(name, path=None) -> str | None:
    """ CSV text of an archive table of the bundle (ARCHIVE_TABLES), None if it has none """
    return load_bundle(path).get('archive_telemetry', {}).get(name)


def archive_cache_path(name) -> Path | None:
    """ Working copy of an archive table in the STD folder (None on the user side) """
    return STD_RESOURCES_FOLDER / ARCHIVE_TABLES[name] if STD_RESOURCES_FOLDER is not None else None


def load_median_Cn2_profile(profile: dict | None = None, path=None):
    """ (median_Cn2_alts [km], median_Cn2_fracs) of the given 'median_Cn2_profile' entry, by default the one of the bundle """
    profile = profile or load_bundle(path).get('median_Cn2_profile')
    if profile is None:
        raise ValueError(f'No median Cn2 profile in {Path(path or BUNDLE_PATH)}: build the bundle with create_STD_dataset.py or install the resource pack')
    return np.array(profile['median_Cn2_alts']), np.array(profile['median_Cn2_fracs'])


def load_bundle_Cn2_grid(path=None) -> Cn2Grid:
    """ Fixed Cn2 grid of the bundle (present with the 'grid' binning only) """
    state = load_bundle(path).get('Cn2_grid')
    if state is None:
        raise ValueError(f"No Cn2 grid in {Path(path or BUNDLE_PATH)} (Cn2 binning '{load_bundle(path).get('Cn2_binning')}'): "
                         "the bundle was built with another Cn2 binning or not at all, see create_STD_dataset.py")
    return Cn2Grid.from_state_dict(state)
