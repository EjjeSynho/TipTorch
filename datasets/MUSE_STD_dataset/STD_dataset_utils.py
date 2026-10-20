import sys
from pathlib import Path

# Make the project root importable when this module is used from datasets/MUSE_STD_dataset/
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_processing.MUSE_data_utils import *


STD_FOLDER   = Path(project_settings["MUSE_STD_data_folder"])
CUBES_FOLDER = STD_FOLDER / 'cubes/'
RAW_FOLDER   = STD_FOLDER / 'raw/'
CUBES_CACHE  = STD_FOLDER / 'cached_cubes/'


def RemoveDuplicatedCubes(cubes_cache_dir, verbose=True, dry_run=False):
    """
    Removes cached cubes that share the same observation (the filename without
    the leading numeric ID prefix, e.g. 'M.MUSE.2021-03-30T15-04-31.323.pickle')
    but have different IDs. The file with the lowest ID is kept.

    Args:
        cubes_cache_dir: Path to cached cubes directory
        verbose:         Print progress information
        dry_run:         If True, only report what would be removed

    Returns:
        list of removed (or, in dry run, to-be-removed) filenames
    """
    import re
    from collections import defaultdict

    id_pattern = re.compile(r'^(?:(\d+)_)?(.+)$')

    groups = defaultdict(list)
    for pfile in os.listdir(cubes_cache_dir):
        if not pfile.endswith('.pickle'):
            continue
        m = id_pattern.match(pfile)
        file_id = int(m.group(1)) if m.group(1) is not None else -1
        groups[m.group(2)].append((file_id, pfile))

    removed = []
    for base, files in groups.items():
        if len(files) < 2:
            continue
        files.sort()
        kept = files[0][1]
        for _, pfile in files[1:]:
            if verbose:
                print(f"  Duplicate of '{kept}': {'would remove' if dry_run else 'removing'} '{pfile}'")
            if not dry_run:
                os.remove(os.path.join(cubes_cache_dir, pfile))
            removed.append(pfile)

    if verbose:
        print(f"{len(removed)} duplicated cube(s) {'found' if dry_run else 'removed'}")
    return removed


def SplitByAcquisitionTime(acq_times, val_fraction=0.2, max_gap_hours=1.0, seed=42, verbose=True):
    """
    Splits samples into training and validation subsets so that observations acquired close in time
    (and thus sharing nearly the same atmospheric conditions and instrument state) never end up on
    different sides of the split.

    Samples are sorted by acquisition time and chunked into groups: a new group starts whenever the
    gap to the previous sample exceeds max_gap_hours. Whole groups are then randomly assigned to the
    validation set until it holds ~val_fraction of the samples.

    Args:
        acq_times:      [N] array-like of real acquisition times (DATE-OBS of the cube header, not the
                        timestamp in the file name), anything pd.to_datetime understands
        val_fraction:   Target fraction of samples in the validation subset
        max_gap_hours:  Max time gap between consecutive samples of the same group
        seed:           RNG seed for the group shuffling
        verbose:        Print summary

    Returns:
        train_idx, val_idx, group_ids: positions into acq_times (sorted), and the [N] group label of every sample
    """
    times = pd.to_datetime(pd.Series(np.asarray(acq_times)), utc=True)
    order = np.argsort(times.values, kind='stable')
    gaps  = times.iloc[order].diff().dt.total_seconds().fillna(np.inf).values / 3600.0

    group_sorted = np.cumsum(gaps > max_gap_hours) - 1
    group_ids = np.empty(len(times), dtype=int)
    group_ids[order] = group_sorted

    rng = np.random.default_rng(seed)
    n_val_target = val_fraction * len(times)
    val_groups, n_val = [], 0

    for g in rng.permutation(group_ids.max() + 1):
        if n_val >= n_val_target:
            break
        size = int((group_ids == g).sum())
        # Skip a group that would overshoot the target by more than it brings closer to it
        if n_val + size - n_val_target > n_val_target - n_val and n_val > 0:
            continue
        val_groups.append(g)
        n_val += size

    is_val = np.isin(group_ids, val_groups)
    train_idx, val_idx = np.where(~is_val)[0], np.where(is_val)[0]

    if verbose:
        print(f'Split by acquisition time: {group_ids.max()+1} groups (gap > {max_gap_hours} h), '
              f'{len(train_idx)} train / {len(val_idx)} val ({100*len(val_idx)/len(times):.1f}%)')
    
    return train_idx, val_idx, group_ids


def ReasignIDsByDate(cubes_cache_dir, verbose=True):
    """
    Renames all cached pickle files so their numeric ID prefix reflects
    chronological order: the earliest observation gets ID 0, the next ID 1,
    and so on.  The observation date is parsed from the basename, e.g.

        M.MUSE.2019-11-13T12-43-58.070.pickle  ->  65_M.MUSE.2019-...

    Date parsing uses the ISO-8601-like fragment 'YYYY-MM-DDTHH-MM-SS.mmm'
    embedded in the filename (colons replaced by dashes as stored on disk).

    Args:
        cubes_cache_dir: Path to cached cubes directory
        verbose:         Print progress information

    Returns:
        list of (old_filename, new_filename) for every renamed file
    """
    import re
    from datetime import datetime

    id_pattern   = re.compile(r'^(?:\d+_)?(.+)$')          # strip optional leading ID_
    date_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)'   # YYYY-MM-DDTHH-MM-SS.mmm
    )

    def parse_date(basename):
        m = date_pattern.search(basename)
        if not m:
            return None
        # Restore colons so strptime can parse it
        return datetime.strptime(m.group(1), '%Y-%m-%dT%H-%M-%S.%f')

    # Collect (datetime, current_full_filename, base_filename)
    samples = []
    for pfile in os.listdir(cubes_cache_dir):
        if not pfile.endswith('.pickle'):
            continue
        
        m = id_pattern.match(pfile)
        if not m:
            continue
        
        base = m.group(1)       # filename without the leading ID prefix
        dt   = parse_date(base)
        
        if dt is None:
            if verbose:
                print(f"  WARNING: cannot parse date from '{pfile}' — skipping")
            continue
        
        samples.append((dt, pfile, base))

    samples.sort(key=lambda x: x[0])   # chronological order

    rename_plan = []  # (old_path, new_path, old_name, new_name)
    for new_id, (dt, full_filename, base_filename) in enumerate(samples):
        new_name = f"{new_id}_{base_filename}"
        if full_filename != new_name:
            rename_plan.append((
                os.path.join(cubes_cache_dir, full_filename),
                os.path.join(cubes_cache_dir, new_name),
                full_filename, new_name
            ))

    if rename_plan:
        if verbose:
            print(f"Reassigning IDs by date: {len(rename_plan)} file(s) will be renamed...")

        # Two-pass rename via .tmp to avoid conflicts
        for old_path, new_path, old_name, new_name in rename_plan:
            if os.path.exists(old_path):
                os.rename(old_path, new_path + '.tmp')

        for old_path, new_path, old_name, new_name in rename_plan:
            if os.path.exists(new_path + '.tmp'):
                os.rename(new_path + '.tmp', new_path)
                if verbose:
                    print(f"  {old_name} -> {new_name}")
    else:
        if verbose:
            print("IDs are already in chronological order. Nothing to rename.")

    return [(old_name, new_name) for old_path, new_path, old_name, new_name in rename_plan]


def SyncLabelsToCubes(labels_file, cubes_cache_dir, verbose=True):
    """
    Re-synchronizes the numeric ID prefix in every labels.txt entry so that
    it matches the actual ID of the corresponding cached pickle file.

    Matching is done purely by basename (the part after 'ID_'), making this
    function robust to any prior renaming.  Entries with no matching pickle
    file are left unchanged with a warning.

    Args:
        labels_file:     Path to labels.txt
        cubes_cache_dir: Path to cached cubes directory
        verbose:         Print progress information

    Returns:
        Number of label entries updated
    """
    import re

    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    id_pattern = re.compile(r'^(\d+)_(.+)$')

    # Build lookup: base pickle name (no ID prefix) -> current ID on disk
    base_to_id = {}
    for pfile in os.listdir(cubes_cache_dir):
        if not pfile.endswith('.pickle'):
            continue
        m = id_pattern.match(pfile)
        if m:
            base_to_id[m.group(2)] = int(m.group(1))  # e.g. 'M.MUSE....pickle' -> 42

    with open(labels_file, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    n_updated = 0

    for line in lines:
        if ': ' not in line:
            updated_lines.append(line)
            continue

        filename_part, labels_part = line.strip().split(': ', 1)
        m = id_pattern.match(filename_part)

        if not m:
            updated_lines.append(line)
            continue

        current_id  = int(m.group(1))
        base_png    = m.group(2)                            # e.g. 'M.MUSE....png'
        base_pickle = base_png.replace('.png', '.pickle')

        if base_pickle not in base_to_id:
            if verbose:
                print(f"  WARNING: no pickle found for '{base_png}' — skipping")
            updated_lines.append(line)
            continue

        correct_id = base_to_id[base_pickle]

        if current_id != correct_id:
            new_entry = f"{correct_id}_{base_png}: {labels_part}\n"
            updated_lines.append(new_entry)
            if verbose:
                print(f"  {filename_part} -> {correct_id}_{base_png}")
            n_updated += 1
        else:
            updated_lines.append(line)

    with open(labels_file, 'w') as f:
        f.writelines(updated_lines)

    if verbose:
        print(f"\nLabels sync complete: {n_updated} entr{'y' if n_updated == 1 else 'ies'} updated.")

    return n_updated


def FindDuplicatesFITS(folder):
    ''' Finds duplicates in a folder of FITS files based on ESO OBS ID, TPL START, and DATE-OBS values. '''
    records = []
    for file in sorted(os.listdir(folder)):
        fpath = folder / file
        if not fpath.is_file():
            continue
        try:
            with fits.open(fpath) as hdul:
                hdr = hdul[0].header
                records.append({
                    'file':     file,
                    'OB_ID':    hdr.get('ESO OBS ID',       hdr.get('HIERARCH ESO OBS ID',    None)),
                    'TPL_START': hdr.get('ESO TPL START',   hdr.get('HIERARCH ESO TPL START', None)),
                    'DATE_OBS': hdr.get('DATE-OBS',         None),
                })
        except Exception as e:
            print(f"  Warning: could not read {file}: {e}")

    df_cubes = pd.DataFrame(records)

    for key in ['OB_ID', 'TPL_START', 'DATE_OBS']:
        dups = df_cubes[df_cubes.duplicated(subset=[key], keep=False) & df_cubes[key].notna()]
        if dups.empty:
            print(f"No duplicates found for {key}")
        else:
            print(f"\nDuplicates on {key}:")
            print(dups[['file', key]].sort_values(key).to_string(index=False))


def LoadSTDStarCacheByID(id):
    ''' Searches a specific STD star by its ID in the list of cached cubes. '''
    with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
        muse_df = pickle.load(handle)

    file = muse_df.loc[muse_df.index == id]['Filename'].values[0]
    full_filename = CUBES_CACHE / f'{id}_{file}.pickle'

    with open(full_filename, 'rb') as handle:
        data_sample, _ = pickle.load(handle)
    
    data_sample['All data']['Pupil angle'] = muse_df.loc[id]['Pupil angle']
    # k_c of the sample relative to the nominal one (1.0 unless the sample is flagged 'k_c mismatch', see add_kc_mismatch_flag)
    data_sample['All data'][KC_FRACTION_FEATURE] = float(muse_df.loc[id][KC_FRACTION_FEATURE]) if KC_FRACTION_FEATURE in muse_df.columns else 1.0
    return data_sample


def LoadSTDStarCache(
    ids,
    derotate_PSF = False,
    normalize = True,
    subtract_background = False,
    wvl_ids = None,
    ensure_odd_pixels = False,
    device = default_device
):
    """ Loads data associated with provided list of STD star by their ID """
    
    def get_radial_backround(img):
        ''' Computed STD star background as a minimum of radial profile '''
        from tiptorch.tools.utils import safe_centroid
        from photutils.profiles import RadialProfile
        
        xycen = safe_centroid(img)
        edge_radii = np.arange(img.shape[-1]//2)
        rp = RadialProfile(img, xycen, edge_radii)
        
        return rp.profile.min()
    
    def load_sample(id):
        ''' Loads individual sample '''
        sample = LoadSTDStarCacheByID(id)
  
        PSF_data = np.copy(sample['images']['cube']) 
        PSF_STD  = np.copy(sample['images']['std'])

        # Make sure that the image is centered in one pixel, so PSF peak has a chance to be in a single pixel
        if ensure_odd_pixels:
            if PSF_data.shape[-1] % 2 == 0:
                PSF_data = PSF_data[:,:-1,:-1]
                PSF_STD  = PSF_STD [:,:-1,:-1]
        
        if subtract_background:
            backgrounds = np.array([ get_radial_backround(PSF_data[i,:,:]) for i in range(PSF_data.shape[0]) ])[:,None,None]
            PSF_data -= backgrounds
            
        else:
            backgrounds = np.zeros(PSF_data.shape[0])[:,None,None]

        if normalize:
            norms     = PSF_data.sum(axis=(-1,-2), keepdims=True)
            PSF_data /= norms
            PSF_STD  /= norms
        else:
            norms = np.ones(PSF_data.shape[0])[:,None,None]

        config_dict = InitNFMConfig(sample, PSF_data, wvl_ids, convert_config=False, plotting=False)
        # Samples recorded with fewer corrected DM modes ('k_c mismatch'): the DM pitch is scaled so that the model's k_c matches theirs
        config_dict = apply_kc_fraction(config_dict, sample['All data'][KC_FRACTION_FEATURE] if KC_FRACTION_FEATURE in sample['All data'] else 1.0)

        if derotate_PSF:
            config_dict['telescope']['PupilAngle'] = 0.0 # Meaning, that the PSF is already derotated

        # Select a subset of wavelengths bins
        if wvl_ids is not None:
            PSF_data = PSF_data[wvl_ids,...]
            PSF_STD = PSF_STD [wvl_ids,...]
            norms = norms[wvl_ids,...]
            backgrounds = backgrounds[wvl_ids,...]
        
        return PSF_data, PSF_STD, norms, backgrounds, config_dict, sample


    PSFs, configs, norms, bgs = [], [], [], []
    
    if not isinstance(ids, list):
        ids = [ids]
    
    for id in ids:
        PSF_, _, norm, bg, config_dict_, sample_ = load_sample(id)
        
        if derotate_PSF:
            PSF_0_rot = RotatePSF(PSF_,  -sample_['All data']['Pupil angle'].item())
            # PSF_std  = RotatePSF(PSF_std, -sample_['All data']['Pupil angle'].item())
            PSFs.append(PSF_0_rot)
        else:
            PSFs.append(PSF_)
            
        configs.append(config_dict_)
        norms.append(norm)
        bgs.append(bg)

    PSFs  = torch.tensor(np.stack(PSFs),  dtype=default_torch_type, device=device)
    norms = torch.tensor(np.array(norms), dtype=default_torch_type, device=device)
    bgs   = torch.tensor(np.array(bgs),   dtype=default_torch_type, device=device)

    return PSFs, norms, bgs, configs 


def RenameMUSECubes(cubes_main_folder, folder_cubes_rename):
    files_to_rename = [file for file in os.listdir(folder_cubes_rename) if file != 'renamed']
    if len(files_to_rename) == 0:
        print(f'No files in: {folder_cubes_rename}')
        return
    
    '''Renames MUSE reduced cubes .fits files according to their exposure date and time'''
    original_cubes_exposure, new_cubes_exposure = [], []
    original_filename, new_filename = [], []

    print(f'Reading cubes in {folder_cubes_rename}')
    for file in tqdm(files_to_rename):
        
        with fits.open(os.path.join(folder_cubes_rename, file)) as hdul_cube:
            new_cubes_exposure.append(hdul_cube[0].header['DATE-OBS'])
            new_filename.append(file)

    print(f'Reading cubes in {cubes_main_folder}')
    for file in tqdm(os.listdir(cubes_main_folder)):
        with fits.open(os.path.join(cubes_main_folder, file)) as hdul_cube:
            original_cubes_exposure.append(hdul_cube[0].header['DATE-OBS'])
            original_filename.append(file)

    intersection = list(set(original_cubes_exposure).intersection(set(new_cubes_exposure)))

    # Remove files which intersect
    if len(intersection) > 0:
        for exposure in intersection:
            file = new_filename[new_cubes_exposure.index(exposure)]
            file_2_rm = os.path.normpath(os.path.join(folder_cubes_rename, file))
            print(f'Removed duplicate: {file_2_rm}')
            os.remove(file_2_rm)

    # Rename files according to the their exposure timestamps (just for convenience)
    renamed_dir = os.path.join(folder_cubes_rename, 'renamed')
    if not os.path.exists(renamed_dir):
        os.makedirs(renamed_dir)

    for file in tqdm(os.listdir(folder_cubes_rename)):
        # Skip the 'renamed' directory
        if file == 'renamed':
            continue

        with fits.open(os.path.join(folder_cubes_rename, file)) as hdul_cube:
            exposure = hdul_cube[0].header['DATE-OBS']

        new_name = 'M.MUSE.' + exposure.replace(':', '-') + '.fits'
        file_2_rm = os.path.normpath(os.path.join(folder_cubes_rename, file))
        file_2_mv = os.path.normpath(os.path.join(renamed_dir, new_name))

        # Check if destination file already exists
        if os.path.exists(file_2_mv):
            print(f"Warning: Duplicate file found for {exposure}. Removing {file_2_rm}")
            os.remove(file_2_rm)
        else:
            os.rename(file_2_rm, file_2_mv)

    return renamed_dir


def RenderDataSample(data_dict, file_name):
    from matplotlib.colors import LogNorm

    # img = 1 + np.abs(data_dict['images']['white'])
    img = 1 + np.abs(data_dict['images']['cube'][-5:,...].mean(axis=0))
    
    if data_dict['images']['IRLOS cube'] is not None:
        IRLOS_img = np.abs(data_dict['images']['IRLOS cube'].mean(axis=-1))
    else:
        IRLOS_img = np.zeros_like(img)

    title = os.path.basename(file_name).replace('.pickle', '')
    _, ax = plt.subplots(1,2, figsize=(14, 7.5))

    ax[0].set_title(title)
    ax[1].set_title('IRLOS (2x2)')

    # Compute smart log norm limits using percentiles
    vmin = np.percentile(img[img > 0], 20) if np.any(img > 0) else 1
    vmax = np.percentile(img[img > 0], 99.975) if np.any(img > 0) else img.max()
    ax[0].imshow(img, cmap='gray', norm=LogNorm(vmin=vmin, vmax=vmax))
    ax[1].imshow(IRLOS_img, cmap='hot')
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    plt.tight_layout()


def update_label_IDs():
    """ Matches the IDs in the labels file with the current samples indexing """
    from pathlib import Path
    import pandas as pd

    # Input: text file contents as provided by the user
    # Read text lines from file
    text_file_path = STD_FOLDER / 'labels.txt'
    with open(text_file_path, 'r') as f:
        text_lines = [line.strip() for line in f.readlines()]

    # Input: folder filenames as provided by the user
    folder_files = os.listdir(STD_FOLDER / "MUSE_images")

    def split_index_and_tail(filename_with_index: str):
        if '_' in filename_with_index:
            idx, tail = filename_with_index.split('_', 1)
            return idx, tail
        return None, filename_with_index

    # Build a lookup from tail -> index for the folder files
    tail_to_index = {}
    for f in folder_files:
        idx, tail = split_index_and_tail(f)
        tail_to_index[tail] = idx

    # Process text lines
    results = []
    updated_lines = []
    for line in text_lines:
        # split at ': ' to separate filename and labels
        if ': ' in line:
            name_part, labels = line.split(': ', 1)
        else:
            name_part, labels = line, ""

        old_idx, tail = split_index_and_tail(name_part)
        new_idx = tail_to_index.get(tail)  # match on tail

        if new_idx is not None:
            # Replace leading index with the matched one
            new_name = f"{new_idx}_{tail}"
            updated_line = f"{new_name}: {labels}" if labels else new_name
            changed = (new_idx != old_idx)
            matched = True
        else:
            # No match found; keep as-is
            new_name = name_part
            updated_line = line
            changed = False
            matched = False

        results.append({
            "original_name": name_part,
            "labels": labels,
            "matched_folder_file": matched,
            "new_name": new_name,
            "index_changed": changed,
        })
        updated_lines.append(updated_line)

    # Save updated text file
    out_path = STD_FOLDER / "updated_labels.txt"
    out_path.write_text("\n".join(updated_lines), encoding="utf-8")

    # Show a concise dataframe summary
    df = pd.DataFrame(results, columns=["original_name","new_name","matched_folder_file","index_changed","labels"])
    df.head()


KC_MISMATCH_FLAG     = 'k_c mismatch'
KC_FRACTION_FEATURE  = 'k_c fraction'

def read_kc_mismatch_labels(labels_file=None):
    """
    Parses `STD_FOLDER / 'kc_mismatch_labels.txt'`: one sample per line,
        <MUSE Filename>: k_c <percent> % (<source>), <class>, amp <ring amplitude>, <date>
    ('#' lines are comments). <source> is 'ring' (ring position in the residuals, the detector's first estimate, the default) or 'fit'
    (loss minimum of PSF fits scanning k_c, see kc_mismatch.py (--scan); never overwritten by the detector).
    Returns {Filename: {'kc_percent', 'kc_source', 'ring_class', 'amp', 'date'}}; missing fields are None.
    """
    import re
    labels_file = Path(labels_file) if labels_file is not None else STD_FOLDER / 'kc_mismatch_labels.txt'
    labels = {}
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            filename, _, tail = line.partition(':')
            m_kc  = re.search(r'k_c\s*([\d.]+)\s*%\s*(?:\((ring|fit)\))?', tail)
            m_amp = re.search(r'amp\s*(-?[\d.]+)', tail)
            m_cls = re.search(r'\b(strong|weak|manual|none)\b', tail)
            m_dat = re.search(r'(\d{4}-\d{2}-\d{2})', tail)
            labels[filename.strip()] = {
                'kc_percent': float(m_kc.group(1)) if m_kc else None,
                'kc_source':  (m_kc.group(2) or 'ring') if m_kc else None,
                'ring_class': m_cls.group(1) if m_cls else None,
                'amp':        float(m_amp.group(1)) if m_amp else None,
                'date':       m_dat.group(1) if m_dat else None,
            }
    return labels


def write_kc_mismatch_labels(labels, labels_file=None):
    """ Writes the labels dict (see read_kc_mismatch_labels) sorted by date; k_c percent is the ring position / nominal k_c. """
    labels_file = Path(labels_file) if labels_file is not None else STD_FOLDER / 'kc_mismatch_labels.txt'
    header = [
        '# MUSE NFM STD stars recorded with a DM correction radius (k_c) different from the model one (k_c = 1 / (2 DmPitch)).',
        '# Detected with datasets/MUSE_STD_dataset/kc_mismatch.py: azimuthally coherent ring in the data - fit residuals.',
        '# "k_c <p> % (ring)" is the ring position relative to the nominal correction radius (the detector estimate, biased high by a few %',
        '# because the fit compensates part of the mismatch); "k_c <p> % (fit)" is the loss minimum of PSF fits scanning k_c (kc_mismatch.py (--scan)),',
        '# the value to trust. "amp" is the ring amplitude relative to the local PSF level. Classes: strong / weak (automatic), manual (added by hand,',
        '# e.g. all samples of an affected observing period), none (checked by hand, NOT a mismatch: excluded from the flag and never re-added).',
        '# Keyed by MUSE filename, not by sample ID. Entries are never removed automatically.',
        '# Format:  <Filename>: k_c <percent> % (ring|fit), <class>, amp <value>, <date>',
    ]
    def fmt(v, spec):
        return format(v, spec) if v is not None else 'nan'
   
    with open(labels_file, 'w') as f:
        f.write('\n'.join(header) + '\n')
        for name, e in sorted(labels.items(), key=lambda kv: (kv[1].get('date') or '', kv[0])):
            f.write(f"{name}: k_c {fmt(e.get('kc_percent'), '.1f')} % ({e.get('kc_source') or 'ring'}), {e.get('ring_class') or 'manual'}, amp {fmt(e.get('amp'), '.2f')}, {e.get('date') or ''}\n")


KC_GROUP_FEATURE     = 'k_c group'
KC_GROUP_TOLERANCE   = 0.03  # max |sample k_c - group k_c| (fraction of the nominal k_c) when grouping the mismatched samples (the ring position scatters by ~1-2 %)
NOMINAL_DM_PITCH     = 0.22  # [m], the DmPitchs InitNFMConfig sets, k_c = 1 / (2 pitch)

def group_kc_fractions(kc_fractions, tolerance=KC_GROUP_TOLERANCE):
    """
    Groups the measured k_c fractions of the mismatched samples by the closest averaged value: the smallest number of 1-D k-means
    clusters such that every sample lies within `tolerance` of its cluster mean. The PSF model holds a single k_c per batch, so the
    samples of one group share the group's averaged k_c (see build_dataset_cache / NFMCalibratorTrainer).
    Returns (group ids [N] in 1..N_groups sorted by increasing k_c, group means [N_groups]).
    """
    x = np.asarray(kc_fractions, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0)
    for k in range(1, x.size + 1):
        centers = np.quantile(x, (np.arange(k) + 0.5) / k)
        for _ in range(100):
            assign = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
            new_centers = np.array([x[assign == j].mean() if np.any(assign == j) else centers[j] for j in range(k)])
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        if np.all(np.abs(x - centers[assign]) <= tolerance):
            break
    order = np.argsort(centers)
    rank  = np.empty_like(order); rank[order] = np.arange(k)
   
    return rank[assign] + 1, centers[order]


def apply_kc_fraction(config_dict, kc_fraction):
    """
    Sets the DM pitch of a (single-source, unconverted) PSF-model config so that the model's cut-off frequency k_c = 1 / (2 pitch)
    becomes kc_fraction x the nominal one: the samples recorded with fewer corrected DM modes have a smaller correction radius.
    """
    kc_fraction = None if kc_fraction is None else float(np.asarray(kc_fraction, dtype=float).ravel()[0])
    if kc_fraction is None or not np.isfinite(kc_fraction) or np.isclose(kc_fraction, 1.0):
        return config_dict
    pitch = np.asarray(config_dict['DM']['DmPitchs'], dtype=float)
    config_dict['DM']['DmPitchs'] = (pitch / float(kc_fraction)).reshape(pitch.shape).tolist() if isinstance(config_dict['DM']['DmPitchs'], list) else pitch / float(kc_fraction)
    
    return config_dict


def kc_fraction_from_config(config_dict, nominal_pitch=NOMINAL_DM_PITCH):
    """ k_c of a config relative to the nominal MUSE NFM one (the inverse of apply_kc_fraction) """
    pitch = config_dict['DM']['DmPitchs']
    pitch = float(pitch.min()) if hasattr(pitch, 'min') and not isinstance(pitch, list) else float(np.min(np.asarray(pitch, dtype=float)))
    
    return nominal_pitch / pitch


def add_kc_mismatch_flag(muse_df, labels_file=None, verbose=True):
    """
    Adds the boolean 'k_c mismatch' label and the 'k_c fraction' feature (real / model k_c, 1.0 when not flagged or unknown)
    to the STD-stars dataframe.

    The flagged samples were recorded with fewer DM modes corrected, so their real correction radius differs from the
    single k_c = 1 / (2 DmPitch) used by the PSF model. They were identified with `kc_mismatch.py` (azimuthally
    coherent ring at ~0.9 r_c in the data - fit residuals) and are listed by MUSE filename in
    `STD_FOLDER / 'kc_mismatch_labels.txt'` (see read_kc_mismatch_labels), so the label survives re-indexing of the samples.
    If the file does not exist yet (fits not done), all samples get False / 1.0 and the file is created by
    `update_kc_mismatch_labels` later in the dataset preparation.
    """
    labels_file = Path(labels_file) if labels_file is not None else STD_FOLDER / 'kc_mismatch_labels.txt'
    labels = read_kc_mismatch_labels(labels_file) if labels_file.exists() else {}
    if not labels_file.exists() and verbose:
        print(f'k_c mismatch labels file not found ({labels_file}): all samples are set to False / 1.0')

    labels = {name: e for name, e in labels.items() if e.get('ring_class') != 'none'} # 'none' = vetoed by hand
    # The mismatched samples are grouped by the closest averaged k_c (the PSF model holds one k_c per batch): every flagged sample gets
    # the averaged k_c of its group, the group id (1..N_groups by increasing k_c) and the nominal samples k_c fraction 1.0 / group 0
    names    = [n for n in labels if labels[n].get('kc_percent') is not None]
    measured = np.array([labels[n]['kc_percent'] / 100.0 for n in names])
    group_ids, group_means = group_kc_fractions(measured)
    kc_frac  = {n: float(group_means[g - 1]) for n, g in zip(names, group_ids)}
    kc_group = {n: int(g) for n, g in zip(names, group_ids)}

    muse_df[KC_MISMATCH_FLAG]    = muse_df['Filename'].isin(labels)
    muse_df[KC_FRACTION_FEATURE] = muse_df['Filename'].map(kc_frac).astype(float)
    muse_df[KC_GROUP_FEATURE]    = muse_df['Filename'].map(kc_group).fillna(0).astype(int)
    muse_df.loc[~muse_df[KC_MISMATCH_FLAG], KC_FRACTION_FEATURE] = 1.0
    muse_df.loc[~muse_df[KC_MISMATCH_FLAG], KC_GROUP_FEATURE]    = 0
    muse_df[KC_FRACTION_FEATURE] = muse_df[KC_FRACTION_FEATURE].fillna(1.0) # listed without a k_c measurement: flagged, nominal k_c

    if verbose:
        n_found = muse_df[KC_MISMATCH_FLAG].sum()
        missing = set(labels) - set(muse_df['Filename'])
        print(f'k_c mismatch flag: {n_found} / {len(labels)} listed samples found in the dataframe')
        
        for g, m in enumerate(group_means, start=1):
            n_g = int((muse_df[KC_GROUP_FEATURE] == g).sum())
            print(f'  k_c group {g}: {n_g} samples, k_c = {100 * m:.1f} % of the nominal one (range {100 * measured[group_ids == g].min():.1f} - {100 * measured[group_ids == g].max():.1f} %)')
       
        if missing:
            print(f'  Listed but not in the dataframe: {sorted(missing)}')
    
    return muse_df


def AOF_Cn2_profiles_stats(STD_stars_df, store=False):
    # Compute median Cn2 profile for MUSE NFM
    all_Cn2_alts, all_Cn2_fracs = [], []

    for idx in STD_stars_df.index:
        Cn2_alt_sample  = [STD_stars_df[f'ALT{i}'].loc[idx].item()          for i in range(1, 9)]
        Cn2_frac_sample = [STD_stars_df[f'CN2_FRAC_ALT{i}'].loc[idx].item() for i in range(1, 9)]
        all_Cn2_alts.append(Cn2_alt_sample)
        all_Cn2_fracs.append(Cn2_frac_sample)

    all_Cn2_alts  = np.array(all_Cn2_alts)
    all_Cn2_fracs = np.array(all_Cn2_fracs)

    # Filter out unreasonable values
    all_Cn2_alts [all_Cn2_alts > 100] = np.nan
    all_Cn2_alts [all_Cn2_alts < 0]   = np.nan
    all_Cn2_fracs[all_Cn2_fracs > 1]  = np.nan
    all_Cn2_fracs[all_Cn2_fracs < 0]  = np.nan
    all_Cn2_alts[:,-1][all_Cn2_alts[:,-1] < 1e-12] = np.nan

    # Compute median and percentiles (ignoring NaNs)
    median_Cn2_alts  = np.nanmedian(all_Cn2_alts, axis=0)
    p16_Cn2_alts     = np.nanpercentile(all_Cn2_alts, 16, axis=0)
    p84_Cn2_alts     = np.nanpercentile(all_Cn2_alts, 84, axis=0)

    median_Cn2_fracs = np.nanmedian(all_Cn2_fracs, axis=0)
    p16_Cn2_fracs    = np.nanpercentile(all_Cn2_fracs, 16, axis=0)
    p84_Cn2_fracs    = np.nanpercentile(all_Cn2_fracs, 84, axis=0)
    
    if store:
        import json
        Cn2_profile_dict = {
            'median_Cn2_alts':  median_Cn2_alts.tolist(),
            'median_Cn2_fracs': median_Cn2_fracs.tolist()
        }
        with open(DATA_FOLDER / 'reduced_telemetry/MUSE/AOF_median_Cn2_profile.json', 'w') as json_file:
            json.dump(Cn2_profile_dict, json_file)

    return (median_Cn2_alts, median_Cn2_fracs), (p16_Cn2_alts, p84_Cn2_fracs), (p16_Cn2_fracs, p84_Cn2_alts)


DEFAULT_CN2_BOUNDARIES_KM = (4.0, 13.0)

def store_Cn2_grid(STD_stars_df, boundaries_km=None, pin_ground_layer=None, store=True, verbose=True):
    """
    Fixed Cn2 grid of the dataset (data_processing.Cn2_profile.Cn2Grid): nodes = median AOF layer altitudes, model-layer boundaries
    in [km], median grid profile. Stored in data/reduced_telemetry/MUSE/AOF_Cn2_grid.json; the dataset cache and the calibrator
    bundles carry a copy. The boundaries are chosen with machine_learning/analysis/analyze_Cn2_binning.py; None keeps the ones of the stored
    grid (DEFAULT_CN2_BOUNDARIES_KM if there is none yet).
    """
    from data_processing.Cn2_profile import Cn2Grid, fit_Cn2_grid, ALT_COLS, FRAC_COLS

    grid_path = DATA_FOLDER / 'reduced_telemetry/MUSE/AOF_Cn2_grid.json'
    stored = Cn2Grid.load(grid_path) if grid_path.exists() else None
    
    if boundaries_km is None:
        boundaries_km = stored.boundaries if stored is not None else DEFAULT_CN2_BOUNDARIES_KM
    
    if pin_ground_layer is None:
        pin_ground_layer = stored.pin_ground_layer if stored is not None else True
   
    grid = fit_Cn2_grid(STD_stars_df[ALT_COLS].to_numpy(float), STD_stars_df[FRAC_COLS].to_numpy(float), boundaries_km, pin_ground_layer=pin_ground_layer)
    
    if verbose:
        w, h = grid.default_layers()
        print(f'{grid} | median grid profile {np.round(grid.median_fracs, 3).tolist()} | median layers: weights {np.round(w, 3).tolist()}, heights {np.round(h, 2).tolist()} km')
    
    if store:
        grid.save(grid_path)
   
    return grid


def fit_telemetry_preprocessor(muse_df, Cn2_binning=None, random_state=3, max_iter=200, verbose=False):
    """
    Calibrator telemetry features from the full MUSE telemetry dataframe: filter / reduce, standardize (StandardScaler ignores NaNs
    when fitting and preserves them, so no pre-filling that would shrink the std of the sparse features) and impute (IterativeImputer,
    imputed values clipped to the observed range of each feature; see benchmark_telemetry_imputation.py for the comparison of methods).
    Returns the pruned dataframe (physical units, NaNs), its standardized version, the standardized imputed and the imputed (physical
    units) dataframes, and the TelemetryPreprocessor (scaler + imputer merged) that goes into the dataset cache and the bundles.
    """
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import StandardScaler
    from data_processing.telemetry_preprocessor import TelemetryPreprocessor
    from data_processing.MUSE_data_utils import filter_dataframe, reduce_dataframe, CN2_BINNING

    Cn2_binning    = Cn2_binning or CN2_BINNING # must match the reduction InitNFMConfig uses for the configs
    muse_df_pruned = reduce_dataframe(filter_dataframe(muse_df.copy()), Cn2_binning=Cn2_binning)
    numeric_cols   = muse_df_pruned.select_dtypes(include="number").columns

    telemetry_scaler = StandardScaler().fit(muse_df_pruned[numeric_cols])
    muse_df_pruned_scaled = muse_df_pruned.copy()
    muse_df_pruned_scaled[numeric_cols] = telemetry_scaler.transform(muse_df_pruned[numeric_cols])

    iterative_imputer = IterativeImputer(
        max_iter     = max_iter,
        min_value    = muse_df_pruned_scaled[numeric_cols].min().values,
        max_value    = muse_df_pruned_scaled[numeric_cols].max().values,
        random_state = random_state,
        verbose      = (2 if verbose else 0)
    )
    imputed_data = iterative_imputer.fit_transform(muse_df_pruned_scaled[numeric_cols])
    muse_df_pruned_scaled_imputed = pd.DataFrame(imputed_data, index=muse_df_pruned_scaled.index, columns=numeric_cols)
   
    muse_df_pruned_imputed = pd.DataFrame(
        telemetry_scaler.inverse_transform(muse_df_pruned_scaled_imputed[numeric_cols]),
        index = muse_df_pruned.index,
        columns = numeric_cols
    )
    telemetry_preprocessor = TelemetryPreprocessor.from_sklearn(telemetry_scaler, iterative_imputer)
    
    assert np.allclose(
        telemetry_preprocessor.transform(muse_df_pruned[numeric_cols]),
        iterative_imputer.transform(muse_df_pruned_scaled[numeric_cols]), atol=1e-10
    ), 'TelemetryPreprocessor does not reproduce the sklearn scaler + imputer'

    if verbose:
        print(f'{len(numeric_cols)} telemetry features, {len(muse_df_pruned)} samples, imputer iterations: {iterative_imputer.n_iter_}')

    return muse_df_pruned, muse_df_pruned_scaled, muse_df_pruned_scaled_imputed, muse_df_pruned_imputed, telemetry_preprocessor


def build_dataset_cache(muse_df, reduced_telemetry, muse_fitted_df, telemetry_preprocessor_state, out_path, Cn2_grid_state=None, verbose=True, derotate_PSF=True):
    """
    Dataset cache of the NFM calibrator: normalized, background-subtracted PSF cubes [N, H, W, N_wvl] of the good samples
    with the standardized imputed telemetry, the per-sample PSF-model configs (InitNFMConfig, incl. the binned Cn2 profile), the fitted
    parameters, the time-grouped train / val split and the telemetry preprocessor / Cn2 grid states (copied into the calibrator bundles).
    With derotate_PSF the cubes are derotated and the configs get PupilAngle 0; otherwise the raw frames are stored with the header pupil
    angle in their configs, and the PSF model rotates its pupil / AO frame per sample instead (the fitted dx / dy of a derotated fit do not
    apply to the raw frames). Returns (good_ids, bad_ids).
    """
    muse_df = muse_df[(muse_df['Corrupted'] == False) & (muse_df['Bad quality'] == False)]
    muse_df = muse_df.loc[muse_fitted_df.index.values] # the same samples in the same order as the fitted parameters
    good_samples = muse_df.index.values

    PSF_cubes, configs, telemetry_records, fitted_values, good_ids, bad_ids = [], [], [], [], [], []
    
    for id in tqdm(good_samples, disable=not verbose):
        try:
            PSF_data, _, _, model_config = LoadSTDStarCache(
                ids = id,
                derotate_PSF = derotate_PSF,
                normalize = True,
                subtract_background = True,
                ensure_odd_pixels = True,
                device = torch.device('cpu')
            )
            telemetry_records.append(reduced_telemetry.loc[id].to_dict())
            fitted_values.append(muse_fitted_df.loc[id].to_dict())
            good_ids.append(id)
            PSF_cubes.append(np.moveaxis(PSF_data.squeeze(0).numpy(), 0, -1)) # [1 x N_wvl x H x W]  -->  [H x W x N_wvl]
            configs.append(model_config)
            
        except Exception as e:
            print(f'Error with id {id}: {e}')
            bad_ids.append(id)

    good_ids  = np.array(good_ids)
    PSF_cubes = np.stack(PSF_cubes, axis=0) # [N_samples x H x W x N_wvl]
    telemetry_records = np.stack(telemetry_records, axis=0)

    # Training/validation split: samples acquired close in time must not be shared between the subsets.
    # The real acquisition time is DATE-OBS from the cube header (muse_df['time']), the timestamp in the file name is not it
    acq_times = muse_df.loc[good_ids, 'time']
    train_idx, val_idx, time_groups = SplitByAcquisitionTime(acq_times, val_fraction=0.2, max_gap_hours=1.0, seed=42, verbose=verbose)

    torch.save({
        'PSF_cubes':           PSF_cubes,
        'telemetry':           telemetry_records,
        'sample_ids':          good_ids,
        'model_configs':       configs,
        'fitted_param_values': fitted_values,
        'acquisition_times':   acq_times.astype(str).values, # [N_samples], UTC
        'train_idx':           train_idx, # positions into sample_ids
        'val_idx':             val_idx,
        'time_groups':         time_groups, # [N_samples], samples with the same label are close in acquisition time
        'derotated':           derotate_PSF, # False: raw frames, the configs carry the per-sample pupil angles
        # Samples flagged 'k_c mismatch' (fewer corrected DM modes): k_c relative to the nominal one (their configs carry the scaled DM
        # pitch) and the k_c group (0 = nominal); the trainer batches the samples per group, since the PSF model holds one k_c per batch
        'kc_fraction':         muse_df.loc[good_ids, KC_FRACTION_FEATURE].to_numpy(float) if KC_FRACTION_FEATURE in muse_df.columns else np.ones(len(good_ids)),
        'kc_group':            muse_df.loc[good_ids, KC_GROUP_FEATURE].to_numpy(int) if KC_GROUP_FEATURE in muse_df.columns else np.zeros(len(good_ids), dtype=int),
        'telemetry_preprocessor': telemetry_preprocessor_state, # TelemetryPreprocessor.state_dict(), copied into calibrator bundles
        'Cn2_grid':            Cn2_grid_state, # Cn2Grid.state_dict() of the Cn2 features / configs, copied into calibrator bundles
    }, out_path)
   
    if verbose:
        print(f'Dataset cache saved -> {out_path} ({len(good_ids)} samples, {len(bad_ids)} failed)')
   
    return good_ids, bad_ids


# ----------------------------------------------------------------------------------------------------------------------------------------------
# Dataset-level priors of the fitted parameters that are poorly constrained by a blurry PSF (see STD_fitter.py): the per-sample fits of the
# blurry stars pull these towards the medians of the sharp stars instead of fitting them freely (the phase bump is pulled to 0 instead).
# ----------------------------------------------------------------------------------------------------------------------------------------------
FIT_PRIORS_FILE = STD_FOLDER / 'fitted_priors.pickle'
FIT_PRIOR_KEYS  = ['LO_coefs', 'wing_frac_ctrl', 'Jx_ctrl', 'Jy_ctrl', 'J_ctrl', 'J_angle']

def blurry_PSF_mask(muse_df):
    """ The 'blurry' class of STD_fitter.py: bad quality, or medium quality without the good-quality flag. """
    return muse_df['Bad quality'].astype(bool) | (muse_df['Medium quality'].astype(bool) & ~muse_df['Good quality'].astype(bool))


def compute_fit_priors(fit_folder, muse_df, out_path=FIT_PRIORS_FILE, keys=FIT_PRIOR_KEYS, store=True, verbose=True):
    """
    Medians over the SHARP stars (not blurry, not corrupted) of the fitted values `keys` found in the pickles of `fit_folder`.
    The phase bump (LO_coefs[0]) is set to 0: it is suppressed, not pinned. Returns the dict {key: median array, 'source': ..., 'N': ...}
    and stores it as `out_path` (read by STD_fitter.py through load_fit_priors).
    """
    fit_folder = Path(fit_folder)
    blurry = blurry_PSF_mask(muse_df)
    values = {k: [] for k in keys}
    N = 0
    for f in sorted(fit_folder.glob('*.pickle'), key=lambda p: int(p.stem)):
        id = int(f.stem)
        if id not in muse_df.index or bool(blurry.loc[id]) or bool(muse_df.loc[id, 'Corrupted']):
            continue
        with open(f, 'rb') as handle:
            d = pickle.load(handle)
        for k in keys:
            if k in d:
                values[k].append(np.asarray(d[k], dtype=float).ravel())
        N += 1
    priors = {k: np.median(np.stack(v), axis=0) for k, v in values.items() if len(v)}
    if 'LO_coefs' in priors:
        priors['LO_coefs'][0] = 0.0
    priors['source'] = str(fit_folder)
    priors['N'] = N
    if verbose:
        print(f"Fit priors from {N} sharp stars of {fit_folder}: " + ', '.join(f'{k} {np.round(v, 3).tolist()}' for k, v in priors.items() if isinstance(v, np.ndarray)))
    if store:
        with open(out_path, 'wb') as handle:
            pickle.dump(priors, handle)
        if verbose:
            print(f'Stored at {out_path}')
    return priors


def load_fit_priors(path=FIT_PRIORS_FILE):
    """ The priors stored by compute_fit_priors, or None when the file does not exist. """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def even_zernike_composite_ids(Z_mode_max):
    """
    Indices, in the composite LO basis [bump, Noll 4, Noll 5, ..., Noll Z_mode_max + 1], of the modes with an even radial order n (defocus,
    astigmatisms, spherical, ...): the ones whose joint sign flip leaves the PSF invariant (OPD(x) -> -OPD(-x) symmetry of a centro-symmetric pupil).
    """
    ids = []
    for k, j in enumerate(range(4, Z_mode_max + 2), start=1): # composite index k holds Noll index j
        n = int(np.ceil((-3 + np.sqrt(9 + 8 * (j - 1))) / 2)) # radial order of Noll index j
        if n % 2 == 0:
            ids.append(k)
    return ids
