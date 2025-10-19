from data_processing.MUSE_data_utils import *
# from torch.utils.data import Dataset


STD_FOLDER     = MUSE_DATA_FOLDER / 'standart_stars/'
CUBES_FOLDER   = STD_FOLDER / 'cubes/'
RAW_FOLDER     = STD_FOLDER / 'raw/'
CUBES_CACHE    = STD_FOLDER / 'cached_cubes/'
# DATASET_FOLDER = STD_FOLDER / 'dataset/'


def LoadSTDStarCacheByID(id):
    ''' Searches a specific STD star by its ID in the list of cached cubes. '''
    with open(STD_FOLDER / 'muse_df.pickle', 'rb') as handle:
        muse_df = pickle.load(handle)

    file = muse_df.loc[muse_df.index == id]['Filename'].values[0]
    full_filename = CUBES_CACHE / f'{id}_{file}.pickle'

    with open(full_filename, 'rb') as handle:
        data_sample, _ = pickle.load(handle)
    
    data_sample['All data']['Pupil angle'] = muse_df.loc[id]['Pupil angle']
    return data_sample


def LoadSTDStarData(
    ids,
    derotate_PSF = False,
    normalize = True,
    subtract_background = False,
    wvl_ids = None,
    ensure_odd_pixels = False,
    device = device
):
    """ Loads data associated with provided list of STD star by their ID """
    
    def get_radial_backround(img):
        ''' Computed STD star background as a minimum of radial profile '''
        from tools.utils import safe_centroid
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
                PSF_STD  = PSF_STD[:,:-1,:-1]
        
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

        config_dict = InitNFMConfig(sample, PSF_data, wvl_ids, convert_config=False)

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

    # merged_config = MultipleTargetsInDifferentObservations(configs, device=device)

    # if derotate_PSF:
        # merged_config['telescope']['PupilAngle'] = 0.0 # Meaning, that the PSF is already derotated

    return PSFs, norms, bgs, configs #merged_config


def RenameMUSECubes(folder_cubes_old, folder_cubes_new):
    '''Renames MUSE reduced cubes .fits files according to their exposure date and time'''
    original_cubes_exposure, new_cubes_exposure = [], []
    original_filename, new_filename = [], []

    print(f'Reding cubes in {folder_cubes_new}')
    for file in tqdm(os.listdir(folder_cubes_new)):
        if file == 'renamed':
            continue
        
        with fits.open(os.path.join(folder_cubes_new, file)) as hdul_cube:
            new_cubes_exposure.append(hdul_cube[0].header['DATE-OBS'])
            new_filename.append(file)

    print(f'Reading cubes in {folder_cubes_old}')
    for file in tqdm(os.listdir(folder_cubes_old)):
        with fits.open(os.path.join(folder_cubes_old, file)) as hdul_cube:
            original_cubes_exposure.append(hdul_cube[0].header['DATE-OBS'])
            original_filename.append(file)

    intersection = list(set(original_cubes_exposure).intersection(set(new_cubes_exposure)))

    # Remove files which intersect
    if len(intersection) > 0:
        for exposure in intersection:
            file = new_filename[new_cubes_exposure.index(exposure)]
            file_2_rm = os.path.normpath(os.path.join(folder_cubes_new, file))
            print(f'Removed duplicate: {file_2_rm}')
            os.remove(file_2_rm)

    # Rename files according to the their exposure timestamps (just for convenience)
    renamed_dir = os.path.join(folder_cubes_new, 'renamed')
    if not os.path.exists(renamed_dir):
        os.makedirs(renamed_dir)

    for file in tqdm(os.listdir(folder_cubes_new)):
        # Skip the 'renamed' directory
        if file == 'renamed':
            continue

        with fits.open(os.path.join(folder_cubes_new, file)) as hdul_cube:
            exposure = hdul_cube[0].header['DATE-OBS']

        new_name = 'M.MUSE.' + exposure.replace(':', '-') + '.fits'
        file_2_rm = os.path.normpath(os.path.join(folder_cubes_new, file))
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

    white = 1 + np.abs(data_dict['images']['white'])
    if data_dict['images']['IRLOS cube'] is not None:
        IRLOS_img = np.abs(data_dict['images']['IRLOS cube'].mean(axis=-1))
    else:
        IRLOS_img = np.zeros_like(white)

    title = os.path.basename(file_name).replace('.pickle', '')
    _, ax = plt.subplots(1,2, figsize=(14, 7.5))

    ax[0].set_title(title)
    ax[1].set_title('IRLOS (2x2)')

    # Compute smart log norm limits using percentiles
    vmin = np.percentile(white[white > 0], 20) if np.any(white > 0) else 1
    vmax = np.percentile(white[white > 0], 99.975) if np.any(white > 0) else white.max()
    ax[0].imshow(white, cmap='gray', norm=LogNorm(vmin=vmin, vmax=vmax))
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
    
