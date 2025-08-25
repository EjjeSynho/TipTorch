from MUSE_data_utils import *

STD_FOLDER   = MUSE_DATA_FOLDER / 'standart_stars/'
CUBES_FOLDER = STD_FOLDER / 'cubes/'
RAW_FOLDER   = STD_FOLDER / 'raw/'
CUBES_CACHE  = STD_FOLDER / 'cached_cubes/'

# Wavelength bins used to bin MUSE NFM multispectral cubes
wvl_bins = np.array([
    478.   , 492.125, 506.25 , 520.375, 534.625, 548.75 , 562.875, 
    577.   , 606.   , 620.25 , 634.625, 648.875, 663.25 , 677.5  ,
    691.875, 706.125, 720.375, 734.75 , 749.   , 763.375, 777.625,
    792.   , 806.25 , 820.625, 834.875, 849.125, 863.5  , 877.75 ,
    892.125, 906.375, 920.75 , 935.
], dtype='float32')

# wvl_bins = np.array([
    # 478, 511, 544, 577, 606,
    # 639, 672, 705, 738, 771,
    # 804, 837, 870, 903, 935
# ], dtype='float32')


def LoadSTDStarCacheByID(id):
    ''' Searches a specific STD star by its ID in the list of cached cubes. '''
    with open(TELEMETRY_CACHE / 'MUSE/muse_df.pickle', 'rb') as handle:
        muse_df = pickle.load(handle)

    file = muse_df.loc[muse_df.index == id]['Filename'].values[0]
    full_filename = CUBES_CACHE / f'{id}_{file}.pickle'

    with open(full_filename, 'rb') as handle:
        data_sample = pickle.load(handle)
    
    data_sample['All data']['Pupil angle'] = muse_df.loc[id]['Pupil angle']
    return data_sample


def LoadSTDStarData(ids, derotate_PSF=False, normalize=True, subtract_background=False, device=device):
    def get_radial_backround(img):
        ''' Computed STD star background as a minimum of radial profile '''
        from tools.utils import safe_centroid
        from photutils.profiles import RadialProfile
        
        xycen = safe_centroid(img)
        edge_radii = np.arange(img.shape[-1]//2)
        rp = RadialProfile(img, xycen, edge_radii)
        
        return rp.profile.min()
    
    
    def load_sample(id):
        sample = LoadSTDStarCacheByID(id)
  
        PSF_data = np.copy(sample['images']['cube']) 
        PSF_STD  = np.copy(sample['images']['std'])
        
        if subtract_background:
            backgrounds = np.array([ get_radial_backround(PSF_data[i,:,:]) for i in range(PSF_data.shape[0]) ])[:,None,None]
            PSF_data -= backgrounds
            
        else:
            backgrounds = np.zeros(PSF_data.shape[0])[:,None,None]

        if normalize:
            norms = PSF_data.sum(axis=(-1,-2), keepdims=True)
            PSF_data /= norms
            PSF_STD  /= norms
        else:
            norms = np.ones(PSF_data.shape[0])[:,None,None]
    
        config_file, PSF_0 = GetConfig(sample, PSF_0, convert_config=False)
        return PSF_0, PSF_STD, norms, backgrounds, config_file, sample


    PSF_0, configs, norms, bgs = [], [], [], []
    
    for id in ids:
        PSF_0_, _, norm, bg, config_dict_, sample_ = load_sample(id)
        configs.append(config_dict_)
        if derotate_PSF:
            PSF_0_rot = RotatePSF(PSF_0_, -sample_['All data']['Pupil angle'].item())
            PSF_0.append(PSF_0_rot)
        else:
            PSF_0.append(PSF_0_)
            
        norms.append(norm)
        bgs.append(bg)

    PSF_0 = torch.tensor(np.vstack(PSF_0), dtype=default_torch_type, device=device)
    norms = torch.tensor(norms, dtype=default_torch_type, device=device)
    bgs   = torch.tensor(bgs, dtype=default_torch_type, device=device)

    config_manager = ConfigManager()
    merged_config  = config_manager.Merge(configs)

    config_manager.Convert(merged_config, framework='pytorch', device=device)

    merged_config['sources_science']['Wavelength'] = merged_config['sources_science']['Wavelength'][0]
    merged_config['sources_HO']['Height']          = merged_config['sources_HO']['Height'].unsqueeze(-1)
    merged_config['sources_HO']['Wavelength']      = merged_config['sources_HO']['Wavelength'].squeeze()
    merged_config['NumberSources'] = len(ids)
    
    if derotate_PSF:
        merged_config['telescope']['PupilAngle'] = 0.0 # Meaning, that the PSF is already derotated

    return PSF_0, norms, bgs, merged_config


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