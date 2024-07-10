from project_globals import MUSE_DATA_FOLDER, device
import pickle
import torch
import numpy as np
from tools.parameter_parser import ParameterParser
from tools.config_manager import ConfigManager

#%% My new process
def LoadMUSEsampleByID(id): # searches for the sample with the specified ID in
    with open(MUSE_DATA_FOLDER+'muse_df.pickle', 'rb') as handle:
        request_df = pickle.load(handle)

    file = request_df.loc[request_df.index == id]['Filename'].values[0]
    full_filename = MUSE_DATA_FOLDER + f'DATA_reduced/{id}_{file}.pickle'

    with open(full_filename, 'rb') as handle:
        data_sample = pickle.load(handle)   
    return data_sample


def LoadImages(sample):

    def get_bg(img):
        from tools.utils import safe_centroid
        from photutils.profiles import RadialProfile
        xycen = safe_centroid(img)
        edge_radii = np.arange(img.shape[-1]//2)
        rp = RadialProfile(img, xycen, edge_radii)
        return rp.profile.min()


    PSF_data = np.copy(sample['images']['cube'][:,1:,1:])
    # bgs = np.array([get_bg(PSF_0_[i,...]) for i in range(PSF_0_.shape[0])])
    # PSF_0_ = PSF_0_ - bgs[:,None,None]
    norms = PSF_data.sum(axis=(-1,-2), keepdims=True)
    PSF_data /= norms
    PSF_data = torch.tensor(PSF_data).float().unsqueeze(0).to(device)

    var_mask = sample['images']['std'][:,1:,1:]
    var_mask = np.clip(1 / var_mask, 0, 1e6)
    var_mask = var_mask / var_mask.max(axis=(-1,-2), keepdims=True)
    var_mask = torch.tensor(var_mask).float().unsqueeze(0).to(device)

    return PSF_data, var_mask, norms


# 'images'
# 'IRLOS data'
# 'LGS data'
# 'MUSE header data'
# 'Raw Cn2 data'
# 'Raw atm data'
# 'Raw ASM data'
# 'ASM data'
# 'MASS-DIMM data',
# 'DIMM data',
# 'SLODAR data',
# 'All data',
# 'spectral data'

# x0s = []
# PSF_1s = []

# Manage config files
# for wvl_id in tqdm(range(N_wvl_)):


def GetConfig(sample, PSF_data, wvl_id=None):

    wvls_ = [(sample['spectral data']['wvls binned']*1e-9).tolist()]

    # All wavelengths
    if wvl_id is None:
        wvls  = wvls_
        PSF_0 = PSF_data

    # Select one particular wavelength
    else:
        wvls = [wvls_[0][wvl_id]]
        PSF_0 = PSF_data[:,wvl_id,...].unsqueeze(1)
        # N_wvl = 1

    # wvls = [wvls_[0][0], wvls_[0][-1]]
    # N_wvl = 2
    # PSF_0 = torch.stack( [PSF_0_[:,0,...], PSF_0_[:,-1,...]], axis=1 )
  
    config_manager = ConfigManager()
    config_file    = ParameterParser('../data/parameter_files/muse_ltao.ini').params
    # merged_config  = config_manager.Merge([config_manager.Modify(config_file, sample, *config_loader()) for sample in data_samples])

    h_GL = 2000

    try:
        # For NFM it's save to assume gound layer to be below 2 km, for WFM it's lower than that
        Cn2_weights = np.array([sample['Raw Cn2 data'][f'CN2_FRAC_ALT{i}'].item() for i in range(1, 9)])
        altitudes   = np.array([sample['Raw Cn2 data'][f'ALT{i}'].item() for i in range(1, 9)])*100 # in meters

        Cn2_weights_GL = Cn2_weights[altitudes < h_GL]
        altitudes_GL   = altitudes  [altitudes < h_GL]

        GL_frac  = Cn2_weights_GL.sum()  # Ground layer fraction   
        Cn2_w_GL = np.interp(h_GL, altitudes, Cn2_weights)
    except:
        GL_frac = 0.9
        

    config_file['NumberSources'] = 1

    config_file['telescope']['TelescopeDiameter'] = 8.0
    config_file['telescope']['ZenithAngle'] = [90.0 - sample['MUSE header data']['Tel. altitude'].item()]
    config_file['telescope']['Azimuth']     = [sample['MUSE header data']['Tel. azimuth'].item()]
    config_file['telescope']['PupilAngle']  = sample['All data']['Pupil angle'].item()
    
    if sample['Raw Cn2 data'] is not None:
        config_file['atmosphere']['L0']  = [sample['Raw Cn2 data']['L0Tot'].item()]
    else:
        config_file['atmosphere']['L0']  = [config_file['atmosphere']['L0']]
        
    config_file['atmosphere']['Seeing'] = [sample['MUSE header data']['Seeing (header)'].item()]
    config_file['atmosphere']['Cn2Weights'] = [[GL_frac, 1-GL_frac]]
    config_file['atmosphere']['Cn2Heights'] = [[0, h_GL]]

    config_file['atmosphere']['WindSpeed']     = [[sample['MUSE header data']['Wind speed (header)'].item(),]*2]
    config_file['atmosphere']['WindDirection'] = [[sample['MUSE header data']['Wind dir (header)'].item(),]*2]
    config_file['sources_science']['Wavelength'] = wvls

    config_file['sources_LO']['Wavelength'] = (1215+1625)/2.0 * 1e-9

    config_file['sensor_science']['PixelScale'] = sample['MUSE header data']['Pixel scale (science)'].item()
    config_file['sensor_science']['FieldOfView'] = PSF_0.shape[-1]

    try:
        LGS_ph = np.array([sample['All data'][f'LGS{i} photons, [photons/m^2/s]'].item() / 1240e3 for i in range(1,5)])
        LGS_ph[LGS_ph < 1] = np.mean(LGS_ph)
        LGS_ph = [LGS_ph.tolist()]
    except:
        LGS_ph = [[2000,]*4]
        
    config_file['sensor_HO']['NumberPhotons'] = LGS_ph
    config_file['sensor_HO']['SizeLenslets']  = config_file['sensor_HO']['SizeLenslets'][0]
    # config_file['sensor_HO']['NoiseVariance'] = 4.5

    IRLOS_ph_per_subap_per_frame = sample['IRLOS data']['IRLOS photons (cube), [photons/s/m^2]'].item()
    if IRLOS_ph_per_subap_per_frame is not None:
        IRLOS_ph_per_subap_per_frame /= sample['IRLOS data']['frequency'].item() / 4

    config_file['sensor_LO']['PixelScale']    = sample['IRLOS data']['plate scale, [mas/pix]'].item()
    config_file['sensor_LO']['NumberPhotons'] = [IRLOS_ph_per_subap_per_frame]
    config_file['sensor_LO']['SigmaRON']      = sample['IRLOS data']['RON, [e-]'].item()
    config_file['sensor_LO']['Gain']          = [sample['IRLOS data']['gain'].item()]

    config_file['RTC']['SensorFrameRate_LO'] = [sample['IRLOS data']['frequency'].item()]
    config_file['RTC']['SensorFrameRate_HO'] = [config_file['RTC']['SensorFrameRate_HO']]

    config_file['RTC']['LoopDelaySteps_HO']  = [config_file['RTC']['LoopDelaySteps_HO']]
    config_file['RTC']['LoopGain_HO']        = [config_file['RTC']['LoopGain_HO']]

    # config_file['DM']['DmPitchs'] = [config_file['DM']['DmPitchs'][0]*1.25]
    config_file['DM']['DmPitchs'][0] = 0.22

    config_file['sensor_HO']['ClockRate'] = np.mean([config_file['sensor_HO']['ClockRate']])
    config_manager.Convert(config_file, framework='pytorch', device=device)

    return config_file, PSF_0


#% Fernandos's process
'''
# Load image
data_dir = path.normpath('C:/Users/akuznets/Data/MUSE/DATA_Fernando/')
listData = os.listdir(data_dir)
sample_id = 5
sample_name = listData[sample_id]
path_im = path.join(data_dir, sample_name)
angle = np.zeros([len(listData)])
angle[0] = -46
angle[5] = -44
angle = angle[sample_id]

data_cube = MUSEcube(path_im, crop_size=200, angle=angle)
im, _, wvl = data_cube.Layer(5)
obs_info = dict( data_cube.obs_info )

PSF_0 = torch.tensor(im).unsqueeze(0).unsqueeze(0).to(device)
PSF_0 /= PSF_0.sum(dim=(-1,-2), keepdim=True)


config_manager = ConfigManager()
config_file    = ParameterParser('../data/parameter_files/muse_ltao.ini').params

config_file['NumberSources'] = 1

config_file['telescope']['TelescopeDiameter'] = 8.0
config_file['telescope']['ZenithAngle'] = [90.0-obs_info['TELALT']]
config_file['telescope']['Azimuth']     = [obs_info['TELAZ']]

config_file['atmosphere']['Cn2Weights'] = [config_file['atmosphere']['Cn2Weights']]
config_file['atmosphere']['Cn2Heights'] = [config_file['atmosphere']['Cn2Heights']]

config_file['atmosphere']['Seeing']        = [obs_info['SPTSEEIN']]
config_file['atmosphere']['L0']            = [obs_info['SPTL0']]
config_file['atmosphere']['WindSpeed']     = [[obs_info['WINDSP'],] * 2]
config_file['atmosphere']['WindDirection'] = [[obs_info['WINDIR'],] * 2]

config_file['sensor_science']['Zenith']      = [90.0-obs_info['TELALT']]
config_file['sensor_science']['Azimuth']     = [obs_info['TELAZ']]
config_file['sensor_science']['PixelScale']  = 25
config_file['sensor_science']['FieldOfView'] = im.shape[0]

config_file['sources_science']['Wavelength'] = [wvl]

config_file['sources_LO']['Wavelength'] = (1215+1625)/2.0 * 1e-9

config_file['sensor_HO']['NoiseVariance'] = 4.5
config_file['sensor_HO']['SizeLenslets']  = config_file['sensor_HO']['SizeLenslets'][0]
# config_file['sensor_HO']['NumberPhotons'] = [[200,]*4]
config_file['sensor_HO']['ClockRate'] = np.mean([config_file['sensor_HO']['ClockRate']])

config_file['RTC']['SensorFrameRate_HO'] = [config_file['RTC']['SensorFrameRate_HO']]
config_file['RTC']['LoopDelaySteps_HO']  = [config_file['RTC']['LoopDelaySteps_HO']]
config_file['RTC']['LoopGain_HO']        = [config_file['RTC']['LoopGain_HO']]

config_manager.Convert(config_file, framework='pytorch', device=device)
'''