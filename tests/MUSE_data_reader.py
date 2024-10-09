
# For NFM it's save to assume gound layer to be below 2 km
h_GL = 2000

Cn2_weights = np.array([sample['All data'][f'CN2_FRAC_ALT{i}'].item() for i in range(1, 9)])
altitudes   = np.array([sample['All data'][f'ALT{i}'].item() for i in range(1, 9)])*100 # in meters

Cn2_weights[Cn2_weights > 1] = 0.0
Cn2_weights[Cn2_weights < 0] = 0.0

Cn2_weights_GL = Cn2_weights[altitudes < h_GL]
altitudes_GL   = altitudes  [altitudes < h_GL]

GL_frac  = Cn2_weights_GL.sum()  # Ground layer fraction


config_file['NumberSources'] = 1

config_file['telescope']['TelescopeDiameter'] = 8.0
config_file['telescope']['ZenithAngle'] = [90.0 - all_data['Tel. altitude'].item()]
config_file['telescope']['Azimuth']     = [all_data['Tel. azimuth'].item()]
config_file['telescope']['PupilAngle']  = all_data['Pupil angle'].item()

config_file['atmosphere']['L0']  = [all_data['L0Tot'].item()]
            
config_file['atmosphere']['Seeing'] = [all_data['Seeing (header)'].item()]
config_file['atmosphere']['Cn2Weights'] = [[GL_frac, 1-GL_frac]]
config_file['atmosphere']['Cn2Heights'] = [[0, h_GL]]

# Here, the same winspeed and winddirection is used for both layers
# This is not particularly corect, but works as a simplification
config_file['atmosphere']['WindSpeed']     = [[all_data['Wind speed (header)'].item(),]*2]
config_file['atmosphere']['WindDirection'] = [[all_data['Wind dir (header)'].item(),]*2]
config_file['sources_science']['Wavelength'] = # List of simulated wavelengths

config_file['sources_LO']['Wavelength'] = (1215+1625)/2.0 * 1e-9 #J+H band

config_file['sensor_science']['PixelScale'] = all_data['Pixel scale (science)'].item()
config_file['sensor_science']['FieldOfView'] = # PSF size in pixels

LGS_ph = np.array([sample['All data'][f'LGS{i} photons, [photons/m^2/s]'].item() / 1240e3 for i in range(1,5)])
LGS_ph[LGS_ph < 1] = np.mean(LGS_ph)
LGS_ph = [LGS_ph.tolist()]
    
config_file['sensor_HO']['NumberPhotons']  = LGS_ph

IRLOS_ph_per_subap_per_frame = all_data['IRLOS photons (cube), [photons/s/m^2]'].item()
if IRLOS_ph_per_subap_per_frame is not None:
    IRLOS_ph_per_subap_per_frame /= all_data['frequency'].item() / 4

config_file['sensor_LO']['PixelScale']    = all_data['plate scale, [mas/pix]'].item()
config_file['sensor_LO']['NumberPhotons'] = [IRLOS_ph_per_subap_per_frame]
config_file['sensor_LO']['SigmaRON']      = all_data['RON, [e-]'].item()
config_file['sensor_LO']['Gain']          = [all_data['gain'].item()]

config_file['RTC']['SensorFrameRate_LO']  = [all_data['frequency'].item()]
