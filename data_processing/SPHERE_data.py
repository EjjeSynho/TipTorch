#%%
import sys
sys.path.insert(0, '..')

import os
import re
from os import path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from tools.utils import GetROIaroundMax, mask_circle
from scipy.ndimage.measurements import center_of_mass
try:
    from query_eso_archive import query_simbad
except:
    pass

from project_globals import SPHERE_DATA_FOLDER

ROOT = Path(SPHERE_DATA_FOLDER+'IRDIS_RAW')
path_dtts = Path(SPHERE_DATA_FOLDER+'DTTS/')

folder_L = ROOT/'SPHERE_DC_DATA_LEFT/'
folder_R = ROOT/'SPHERE_DC_DATA_RIGHT/'

path_output = SPHERE_DATA_FOLDER+'IRDIS_reduced/'

h = 'HIERARCH ESO '

#%%
def GetFilesList():
    files_L = []
    files_R = []

    if os.path.isdir(folder_L) == False:
        print('Folder does not exist: ', folder_L)
        # return None, None
    if os.path.isdir(folder_R) == False:
        print('Folder does not exist: ', folder_R)
        # return None, None

    def list_files(path):
        file_list = []
        for root, _, files in os.walk(path):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    files_L = list_files(folder_L)
    files_L_only = []
    files_R = []

    # Scanning for matchjng files
    pure_filename = lambda x: re.search(r'([^\\\/]*)(_left|_right)?\.fits$', str(x)).group(1).replace('_left','').replace('_right','')

    for i,file_L in enumerate(files_L):
        file_R = file_L.replace('_left', '_right').replace('_LEFT', '_RIGHT')
        if os.path.isfile(file_R) == False:
            print('File not found: ', file_R)
            files_L_only.append(files_L.pop(i))
        else:
            files_R.append(file_R)

    return files_L, files_R

files_L, files_R = GetFilesList()

#%%
class SPHERE_loader():
    
    def ReadImageCube(self, hdr, crop_size=256):
        cube = hdr[0].data
        mask_center = mask_circle(cube.shape[1], 128, center=(0,0), centered=True)
        # Get center of PSF
        _, ids, _ = GetROIaroundMax(cube.sum(axis=0)*mask_center, win=crop_size)
        # Crop image
        cube = cube[:, ids[0], ids[1]]
        # Measure the background noise
        mask_center = 1.0 - mask_circle(cube.shape[1], 64, center=(0,0), centered=True)
        mask_center[mask_center == 0] = np.nan
        # Filter out the noise
        noise_level = np.nanmedian(cube*mask_center)
        cube -= noise_level
        return cube


    def get_telescope_pointing(self):
        TELAZ  = float(self.hdr[0].header[h+'TEL AZ'])
        TELALT = float(self.hdr[0].header[h+'TEL ALT'])
        airmass = 0.5*(float(self.hdr[0].header[h+'TEL AIRM END']) + float(self.hdr[0].header[h+'TEL AIRM START']))
        return TELAZ, TELALT, airmass


    def get_ambi_parameters(self):
        tau0 = float(self.hdr[0].header[h+'TEL AMBI TAU0'])
        wDir = float(self.hdr[0].header[h+'TEL AMBI WINDDIR'])
        wSpeed = float(self.hdr[0].header[h+'TEL AMBI WINDSP'])
        RHUM  = float(self.hdr[0].header[h+'TEL AMBI RHUM'])
        pressure = 0.5*(float(self.hdr[0].header[h+'TEL AMBI PRES START']) + float(self.hdr[0].header[h+'TEL AMBI PRES END']))
        try:
            fwhm_linobs = float(self.hdr[0].header[h+'TEL IA FWHMLINOBS'])
        except:
            fwhm_linobs = np.nan
        return tau0, wDir, wSpeed, RHUM, pressure, fwhm_linobs


    def get_detector_config(self):
        psInMas = float(self.hdr['PIXSCAL']) if 'PIXSCAL' in self.hdr else 12.27

        # http://www.eso.org/observing/dfo/quality/SPHERE/reports/HEALTH/trend_report_IRDIS_DARK_ron_HC.html
        DIT  = self.hdr[0].header[h+'DET SEQ1 DIT']
        NDIT = self.hdr[0].header[h+'DET NDIT']
        try:
            NDSKIP = self.hdr[0].header[h+'DET NDSKIP']
        except:
            NDSKIP = 0

        EXPTIME = self.hdr[0].header[h+'DET SEQ1 EXPTIME']

        gain = float(self.hdr[0].header[h+'DET CHIP1 GAIN']) #IRDIS gain
        ron  = float(self.hdr[0].header[h+'DET CHIP1 RON'])

        if ron == 0:
            ron = 10.0 if DIT < 1.0 else 4.0   # e- rms/pix/readout 

        return psInMas, gain, ron, DIT, NDIT, NDSKIP, EXPTIME


    def get_star_coordinates(self):
        return self.hdr[0].header[h+'OBS NAME'], float(self.hdr[0].header['RA']), float(self.hdr[0].header['DEC'])


    def get_star_magnitudes(self):
        RA = DEC = np.nan
        OB_NAME, RA, DEC = self.get_star_coordinates()

        # Query SIMBAD
        dict_SIMBAD = query_simbad(Time(self.hdr[0].header['DATE-OBS']), SkyCoord(RA*u.degree, DEC*u.degree), name=OB_NAME)

        if dict_SIMBAD is None:
            return np.nan, OB_NAME, RA, DEC

        # Get magnitudes
        magnitudes = {}
        if 'simbad_FLUX_V' in dict_SIMBAD: magnitudes['V'] = dict_SIMBAD['simbad_FLUX_V'],
        if 'simbad_FLUX_R' in dict_SIMBAD: magnitudes['R'] = dict_SIMBAD['simbad_FLUX_R'],
        if 'simbad_FLUX_G' in dict_SIMBAD: magnitudes['G'] = dict_SIMBAD['simbad_FLUX_G'],
        if 'simbad_FLUX_J' in dict_SIMBAD: magnitudes['J'] = dict_SIMBAD['simbad_FLUX_J'],
        if 'simbad_FLUX_H' in dict_SIMBAD: magnitudes['H'] = dict_SIMBAD['simbad_FLUX_H'],
        if 'simbad_FLUX_K' in dict_SIMBAD: magnitudes['K'] = dict_SIMBAD['simbad_FLUX_K']

        return magnitudes, OB_NAME, RA, DEC


    def GetTelemetryDTTS(self):
        # Extract date and time from filename
        night_current = pd.to_datetime(re.search(r'(\d{4}-\d{2}-\d{2})', str(self.filename_L)).group(1))
        night_next = night_current + pd.DateOffset(days=1)
        night_prev = night_current - pd.DateOffset(days=1)

        # Add the adjuscent folders
        path_sub = [path_dtts.joinpath(night_current.strftime('%Y-%m-%d')),
                    path_dtts.joinpath(night_prev.strftime('%Y-%m-%d')),
                    path_dtts.joinpath(night_next.strftime('%Y-%m-%d'))]

        # Initialize internal functions
        def table_from_files(path_sub, search_name, condition=lambda x: True):
            df = pd.DataFrame()
            for current_path in path_sub:
                if current_path.exists():
                    tmp = [file for file in os.listdir(current_path) if search_name in file and condition(search_name)]
                    if len(tmp) > 0:
                        df_buf = pd.read_csv(current_path/tmp[0])
                        df = pd.concat([df, df_buf])
            if not df.empty:
                df.sort_values(by='date', inplace=True)
                df.drop_duplicates(subset=['date'], inplace=True)
                df.reset_index(inplace=True, drop=True)
                return df
            else:
                return None


        def find_closest(df, timestamp, T_exp):
            df["date"] = pd.to_datetime(df["date"])
            timestamp_start = timestamp - pd.DateOffset(seconds=1)
            timestamp_end   = timestamp + pd.DateOffset(seconds=T_exp)
            selection = df.loc[(df['date'] >= timestamp_start) & (df['date'] <= timestamp_end)]

            if selection.empty:
                def nearest(items, pivot):
                    return min(items, key=lambda x: abs(x - pivot))
                
                nearest_date = nearest(df['date'], timestamp)

                if abs(nearest_date - timestamp).total_seconds() < 60*2:
                    id_start = df.index[df['date'] == nearest_date][0]
                    id_end   = id_start
                else:
                    id_start = None
                    id_end   = None
            else:
                id_start  = selection.index[0]
                id_end    = selection.index[-1]
                
            return id_start, id_end

        # Initialize the match tables for entries <-> telemetry
        entries_DTTS = [('n_ADU', 'flux_IRLoop_ADU')]

        # sparta_visible_WFS
        entries_spart_vis = [
            ('n_ph',       'flux_VisLoop[#photons/subaperture/frame]'),
            ('WFS flux',   'flux_VisLoop[#photons/s]'),
            # ('Jitter X',   'Jitter X (avg)'),
            # ('Jitter Y',   'Jitter Y (avg)'),
            ('Jitter X',   'Jitter X (std)'),
            ('Jitter Y',   'Jitter Y (std)'),
            ('Focus',      'Focus (avg)'),
            ('ITTM pos (avg)', 'ITTM position (avg)'),
            ('ITTM pos (std)', 'ITTM position (std)'),
            ('DM pos (avg)',   'DM position (avg)'),
            ('DM pos (std)',   'DM position (std)'),     
            ('rate',       'WFS frequency'),
            ('WFS filter', 'spectral_filter'),
            ('WFS gain',   'WFS gain')]

        # sparta_atmospheric_params
        entries_spart_atm = [
            ('r0',        'r0_los_sparta'),
            ('seeing',    'seeing_los_sparta'),
            ('tau0',      'tau0_los_sparta'),
            ('windspeed', 'wind_speed_los_sparta'),
            ('SR',        'strehl_sparta')]

        # mass_dimm
        entries_mass_dimm = [
            ('MASS_tau0',           'MASS_tau0'),
            ('MASSDIMM_seeing',     'MASS-DIMM_seeing'),
            ('MASSDIMM_tau0',       'MASS-DIMM_tau0'),
            ('MASSDIMM_turb_speed', 'MASS-DIMM_turb_speed'),
            ('temperature',         'air_temperature_30m[deg]'),
            ('winddir',             'winddir_30m'),
            ('windspeed',           'windspeed_30m')]

        # dimm
        entries_dimm = [
            ('DIMM_seeing', 'dimm_seeing')]

        # asm
        entries_asm = [
            ('temperature', 'air_temperature_30m[deg]'),
            ('winddir',     'winddir_30m'),
            ('windspeed',   'windspeed_30m')]
        
        #ecmwf
        entries_ecmwf = [
            ('temperature (200 mbar)', 'temperature (degrees)'),
            ('winddir (200 mbar)',     'ecmwf_200mbar_winddir[deg]'),
            ('windspeed (200 mbar)',   'ecmwf_200mbar_windspeed[m/s]')]

        def get_entry_value(df, entry, id_start, id_end):
            if entry in df and id_start is not None and id_end is not None:
                # In case the entry has a text format
                if isinstance(df[entry][id_start:id_end+1].values[0], str):
                    id = int(np.round(id_start/2+id_end/2))
                    return df[entry][id:id+1].values[0]
                else:
                    # If it's a number, it's possible to read 
                    value   = np.nanmedian(df[entry][id_start:id_end+1])
                    std_dev = np.nanstd   (df[entry][id_start:id_end+1])
                    if not np.isreal(value) or np.isinf(value):
                        return np.nan
                    else:
                        return value + 1j*std_dev if std_dev > 1e-12 else value
            else:
                return np.nan
            

        def fill_dictionary(dictionary, df, entries_table, date, T_exp):
            if df is not None:
                id_min, id_max = find_closest(df, date, T_exp)
                for entry in entries_table:
                    dictionary[entry[0]] = get_entry_value(df, entry[1], id_min, id_max)
            else:
                for entry in entries_table:
                    dictionary[entry[0]] = np.nan

        dimm_df      =  table_from_files(path_sub, 'dimm', condition=lambda x: x[0]=='d')
        dtts_flux_df  = table_from_files(path_sub, 'sparta_IR_DTTS')
        sparta_vis_df = table_from_files(path_sub, 'sparta_visible_WFS')
        sparta_atm_df = table_from_files(path_sub, 'sparta_atmospheric_params')
        mass_dimm_df  = table_from_files(path_sub, 'mass_dimm', condition=lambda x: x[0]=='m')
        asm_df        = table_from_files(path_sub, 'asm')
        ecmwf_df      = table_from_files(path_sub, 'ecmwf')

        ECMWF_data = {}
        SPARTA_data = {}
        MASSDIMM_data = {}

        # Start of the exposure
        timestamp = pd.to_datetime(self.hdr[0].header[h+'DET SEQ UTC'])
        # Exposure time
        T_exp = self.hdr[0].header[h+'DET SEQ1 EXPTIME']

        fill_dictionary(SPARTA_data,   dtts_flux_df , entries_DTTS,      timestamp, T_exp)
        fill_dictionary(SPARTA_data,   sparta_vis_df, entries_spart_vis, timestamp, T_exp)
        fill_dictionary(SPARTA_data,   sparta_atm_df, entries_spart_atm, timestamp, T_exp)
        fill_dictionary(MASSDIMM_data, mass_dimm_df,  entries_mass_dimm, timestamp, T_exp)
        fill_dictionary(MASSDIMM_data, dimm_df,       entries_dimm,      timestamp, T_exp)
        fill_dictionary(MASSDIMM_data, asm_df,        entries_asm,       timestamp, T_exp)
        fill_dictionary(ECMWF_data,    ecmwf_df,      entries_ecmwf,     timestamp, T_exp)

        # Get the closest available DTTS images
        try:
            tables = pd.DataFrame()
            DTTS_PSFs = []

            for current_path in path_sub:
                if current_path.exists():
                    tmp = [file for file in os.listdir(current_path) if "DTTS_cube" in file]
                    if len(tmp) > 0:
                        DTTS_csv  = [file for file in tmp if ".csv"  in file][0]
                        DTTS_fits = [file for file in tmp if ".fits" in file][0]

                        tables = pd.concat([tables, pd.read_csv(current_path/DTTS_csv)])

                        with fits.open(current_path/DTTS_fits) as hdul:
                            for layer in np.split(hdul[0].data, hdul[0].data.shape[0], axis=0):
                                DTTS_PSFs.append(layer.squeeze(0).tolist())
                                
            tables.rename(columns = {tables.columns[0]: "id"}, inplace=True)
            tables['PSF'] = DTTS_PSFs
            tables.sort_values(by='date', inplace=True)
            tables.drop_duplicates(subset=['date'], inplace=True)
            # tables.reset_index(inplace=True, drop=True)
            cube_array = np.array( [np.array(tables.iloc[i]['PSF']) for i in range(*find_closest(tables, timestamp, T_exp))] )
        except:
            cube_array = np.nan

        #  Get the WFSing spectrum
        if    SPARTA_data['WFS filter'] == 'OPEN':   SPARTA_data['WFS wavelength'] = 658 #[nm]
        elif  SPARTA_data['WFS filter'] == 'LP_780': SPARTA_data['WFS wavelength'] = 780 #[nm]
        else: SPARTA_data['WFS wavelength'] = np.nan

        return SPARTA_data, MASSDIMM_data, ECMWF_data, cube_array


    def GetSpectrum(self):
        def get_filter_transmission(filter_name): #TODO: fix the path
            root_filters = '../data/calibrations/VLT_CALIBRATION/IRDIS_filters/'
            filename = root_filters + 'SPHERE_IRDIS_' + filter_name + '.dat'
            # set up an empty array to store the data
            data = []
            # open the file for reading

            try:
                with open(filename, "r") as file:
                    for line in file:
                        # strip any leading/trailing whitespace
                        line = line.strip()
                        # skip any lines that don't start with "#"
                        if not line.startswith("#"):
                            entry = [np.array(float(x)) for x in line.split()]
                            data.append(entry)
                return np.array(data)
            except:
                if filter_name in ['P0-90', 'P45-135']:
                    return np.array(0.5)
                elif filter_name == 'CLEAR':
                    return np.array(1)
                else:
                    return np.nan

        def get_filter_wavelength(filter_name):
            if filter_name in ['CLEAR', 'P0-90', 'P45-135']:
                return np.nan
            
            filter_data = {
                'B_Y':      [1043, 140],
                'B_J':      [1245, 240],
                'B_H':      [1625, 290],
                'B_Ks':     [2182, 300],
                'N_HeI':    [1085, 14],
                'N_CntJ':   [1213, 17],
                'N_PaB':    [1283, 18],
                'N_CntH':   [1573, 23],
                'N_FeII':   [1642, 24],
                'N_CntK1':  [2091, 34],
                'N_H2':     [2124, 31],
                'N_BrG':    [2170, 31],
                'N_CntK2':  [2266, 32],
                'N_CO':     [2290, 33],
                'D_Y23':    [1022, 1076, 49,  50],
                'D_J23':    [1190, 1273, 42,  46],
                'D_H23':    [1593, 1667, 52,  54],
                'D_ND-H23': [1593, 1667, 52,  54],
                'D_H34':    [1667, 1733, 54,  57],
                'D_K12':    [2110, 2251, 102, 109]
            }

            if len(filter_data[filter_name]) == 4:
                wvl1, wvl2, bw1, bw2 = filter_data[filter_name]
                return wvl1, bw1, wvl2, bw2
            elif len(filter_data[filter_name]) == 2:
                wvl, bw = filter_data[filter_name]
                return wvl, bw

        filter_name_common = self.hdr[0].header[h+'INS1 FILT NAME']
        filter_name_LR     = self.hdr[0].header[h+'INS1 OPTI2 NAME']

        transmission_filt1 = get_filter_transmission(filter_name_common)
        transmission_opti2 = get_filter_transmission(filter_name_LR)
        wavelength_range   = transmission_filt1[:,0]

        if transmission_opti2.ndim == 0:
            wavelength_range_  = np.copy(wavelength_range)
            buf = transmission_opti2 * np.ones_like(wavelength_range)
            transmission_opti2 = np.vstack([wavelength_range, buf, buf]).T
        else:
            wavelength_range_  = transmission_opti2[:,0]
            if np.sum(np.abs(wavelength_range - wavelength_range_)) > 0.1:
                raise ValueError('Error: wavelength ranges of the filters are different')

        # Caclulate the transmission after two filters
        transmission_L = transmission_filt1[:,1] * transmission_opti2[:,1]
        transmission_R = transmission_filt1[:,1] * transmission_opti2[:,2]

        # calculate the center of mass of the filter transmission
        wvl1_L_c = wavelength_range[np.round(center_of_mass(transmission_L)).astype('uint').item()]
        wvl1_R_c = wavelength_range[np.round(center_of_mass(transmission_R)).astype('uint').item()]

        if filter_name_LR in ['CLEAR','P0-90', 'P45-135']:
            wvl, bw = get_filter_wavelength(filter_name_common)
            wvl1_L, bw_L, wvl1_R, bw_R = (wvl, bw, wvl, bw)
        else:
            wvl1_L, bw_L, wvl1_R, bw_R = get_filter_wavelength(filter_name_LR)

        dl = 7 # cut off the edges of the filter transmission
        wvl0_L, wvl2_L = (wvl1_L_c-bw_L/2+dl, wvl1_L_c+bw_L/2-dl)
        wvl0_R, wvl2_R = (wvl1_R_c-bw_R/2+dl, wvl1_R_c+bw_R/2-dl)

        mask_L = (wavelength_range > wvl0_L) & (wavelength_range < wvl2_L)
        mask_R = (wavelength_range > wvl0_R) & (wavelength_range < wvl2_R)

        spectrum = {
            'filter common': filter_name_common,
            'filter LR':     filter_name_LR,
            'range L':       wavelength_range[mask_L],
            'range R':       wavelength_range[mask_R],
            'spectrum L':    transmission_L[mask_L],
            'spectrum R':    transmission_R[mask_R],
            'central L':     wvl1_L,
            'central R':     wvl1_R,
        }
        return spectrum


    def LoadObservationData(self):
        TELAZ, TELALT, airmass = self.get_telescope_pointing()
        magnitudes, OB_NAME, RA, DEC = self.get_star_magnitudes()
        tau0, wDir, wSpeed, RHUM, pressure, fwhm = self.get_ambi_parameters()
        psInMas, gain, ron, DIT, NDIT, NDSKIP, exp_time = self.get_detector_config()
        
        SPARTA_data, MASSDIMM_data, ECMWF_data, cube_array = self.GetTelemetryDTTS()
        spectrum = self.GetSpectrum()

        data = {
            'r0': SPARTA_data['r0'],
            'FWHM': fwhm,
            'Strehl': SPARTA_data['SR'],
            'turb speed': MASSDIMM_data['MASSDIMM_turb_speed'],

            'DTTS': cube_array,
            'spectra': spectrum,

            'tau0': {
                'header': tau0,
                'SPARTA': SPARTA_data['tau0'],
                'MASSDIMM': MASSDIMM_data['MASSDIMM_tau0'],
                'MASS': MASSDIMM_data['MASS_tau0']
            },

            'seeing': {
                'SPARTA': SPARTA_data['seeing'],
                'MASSDIMM': MASSDIMM_data['MASSDIMM_seeing'],
                'DIMM': MASSDIMM_data['DIMM_seeing']
            },

            'Wind speed': {
                'header': wSpeed,
                'SPARTA': SPARTA_data['windspeed'],
                'MASSDIMM': MASSDIMM_data['windspeed'],
                '200 mbar': ECMWF_data['windspeed (200 mbar)']
            },

            'Wind direction': {
                'header': wDir,
                'MASSDIMM': MASSDIMM_data['winddir'],
                '200 mbar': ECMWF_data['winddir (200 mbar)']
            },

            'Environment': {
                'pressure': pressure,
                'humidity': RHUM,
                'temperature': MASSDIMM_data['temperature'],
                'temperature (200 mbar)': ECMWF_data['temperature (200 mbar)']
            },

            'Detector': {
                'psInMas': psInMas,
                'gain': gain,
                'ron': ron
            },

            'Integration': {
                'DIT': DIT,
                'Num. DITs': NDIT,
                'NDSKIP': NDSKIP,
                'sequence time': exp_time
            },

            'WFS': {
                'Nph vis':        SPARTA_data['n_ph'],
                'Flux vis':       SPARTA_data['WFS flux'],
                'TT jitter X':    SPARTA_data['Jitter X'],
                'TT jitter Y':    SPARTA_data['Jitter Y'],
                'Focus':          SPARTA_data['Focus'],
                'ITTM pos (avg)': SPARTA_data['ITTM pos (avg)'],
                'ITTM pos (std)': SPARTA_data['ITTM pos (std)'],
                'DM pos (avg)':   SPARTA_data['DM pos (avg)'],
                'DM pos (std)':   SPARTA_data['DM pos (std)'],
                'rate':           SPARTA_data['rate'],
                'filter':         SPARTA_data['WFS filter'],
                'wavelength':     SPARTA_data['WFS wavelength']*1e-9,
                'gain':           SPARTA_data['WFS gain'] 
            },

            'observation': {
                'date': self.hdr[0].header['DATE-OBS'],
                'name': OB_NAME,
                'RA': RA,
                'DEC': DEC,
                'magnitudes': magnitudes
            },

            'telescope': {
                'azimuth': TELAZ,
                'altitude': TELALT,
                'airmass': airmass
            }
        }
        return data


    def load(self, fits_L, fits_R):
        self.filename_L, self.filename_R = (fits_L, fits_R)
        with fits.open(self.filename_L) as hdr_L:
            with fits.open(self.filename_R) as hdr_R:
                self.hdr = hdr_L # This header is used to extract the telemetry data
                self.data = self.LoadObservationData()
                self.data['PSF L'] = self.ReadImageCube(hdr_L)
                self.data['PSF R'] = self.ReadImageCube(hdr_R)
        return self.data


    def save(self, path, prefix=''):
        path = os.path.normpath(path)
        if os.path.exists(path):
            file = os.path.basename(self.filename_L).replace('.fits','').replace('_left','')
            path_new = os.path.join(path, prefix+file+'.pickle')
            if not os.path.exists(path_new):
                print('Saving to: ', path_new)
                with open(path_new, 'wb') as handle:
                    pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print('File already exists: ', path_new)
        else:
            raise ValueError('Wrong path is specified!')
        

    def __init__(self, path_dtts) -> None:
        self.data = {}
        self.path_dtts = path_dtts

#%%
# Load the data
def plot_sample(id):
    loader = SPHERE_loader(path_dtts)
    samp = loader.load(files_L[id], files_R[id])

    buf = samp['spectra'].copy()
    buf = [buf['central L']*1e-9, buf['central R']*1e-9]
    samp['spectra'] = buf

    PSF_L_0 = samp['PSF L'].sum(axis=0)
    PSF_R_0 = samp['PSF R'].sum(axis=0)

    ROI = (slice(128-32, 128+32), slice(128-32, 128+32))

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(np.log(np.abs(PSF_L_0[ROI])), origin='lower')
    axs[1].imshow(np.log(np.abs(PSF_R_0[ROI])), origin='lower')
    axs[0].set_title(str(np.round(samp['spectra'][0]*1e9).astype('uint'))+' [nm]')
    axs[1].set_title(str(np.round(samp['spectra'][1]*1e9).astype('uint'))+' [nm]')
    for ax in axs: ax.axis('off')
    fig.suptitle(id)
    fig.tight_layout()
    plt.show()
    
    return samp

# sample = plot_sample(1632)

#%%
def ReduceIRDISData():
    bad_files, good_files = [], []

    loader = SPHERE_loader(path_dtts)

    for id in range(len(files_L)):
        print('Processing: '+str(id)+'/'+str(len(files_L)))
        try:
            _ = loader.load(files_L[id], files_R[id])
            loader.save(path_output, prefix=str(id)+'_')
            good_files.append(files_L[id])
        except:
            print('################### Error occured for file: ', files_L[id], ('###################'))
            bad_files.append(files_L[id])

    print('===================== Completed! =====================')
    return good_files, bad_files

# good_files, bad_files = ReduceIRDISData()

#%%
def SaveReducedAsImages():
    # Visualize PSFs and save them into temporary folder
    import matplotlib

    folder        = SPHERE_DATA_FOLDER + 'IRDIS_reduced/'
    dir_save_imgs = SPHERE_DATA_FOLDER + 'IRDIS_images/'

    files = os.listdir(folder)
    crop  = slice(128-32, 128+32)

    for file in tqdm(files):
        if file not in os.listdir(dir_save_imgs):
            with open(os.path.join(folder, file), 'rb') as handle:
                data  = pickle.load(handle)
                image = data['PSF L'].sum(axis=0)[crop,crop]
                plt.figure(figsize=(7.5,7.5))
                current_cmap = matplotlib.cm.get_cmap()
                current_cmap.set_bad(color='red')
                plt.imshow(np.log(np.abs(image)))
                plt.title(file.replace('.pickle',''))
                plt.axis('off')
                # plt.show()
                plt.savefig(dir_save_imgs + file.replace('.pickle','.png'), bbox_inches='tight', pad_inches=0)

# SaveReducedAsImages()

#%%
'''
folda_init = 'F:/ESO/Data/SPHERE/images_sortedmore/'

# Scan all files in the folder and subfolders
def list_files(folder):
    file_list = []
    for _, _, files in os.walk(folder):
        for file in files:
            # file_list.append(os.path.join(root, file))
            if file.endswith('.png'):
                file_list.append(file)
    return file_list

file_list_old = list_files(folda_init)

#%
import shutil

file_list_new = os.listdir(SPHERE_DATA_FOLDER+'IRDIS_images/')
set_old  = set(file_list_old)
set_new  = set(file_list_new)
set_diff = list(set_new.intersection(set_old))

folda_new = SPHERE_DATA_FOLDER+'images_new/'

# Move the files
for file in set_diff:
    shutil.move(SPHERE_DATA_FOLDER+'IRDIS_images/'+file, folda_new+file)

'''

#%%
def CreateSPHEREdataframe(save_df_dir=None):

    reduced_files = os.listdir(path_output)

    columns_main = [
        'ID',
        'Filename',
        'Date',
        'Observation',
        'RA',
        'DEC',
        'Airmass',
        'Azimuth',
        'Altitude',
        'r0 (SPARTA)',
        'Seeing (SPARTA)',
        'Seeing (MASSDIMM)',
        'Seeing (DIMM)',
        'FWHM',
        'Strehl',
        'Turb. speed',
        'Wind direction (header)',
        'Wind direction (MASSDIMM)',
        'Wind direction (200 mbar)',
        'Wind speed (header)',
        'Wind speed (SPARTA)',
        'Wind speed (MASSDIMM)',
        'Wind speed (200 mbar)',
        'Tau0 (header)',
        'Tau0 (SPARTA)',
        'Tau0 (MASSDIMM)',
        'Tau0 (MASS)',
        'Jitter X',
        'Jitter Y',
        'Focus',
        'ITTM pos (avg)',
        'ITTM pos (std)',
        'DM pos (avg)',
        'DM pos (std)',
        'Pressure',
        'Humidity',
        'Temperature',
        'Temperature (200 mbar)',
        'Nph WFS',
        'Flux WFS',
        'Rate',
        'Filter common',
        'Filter LR',
        'Filter WFS',
        'DIT Time',
        'Num. DITs',
        'Sequence time']


    main_df = {}
    for col in columns_main:
        main_df[col] = []

    mag_list, files, spectra = [], [], []

    for file in tqdm(reduced_files):
        with open(os.path.join(path_output, file), 'rb') as handle:
            data = pickle.load(handle)

        id = int(file.split('_')[0])
        pure_filename = file.replace('.pickle','').replace(str(id)+'_','')

        main_df['ID'].append(id)
        main_df['Filename'].append(pure_filename)
        main_df['Date'].append(data['observation']['date'])
        main_df['Observation'].append(data['observation']['name'])
        main_df['RA'].append(data['observation']['RA'])
        main_df['DEC'].append(data['observation']['DEC'])
        main_df['Airmass'].append(data['telescope']['airmass'])
        main_df['Azimuth'].append(data['telescope']['azimuth'])
        main_df['Altitude'].append(data['telescope']['altitude'])
        main_df['r0 (SPARTA)'].append(data['r0'])
        main_df['Seeing (SPARTA)'].append(data['seeing']['SPARTA'])
        main_df['Seeing (MASSDIMM)'].append(data['seeing']['MASSDIMM'])
        main_df['Seeing (DIMM)'].append(data['seeing']['DIMM'])
        main_df['FWHM'].append(data['FWHM'])
        main_df['Strehl'].append(data['Strehl'])
        main_df['Turb. speed'].append(data['turb speed'])
        main_df['Wind direction (header)'].append(data['Wind direction']['header'])
        main_df['Wind direction (MASSDIMM)'].append(data['Wind direction']['MASSDIMM'])
        main_df['Wind direction (200 mbar)'].append(data['Wind direction']['200 mbar'])
        main_df['Wind speed (header)'].append(data['Wind speed']['header'])
        main_df['Wind speed (SPARTA)'].append(data['Wind speed']['SPARTA'])
        main_df['Wind speed (MASSDIMM)'].append(data['Wind speed']['MASSDIMM'])
        main_df['Wind speed (200 mbar)'].append(data['Wind speed']['200 mbar'])
        main_df['Tau0 (header)'].append(data['tau0']['header'])
        main_df['Tau0 (SPARTA)'].append(data['tau0']['SPARTA'])
        main_df['Tau0 (MASSDIMM)'].append(data['tau0']['MASSDIMM'])
        main_df['Tau0 (MASS)'].append(data['tau0']['MASS'])
        main_df['Pressure'].append(data['Environment']['pressure'])
        main_df['Humidity'].append(data['Environment']['humidity'])
        main_df['Temperature'].append(data['Environment']['temperature'])
        main_df['Temperature (200 mbar)'].append(data['Environment']['temperature (200 mbar)'])
        main_df['Nph WFS'].append(data['WFS']['Nph vis'])
        main_df['Flux WFS'].append(data['WFS']['Flux vis'])
        main_df['Jitter X'].append(data['WFS']['TT jitter X'])
        main_df['Jitter Y'].append(data['WFS']['TT jitter Y'])
        main_df['Focus'].append(data['WFS']['Focus'])
        main_df['ITTM pos (avg)'].append(data['WFS']['ITTM pos (avg)'])
        main_df['ITTM pos (std)'].append(data['WFS']['ITTM pos (std)'])
        main_df['DM pos (avg)'].append(data['WFS']['DM pos (avg)'])
        main_df['DM pos (std)'].append(data['WFS']['DM pos (std)'])
        main_df['Rate'].append(data['WFS']['rate'])
        main_df['Filter common'].append(data['spectra']['filter common'])
        main_df['Filter LR'].append(data['spectra']['filter LR'])
        main_df['Filter WFS'].append(data['WFS']['wavelength'])
        main_df['DIT Time'].append(data['Integration']['DIT'])
        main_df['Num. DITs'].append(data['Integration']['Num. DITs'])
        main_df['Sequence time'].append(data['Integration']['sequence time'])

        # Get spectrum information
        files.append(file)
        mag_list.append(data['observation']['magnitudes'])
        spectra.append(data['spectra'])

    main_df = pd.DataFrame(main_df)

    # To find the closest filter to the wavelength of the observation
    wvl = np.array([1043, 1245, 1625, 2182, 1085, 1213, 1283, 1573, 1642, 2091, 2124, 2170, 2266,
                    2290, 1022, 1190, 1593, 1593, 1667, 2110, 1076, 1273, 1667, 1667, 1733, 2251])

    bw =  np.array([140, 240, 290, 300, 14, 17, 18,  23, 24, 34, 31, 31, 32,
                    33,  49,  42,  52,  52, 54, 102, 50, 46, 54, 54, 57, 109])

    # Get all possible magnitudes
    mags = []
    for mag_ in mag_list:
        if isinstance(mag_, dict):
            mags += [mag for mag in mag_ if mag not in mags]

    mag_cols_names = ['mag '+mag for mag in mags]

    spectrum_df = { i: [] for i in ['ID', 'Filename', 'λ left (nm)', 'λ right (nm)', 'Δλ left (nm)', 'Δλ right (nm)'] }
    for col in mag_cols_names: spectrum_df[col] = []

    for record in tqdm(files):
        id = int(record.split('_')[0])
        pure_filename = record.replace('.pickle','').replace(str(id)+'_','')
            
        spectrum_df['ID'].append(id)
        spectrum_df['Filename'].append(pure_filename)

        # Get source magnitude information
        current_mag_list = mag_list[files.index(record)]
        for mag in mags:
            if pd.isnull(current_mag_list):
                spectrum_df['mag '+mag].append(np.nan)
            else:
                if mag in current_mag_list:
                    x = current_mag_list[mag]
                    if isinstance(x, list) or isinstance(x, tuple):
                        x = x[0]
                    spectrum_df['mag '+mag].append(x)
                else:
                    spectrum_df['mag '+mag].append(np.nan)

        # Get spectrum information
        sp = deepcopy(spectra[files.index(record)])
        spectrum_df['λ left (nm)'  ].append(sp['central R'])
        spectrum_df['λ right (nm)' ].append(sp['central L'])
        spectrum_df['Δλ left (nm)' ].append(np.round(sp['range L'][-1] - sp['range L'][0]).astype(np.int_).item())
        spectrum_df['Δλ right (nm)'].append(np.round(sp['range R'][-1] - sp['range R'][0]).astype(np.int_).item())

    spectrum_df = pd.DataFrame(spectrum_df)

    # sns.set_theme()  # <-- This actually changes the look of plots.
    # plt.hist([df['λ left (nm)'], df['λ right (nm)']], bins=30, color=['r','b'], alpha=0.5)
    # plt.xlabel('λ [nm]')
    # plt.ylabel('Count')
    # plt.legend(['λ left', 'λ right'])

    # sns.set_theme()  # <-- This actually changes the look of plots.
    # plt.hist([df['Δλ left (nm)'], df['Δλ right (nm)']], bins=30, color=['r','b'], alpha=0.5)
    # plt.legend(['Δλ left', 'Δλ right'])
    # plt.xlabel('λ [nm]')
    # plt.ylabel('Count')

    # Load labels information
    all_labels = []
    labels_df  = { 'ID': [], 'Filename': [] }

    if os.path.exists(SPHERE_DATA_FOLDER+'labels.txt'):
        with open(SPHERE_DATA_FOLDER+'labels.txt', 'r') as f:
            for line in f:
                filename, labels = line.strip().split(': ')

                ID = filename.split('_')[0]
                pure_filename = filename.replace(ID+'_', '').replace('.png', '')

                labels_df['ID'].append(int(ID))
                labels_df['Filename'].append(pure_filename)
                all_labels.append(labels.split(', '))
    else:
        raise ValueError('Labels file does not exist!')

    labels_list = list(set( [x for xs in all_labels for x in xs] ))
    labels_list.sort()

    for i in range(len(labels_list)):
        labels_df[labels_list[i]] = []

    for i in range(len(all_labels)):
        for label in labels_list:
            labels_df[label].append(label in all_labels[i])

    labels_df = pd.DataFrame(labels_df)

    # Merge all dataframes
    df = pd.merge(main_df, spectrum_df, on=['ID','Filename'])
    df = pd.merge(df, labels_df, on=['ID','Filename'])
    df.sort_values('ID', inplace=True)
    df.set_index('ID', inplace=True)

    try:
        if save_df_dir is not None:
            print('Saving dataframe to:', save_df_dir)
            with open(save_df_dir, 'wb') as handle:
                pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print('Error: Could not save the dataframe!')

    return df

# df = CreateSPHEREdataframe(save_df_dir=SPHERE_DATA_FOLDER+'sphere_df.pickle')

#%%
def LoadSPHEREsampleByID(id): # searches for the sample with the specified ID in
    with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
        request_df = pickle.load(handle)

    file = request_df.loc[request_df.index == id]['Filename'].values[0]
    full_filename = SPHERE_DATA_FOLDER+'IRDIS_reduced/'+str(id)+'_'+file+'.pickle'

    with open(full_filename, 'rb') as handle:
        data_sample = pickle.load(handle)
    return data_sample


def plot_sample(id):
    samp = LoadSPHEREsampleByID(id)

    buf = samp['spectra'].copy()
    buf = [buf['central L']*1e-9, buf['central R']*1e-9]
    samp['spectra'] = buf

    PSF_L_0 = samp['PSF L'].sum(axis=0)
    PSF_R_0 = samp['PSF R'].sum(axis=0)

    ROI = slice(PSF_L_0.shape[0]-32, PSF_L_0.shape[0]+32)
    ROI = (ROI, ROI)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(np.log(np.abs(PSF_L_0[ROI])), origin='lower')
    axs[1].imshow(np.log(np.abs(PSF_R_0[ROI])), origin='lower')
    axs[0].set_title(str(np.round(samp['spectra'][0]*1e9).astype('uint'))+' [nm]')
    axs[1].set_title(str(np.round(samp['spectra'][1]*1e9).astype('uint'))+' [nm]')
    for ax in axs: ax.axis('off')
    fig.suptitle(id)
    fig.tight_layout()
    plt.show()
    # plt.savefig(save_folder / f'{id}.png', dpi=300)


