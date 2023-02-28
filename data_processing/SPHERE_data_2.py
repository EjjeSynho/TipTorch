#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import os
from os import path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm
from query_eso_archive import query_simbad
import pickle
import re
from copy import deepcopy, copy
from pathlib import Path
from tools.utils import GetROIaroundMax, mask_circle
from scipy.ndimage.measurements import center_of_mass


ROOT = Path('E:/ESO/Data/SPHERE/IRDIS_RAW')
path_dtts = Path('E:/ESO/Data/SPHERE/DTTS/')

folder_L = ROOT/'SPHERE_DC_DATA_LEFT/'
folder_R = ROOT/'SPHERE_DC_DATA_RIGHT/'

path_output = 'E:/ESO/Data/SPHERE/IRDIS_reduced/'

h = 'HIERARCH ESO '

files_L = []
files_R = []

#%%
def GetFilesList():
    global files_L, files_R

    def list_files(path):
        file_list = []
        for root, _, files in os.walk(path):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list

    files_L = list_files(folder_L)
    files_R = list_files(folder_R)


    # Scanning for matchjng files
    pure_filename = lambda x: re.search(r'([^\\\/]*)(_left|_right)?\.fits$', str(x)).group(1).replace('_left','').replace('_right','')

    pure_filenames_L = [pure_filename(file) for file in files_L]
    pure_filenames_R = [pure_filename(file) for file in files_R]

    # Find files that are not in both folders
    files_L_only = [file for file in files_L if pure_filename(file) not in pure_filenames_R]
    files_R_only = [file for file in files_R if pure_filename(file) not in pure_filenames_L]

    # Find files that are in both folders
    files_L_in_R = [file for file in files_L if pure_filename(file) in pure_filenames_R]
    files_R_in_L = [file for file in files_R if pure_filename(file) in pure_filenames_L]

    if len(files_L_only) == 0 and len(files_R_only) == 0:
        print('All files are in both folders')
    else:
        print('Files only in left folder:')
        for file in files_L_only:
            print(file)
        print('Files only in right folder:')
        for file in files_R_only:
            print(file)


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
        if 'PIXSCAL' in self.hdr:
            psInMas = float(self.hdr['PIXSCAL'])
        else:
            psInMas = 12.27
        gain = float(self.hdr[0].header[h+'DET CHIP1 GAIN'])
        ron  = float(self.hdr[0].header[h+'DET CHIP1 RON'])
        if ron == 0:
            ron = 4.0
        # http://www.eso.org/observing/dfo/quality/SPHERE/reports/HEALTH/trend_report_IRDIS_DARK_ron_HC.html
        DIT    = self.hdr[0].header[h+'DET SEQ1 DIT']
        NDIT   = self.hdr[0].header[h+'DET NDIT']
        try:
            NDSKIP = self.hdr[0].header[h+'DET NDSKIP']
        except:
            NDSKIP = 0

        return psInMas, gain, ron, DIT, NDIT, NDSKIP


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
                return df
            else:
                return None


        def find_closest(df, timestamp, T_exp):
            df["date"] = pd.to_datetime(df["date"])
            timestamp_start = timestamp - pd.DateOffset(seconds=0.1)
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

        # Initialize the recepies to match the entries wiith the telemetry
        # sparta_IR_DTTS
        entries_DTTS = [('n_ADU', 'flux_IRLoop_ADU')]

        # sparta_visible_WFS
        entries_spart_vis = [
            ('n_ph', 'flux_VisLoop[#photons/subaperture/frame]'),
            ('rate', 'WFS frequency')]



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


        def get_entry_value(df, entry, id_start, id_end):
            if entry in df and id_start is not None and id_end is not None:
                value = np.nanmedian(df[entry][id_start:id_end+1])
                if not np.isreal(value) or np.isinf(value):
                    return np.nan
                else:
                    return value
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
            tables.sort_values(by='date', inplace=True)
            tables['PSF'] = DTTS_PSFs
            tables.drop_duplicates(subset=['date'], inplace=True)

            id_min, id_max = find_closest(tables, timestamp, T_exp)

            cube_array = []
            cube = tables.iloc[id_min:id_max+1]['PSF']
            for i in range(len(cube)):
                cube_array.append(np.array(cube[i]))
            cube_array = np.array(cube_array)
        except:
            cube_array = np.nan
        
        return SPARTA_data, MASSDIMM_data, cube_array


    def GetSpectrum(self):
        def get_filter_transmission(filter_name):
            root_filters = 'C:/Users/akuznets/Projects/TipToy/data/calibrations/VLT_CALIBRATION/IRDIS_filters/'
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
            if filter_name in ['CLEAR','P0-90', 'P45-135']:
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
            'range L':    wavelength_range[mask_L],
            'range R':    wavelength_range[mask_R],
            'spectrum L': transmission_L[mask_L],
            'spectrum R': transmission_R[mask_R],
            'central L':  wvl1_L,
            'central R':  wvl1_R,
        }
        return spectrum


    def LoadObservationData(self):
        TELAZ, TELALT, airmass = self.get_telescope_pointing()
        magnitudes, OB_NAME, RA, DEC = self.get_star_magnitudes()
        tau0, wDir, wSpeed, RHUM, pressure, fwhm = self.get_ambi_parameters()
        psInMas, gain, ron, DIT, NDIT, NDSKIP = self.get_detector_config()
        
        SPARTA_data, MASSDIMM_data, cube_array = self.GetTelemetryDTTS()
        spectrum = self.GetSpectrum()

        data = {
            'spectra': spectrum,

            'DTTS': cube_array,

            'tau0': {
                'header': tau0,
                'SPARTA': SPARTA_data['tau0'],
                'MASSDIMM': MASSDIMM_data['MASSDIMM_tau0'],
                'MASS': MASSDIMM_data['MASS_tau0']
            },

            'r0': SPARTA_data['r0'],
            'seeing': {
                'SPARTA': SPARTA_data['seeing'],
                'MASSDIMM': MASSDIMM_data['MASSDIMM_seeing'],
                'DIMM': MASSDIMM_data['DIMM_seeing']
            },

            'Wind speed': {
                'header': wSpeed,
                'SPARTA': SPARTA_data['windspeed'],
                'MASSDIMM': MASSDIMM_data['windspeed'],
            },

            'Wind direction': {
                'header': wDir,
                'MASSDIMM': MASSDIMM_data['winddir'],
            },

            'Environment': {
                'pressure': pressure,
                'humidity': RHUM,
                'temperature': MASSDIMM_data['temperature']
            },

            'FWHM': fwhm,
            'Strehl': SPARTA_data['SR'],
            'turb speed': MASSDIMM_data['MASSDIMM_turb_speed'],

            'Detector': {
                'psInMas': psInMas,
                'gain': gain,
                'ron': ron
            },

            'WFS': {
                'Nph vis': SPARTA_data['n_ph'],
                'rate': SPARTA_data['rate']
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
            print('Saving to: ', path_new)
            with open(path_new, 'wb') as handle:
              pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError('Wrong path is specified!')
        

    def __init__(self, path_dtts) -> None:
        self.data = {}
        self.path_dtts = path_dtts


#%%
def ReduceIRDISData():
    bad_files = []
    good_files = []

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


#%%
def SaveReducedAsImages():
    # Visualize PSFs and save them into temporary folder
    import matplotlib

    folder = 'E:/ESO/Data/SPHERE/IRDIS_reduced/'
    dir_save_imgs = 'E:/ESO/Data/SPHERE/reduced_imgs/'

    files = os.listdir(folder)
    crop = slice(128-32, 128+32)

    # id, pure_file = purify_filename(file)
    for file in tqdm(files):
        with open(os.path.join(folder, file), 'rb') as handle:
            data = pickle.load(handle)

            image = data['PSF L'].sum(axis=0)[crop,crop]
            plt.figure(figsize=(7.5,7.5))
            current_cmap = matplotlib.cm.get_cmap()
            current_cmap.set_bad(color='red')
            plt.imshow(np.log(np.abs(image)))
            plt.title(file.replace('.pickle',''))
            plt.axis('off')
            # plt.show()
            plt.savefig(dir_save_imgs + file.replace('.pickle','.png'), bbox_inches='tight', pad_inches=0)


#%%
class SPHERE_database:
    def __init__(self, path_input, path_fitted):
        files_input  = os.listdir(path_input)
        files_fitted = os.listdir(path_fitted)

        # Some samples were skipped during fittin. Check which ones were not by their id
        ids_fitted = set( [int(re.findall('[0-9]+', file)[0]) for file in files_fitted] )
        self.file_ids = []
        self.data = []

        print('Loading input data from '+path_input+'...\nLoading fitted data from '+path_fitted+'...')
        for file in tqdm(files_input):
            file_id = int(re.findall('[0-9]+', file)[0])
            if file_id in ids_fitted:
                try:
                    with open(os.path.join(path_input, file), 'rb') as handle:
                        data_input = pickle.load(handle)
                    with open(os.path.join(path_fitted, str(file_id)+'.pickle'), 'rb') as handle:
                        data_fitted = pickle.load(handle)
                    self.data.append((data_input, data_fitted, file_id))
                    self.file_ids.append(file_id)

                except OSError:
                    print('Cannot read a file with such name! Skipping...')

    def find(self,sample_id):
        try: #file_id is the identifier of a sample in the database, id is just an index in array
            id = self.file_ids.index(sample_id)
        except ValueError:
            print('Cannot find sample with index', sample_id)
            return None
        return {'input': self.data[id][0], 'fitted': self.data[id][1], 'index': id}

    def __getitem__(self, key):
        return {'input': self.data[key][0], 'fitted': self.data[key][1], 'file_id': self.data[key][2] }

    def pop(self, id): #removes by file index
        if hasattr(id, '__iter__'):
            for i in id:
                self.data.pop(i)
                self.file_ids.pop(i)
        else:
            self.data.pop(id)
            self.file_ids.pop(id)
    
    def remove(self, file_id): #removes by file id
        if hasattr(id, '__iter__'):
            for i in file_id:
                buf = self.find(i)
                self.data.pop(buf['index'])
                self.file_ids.pop(buf['index'])
        else:
            buf = self.find(file_id)
            self.data.pop(buf['index'])
            self.file_ids.pop(buf['index'])

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for sample in self.data:
            yield {'input': sample[0], 'fitted': sample[1], 'file_id': sample[2]}

    def __add__(self, o):
        for sample, file_id in zip(o.data, o.file_ids):
            self.data.append(sample)
            self.file_ids.append(file_id)

    def __sub__(self, o):
        self.remove(o.file_ids)

    def subset(self, ids):
        buf = copy(self)
        if hasattr(ids, '__iter__') or hasattr(ids, '__len__'):
            buf.data = [self.data[i] for i in ids]
            buf.file_ids = [self.file_ids[i] for i in ids]
        else:
            buf.data = [self.data[ids]]
            buf.file_ids = [self.file_ids[ids]]
        return buf

    def split(self, ids):
        exclude_ids = set(ids)
        inlude_ids  = set(range(len(self.data))) - exclude_ids
        return self.subset(inlude_ids), self.subset(exclude_ids)

#%%
class SPHERE_dataset:

    def GetLabeledIDs(self, labels_file='C:/Users/akuznets/Data/SPHERE/labels.txt'):
        with open(labels_file) as f: lines = f.readlines()
        tags = []
        data = []
        for line in lines:
            buf = line.replace(' \n', '').replace('\n', '').split(' ')
            buf_dict = {'id': int(buf[0]), 'properties': buf[1:]}
            for i in buf[1:]: tags.append(i)
            data.append(buf_dict)

        def unique_list(inp):
            buf = []
            for x in inp:
                if x not in buf: buf.append(x)
            return buf

        tags = unique_list(tags) # get the list of all PSF labels

        # Generate set of ids
        sets_raw = {}
        all_ids = []
        for tag in tags:
            ids = []
            for record in data:
                if tag in record['properties']:
                    ids.append(record['id'])
                all_ids.append(record['id'])
            sets_raw[tag] = set(ids)

        all_ids = set(unique_list(all_ids))
        double_ids = sets_raw['Double']
        LWE_ids    = sets_raw['LWE']
        good_ids   = sets_raw['Good'] - double_ids
        wasted_ids = sets_raw['Corrupted'].union(sets_raw['Empty']).union(sets_raw['Blurry']).union(sets_raw['Clipped'])-good_ids

        golden_ids = deepcopy(good_ids)
        for tag in sets_raw:
            if tag not in 'Good': golden_ids -= sets_raw[tag]

        soso_ids  = all_ids - good_ids - wasted_ids
        valid_ids = all_ids - wasted_ids

        sets = {
            'Total':      all_ids,
            'Valid':      valid_ids,
            'Invalid':    wasted_ids,
            'Golden':     golden_ids,
            'Good':       good_ids,
            'Category B': soso_ids,
            'LWE':        LWE_ids,
            'Doubles':    double_ids
        }

        print('=== Dataset stats ===')
        print('Total:      ', len(all_ids))
        print('Valid:      ', len(valid_ids))
        print('Invalid:    ', len(wasted_ids))
        print('----------------------')
        print('Good:       ', len(good_ids))
        print('Golden:     ', len(golden_ids))
        print('Category B: ', len(soso_ids))
        print('LWE:        ', len(LWE_ids))
        print('Doubles:    ', len(double_ids))

        return sets_raw, sets


    # Filter samples with the same (the most adundant) wavelength
    def FilterWavelength(self):
        wvls = []
        for data_sample in self.database:
            wvl = data_sample['input']['spectrum']['lambda']
            wvls.append(wvl)
        wvls = np.array(wvls)

        sample_ids = np.arange(len(self.database))
        wvl_unique, _, unique_indices, counts = np.unique(wvls, return_index=True, return_inverse=True, return_counts=True)
        return self.database.subset(sample_ids[unique_indices==np.argmax(counts)])


    def __init__(self):
        self.sets_raw, self.sets = self.GetLabeledIDs()

        self.normalizer = {
            'wavelength':     [1e6,    0],
            'r0':             [5,     -1],
            'F':              [1,      0],
            'tau0':           [50,     0],
            'wind speed':     [1/10,  -0.75],
            'wind direction': [1/180,  0],
            'airmass':        [1,     -1],
            'WFS photons':    [0.625, -4.5],
            'dx':             [2,      0],
            'dy':             [2,      0],
            'background':     [0.5e6,  0],
            'dn':             [1,      0],
            'Jx':             [0.1,    0],
            'Jy':             [0.1,    0],
            'Jxy':            [0.05,   0] }

        self.f     = lambda x,n: x * n[0] + n[1]
        self.f_inv = lambda y,n: (y - n[1]) / n[0]

        # Load the SPHERE PSF database
        root = 'C:/Users/akuznets/Data/SPHERE/'

        #path_fitted = root + 'fitted_TipToy_sumnorm/'
        path_fitted = root + 'fitted_TipToy_maxnorm/'
        path_input  = root + 'test/'

        self.database = SPHERE_database(path_input, path_fitted)

        bad_samples = []
        for data_sample in self.database:
            entries = ['r0', 'F', 'n', 'dn', 'Nph WFS', 'Jx', 'Jy', 'Jxy', 'dx', 'dy', 'bg']
            buf = np.array([data_sample['fitted'][entry] for entry in entries])
            if np.any(np.isnan(buf)):
                bad_samples.append(data_sample['file_id'])
                
        for bad_sample in bad_samples:
            self.database.remove(bad_sample)
        print(str(len(bad_samples))+' samples were filtered, '+str(len(self.database.data))+' samples remained')


    def GetInputs(self, data_sample):
        wvl     = data_sample['input']['spectrum']['lambda']
        r_0     = 3600*180/np.pi*0.976*0.5e-6 / data_sample['input']['seeing']['SPARTA'] # [m]
        tau0    = data_sample['input']['tau0']['SPARTA']
        wspeed  = data_sample['input']['Wind speed']['MASSDIMM']
        wdir    = data_sample['input']['Wind direction']['MASSDIMM']
        airmass = data_sample['input']['telescope']['airmass']
        N_ph    = np.log10(data_sample['input']['WFS']['Nph vis'] * data_sample['input']['WFS']['rate']*1240)
        
        if wdir >= 180: wdir -= 360.0

        vals = {
            'wavelength':     wvl,
            'r0':             r_0,
            'tau0':           tau0,
            'wind speed':     wspeed,
            'wind direction': wdir,
            'airmass':        airmass,
            'WFS photons':    N_ph,
            'dx':             data_sample['fitted']['dx'],
            'dy':             data_sample['fitted']['dy'],
            'background':     data_sample['fitted']['bg']
        }
        return torch.tensor([self.f(vals[entry], self.normalizer[entry]) for entry in vals]).flatten()


    def GetOutputs(self, data_sample):
        vals = {
            'r0':          np.abs(data_sample['fitted']['r0']),
            'F':           data_sample['fitted']['F'],
            'dx':          data_sample['fitted']['dx'],
            'dy':          data_sample['fitted']['dy'],
            'background':  data_sample['fitted']['bg'],
            'dn':          data_sample['fitted']['dn'],
            'Jx':          data_sample['fitted']['Jx'],
            'Jy':          data_sample['fitted']['Jy'],
            'Jxy':         data_sample['fitted']['Jxy'],
            'WFS photons': data_sample['fitted']['Nph WFS']
        }
        return torch.tensor([self.f(vals[entry], self.normalizer[entry]) for entry in vals]).flatten()


    def GetPSFs(self, data_sample, norm_regime):
        img_data   = data_sample['fitted']['Img. data'].squeeze(0)
        img_fitted = data_sample['fitted']['Img. fit'].squeeze(0)

        if norm_regime == 'sum':
            img_data   /= img_data.sum()
            img_fitted /= img_fitted.sum()

        if norm_regime == 'max':
            img_data   /= img_data.max()
            img_fitted /= img_fitted.max()

        return torch.tensor(img_data).float(), torch.tensor(img_fitted).float()


    def GenerateDataset(self, dataset):
        x  = [] # inputs
        y  = [] # labels
        i0 = [] # data images
        i1 = [] # fitted images

        for sample in dataset:
            input = self.GetInputs(sample)
            pred  = self.GetOutputs(sample)
            i_dat, i_fit = self.GetPSFs(sample, 'sum')
            x.append(input)
            y.append(pred)
            i0.append(i_dat)
            i1.append(i_fit)

        return torch.vstack(x), torch.vstack(y), torch.dstack(i0).permute([2,0,1]), torch.dstack(i1).permute([2,0,1])


#%%  Let it run!
#ProcessRawData()
#SaveDatasetImages()
#FilterSelectedDatasamples()

#%%
def LoadSPHEREsampleByID(path, sample_num): # searches for the sample with the specified ID in
    files = os.listdir(path)
    sample_nums = []
    for file in files: sample_nums.append( int(re.findall(r'[0-9]+', file)[0]) )

    try: id = sample_nums.index(sample_num)
    except ValueError:
        print('No sample with a such ID was found!')
        return np.nan
        
    with open(path+files[id], 'rb') as handle:
        sample = pickle.load(handle)
    return id, sample
