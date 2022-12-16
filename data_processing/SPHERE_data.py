#%%
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
import pickle5 as pickle
import re
from copy import deepcopy, copy

ROOT = 'C:/Users/akuznets/Data/SPHERE/'

#%%
class SPHERE_loader():

    def FilterNoiseBG(self, img, center, radius=80):
        # Mask empty area of the image
        mask_empty = np.ones(img.shape[0:2])
        mask_empty[np.where(img[:,:]==img[0,0])] = 0.
        # Mask center
        xx,yy = np.meshgrid( np.arange(0,img.shape[1]), np.arange(0,img.shape[0]) )
        mask_PSF = np.ones(img.shape[0:2])
        mask_PSF[np.sqrt((yy-center[0])**2 + (xx-center[1])**2) < radius] = 0.
        # Noise mask
        PSF_background = img[:,:]*mask_PSF*mask_empty
        noise_stats_mask = np.ones(img.shape[0:2])
        PSF_background[PSF_background == 0.0] = np.nan
        PSF_background[np.abs(PSF_background) > 3*np.nanstd(PSF_background)] = np.nan
        noise_stats_mask[np.abs(PSF_background) > 3*np.nanstd(PSF_background)] = 0.0
        return PSF_background, mask_empty*noise_stats_mask


    def LoadImage(self):
        im  = self.hdr[0].data.sum(axis=0)
        var = self.hdr[0].data.var(axis=0)
        n_frames = self.hdr[0].data.shape[2]
        # Crop image
        crop_size = 256
        ind = np.unravel_index(np.argmax(im, axis=None), im.shape)
        win_x = [ ind[0]-crop_size//2, ind[0]+crop_size//2 ]
        win_y = [ ind[1]-crop_size//2, ind[1]+crop_size//2 ]

        crop_x = slice(win_x[0], win_x[1])
        crop_y = slice(win_y[0], win_y[1])

        im = im[crop_x, crop_y]
        var = var[crop_x, crop_y]

        ind = np.unravel_index(np.argmax(im, axis=None), im.shape)
        background, mask = self.FilterNoiseBG(im, ind, radius=140)

        median = np.nanmedian(background)
        im -= median
        return im, var, median, n_frames


    def get_telescope_pointing(self):
        TELAZ = float(self.hdr[0].header['HIERARCH ESO TEL AZ'])
        TELALT = float(self.hdr[0].header['HIERARCH ESO TEL ALT'])
        airmass = 0.5*(float(self.hdr[0].header['HIERARCH ESO TEL AIRM END']) + float(self.hdr[0].header['ESO TEL AIRM START']))
        return TELAZ, TELALT, airmass


    def get_wavelength(self):
        filter_data = {
            'B_Y':  [1043,140],
            'B_J':  [1245,240],
            'B_H':  [1625,290],
            'B_Ks': [2182,300]
        }
        filter_name = self.hdr[0].header['HIERARCH ESO INS1 FILT NAME']

        wvl = 1e-9*filter_data[filter_name][0]
        bw  = 1e-9*filter_data[filter_name][1]
        return wvl, bw


    def get_ambi_parameters(self):
        tau0 = float(self.hdr[0].header['HIERARCH ESO TEL AMBI TAU0'])
        wDir = float(self.hdr[0].header['HIERARCH ESO TEL AMBI WINDDIR'])
        wSpeed = float(self.hdr[0].header['HIERARCH ESO TEL AMBI WINDSP'])
        RHUM  = float(self.hdr[0].header['HIERARCH ESO TEL AMBI RHUM'])
        pressure = 0.5*(float(self.hdr[0].header['HIERARCH ESO TEL AMBI PRES START']) + float(self.hdr[0].header['HIERARCH ESO TEL AMBI PRES END']))
        fwhm_linobs = float(self.hdr[0].header['HIERARCH ESO TEL IA FWHMLINOBS'])
        return tau0, wDir, wSpeed, RHUM, pressure, fwhm_linobs


    def read_DTTS_data(self):
        
        date_obs = self.hdr[0].header['DATE-OBS']
        # in the end of the header I need to find keyword OBS_STA OBS_END
        n_subap = 1240

        # grab the number of photons
        def find_closest(df, date, T_exp):
            df["date"] = pd.to_datetime(df["date"])
            id_closest = np.argmin(abs(df["date"] - date))
            date_min = df["date"][id_closest] - pd.DateOffset(seconds=T_exp/2)
            date_max = df["date"][id_closest] + pd.DateOffset(seconds=T_exp/2)
            id_min = np.argmin(abs(df["date"] - date_min))
            id_max = np.argmin(abs(df["date"] - date_max))
            return id_min, id_max

        def GetEntryValue(df, entry, id_min, id_max):
            if entry in df:
                value = df[entry][id_min:id_max+1].median()
                if not np.isreal(value) or np.isinf(value): return np.nan
                else: return value
            else: return np.nan
            
        if self.path_dtts is not None and date_obs is not None:
            date = pd.to_datetime(date_obs)
            path_sub = self.path_dtts + str(date.year) + "/" + date_obs[:10]

            # initialize dictionaries
            SPARTA_data = {}
            MASSDIMM_data = {}
            entries_SPARTA = ['n_ph', 'rate', 'r0', 'seeing', 'tau0', 'windspeed', 'SR']
            entries_MASSDIMM = ['MASS_tau0', 'MASSDIMM_seeing', 'MASSDIMM_tau0', 'MASSDIMM_turb_speed', 'temperature', 'winddir', 'windspeed']
            for entry in entries_SPARTA:   SPARTA_data[entry] = np.nan
            for entry in entries_MASSDIMM: MASSDIMM_data[entry] = np.nan

            if os.path.isdir(path_sub):
                # Calculate the exposure time
                T_exp  = self.hdr[0].header['HIERARCH ESO DET SEQ1 DIT'] * self.hdr[0].header['HIERARCH ESO DET NDIT']

                # Get the number of photons and rate (SPARTA)
                tmp = [file for file in os.listdir(path_sub) if "sparta_visible_WFS" in file]
                if len(tmp)>0:
                    file_name = tmp[0]
                    df = pd.read_csv(path_sub + "/" + file_name)
                    if 'No data' not in [col for col in df.columns][0]:
                        id_min, id_max = find_closest(df, date, T_exp)
                        SPARTA_data['n_ph'] = GetEntryValue(df, 'flux_VisLoop[#photons/aperture/frame]', id_min, id_max)/n_subap
                        SPARTA_data['rate'] = GetEntryValue(df, 'Frame rate [Hz]', id_min, id_max)

                # Get the atmospheric parameters (SPARTA)
                tmp = [file for file in os.listdir(path_sub) if "sparta_atmospheric_params" in file]
                if len(tmp)>0:
                    file_name = tmp[0]
                    df = pd.read_csv(path_sub + "/" + file_name)
                    if 'No data' not in [col for col in df.columns][0]:
                        id_min, id_max = find_closest(df, date, T_exp)
                        SPARTA_data['r0']        = GetEntryValue(df, 'r0_los_sparta', id_min, id_max)
                        SPARTA_data['seeing']    = GetEntryValue(df, 'seeing_los_sparta', id_min, id_max)
                        SPARTA_data['tau0']      = GetEntryValue(df, 'tau0_los_sparta', id_min, id_max)
                        SPARTA_data['windspeed'] = GetEntryValue(df, 'wind_speed_los_sparta', id_min, id_max)
                        SPARTA_data['SR']        = GetEntryValue(df, 'strehl_sparta', id_min, id_max)

                # Get the atmospheric parameters (MASS-DIMM)
                tmp = [file for file in os.listdir(path_sub) if "mass_dimm" in file]
                if len(tmp)>0:
                    file_name = tmp[0]
                    df = pd.read_csv(path_sub + "/" + file_name)
                    if 'No data' not in [col for col in df.columns][0]:
                        id_min, id_max = find_closest(df, date, T_exp)
                        MASSDIMM_data['MASS_tau0']           = GetEntryValue(df, 'MASS_tau0', id_min, id_max)
                        MASSDIMM_data['MASSDIMM_seeing']     = GetEntryValue(df, 'MASS-DIMM_seeing', id_min, id_max)
                        MASSDIMM_data['MASSDIMM_tau0']       = GetEntryValue(df, 'MASS-DIMM_tau0', id_min, id_max)
                        MASSDIMM_data['MASSDIMM_turb_speed'] = GetEntryValue(df, 'MASS-DIMM_turb_speed', id_min, id_max)

                # Get the ambient parameters asm?)
                tmp = [file for file in os.listdir(path_sub) if "asm" in file]
                if len(tmp) > 0:
                    file_name = tmp[0]
                    df = pd.read_csv(path_sub + "/" + file_name)
                    if 'No data' not in [col for col in df.columns][0]:
                        id_min, id_max = find_closest(df, date, T_exp)
                        MASSDIMM_data['temperature'] = GetEntryValue(df, 'air_temperature_30m[deg]', id_min, id_max)
                        MASSDIMM_data['winddir']     = GetEntryValue(df, 'winddir_30m', id_min, id_max)
                        MASSDIMM_data['windspeed']   = GetEntryValue(df, 'windspeed_30m', id_min, id_max)

                # Get the ambient parameters asm?)
                tmp = [file for file in os.listdir(path_sub) if "asm" in file]
                if len(tmp)>0:
                    file_name = tmp[0]
                    df = pd.read_csv(path_sub + "/" + file_name)
                    if 'No data' not in [col for col in df.columns][0]:
                        id_min, id_max = find_closest(df, date, T_exp)
                        MASSDIMM_data['temperature'] = GetEntryValue(df, 'air_temperature_30m[deg]', id_min, id_max)
                        MASSDIMM_data['winddir']     = GetEntryValue(df, 'winddir_30m', id_min, id_max)
                        MASSDIMM_data['windspeed']   = GetEntryValue(df, 'windspeed_30m', id_min, id_max)


        return SPARTA_data, MASSDIMM_data


    def get_detector_config(self):
        if 'PIXSCAL' in self.hdr:
            psInMas = float(self.hdr['PIXSCAL'])
        else:
            psInMas = 12.27
        gain = float(self.hdr[0].header['HIERARCH ESO DET CHIP1 GAIN'])
        ron  = float(self.hdr[0].header['HIERARCH ESO DET CHIP1 RON'])
        if ron==0:
            ron = 4.0
        # http://www.eso.org/observing/dfo/quality/SPHERE/reports/HEALTH/trend_report_IRDIS_DARK_ron_HC.html
        DIT    = self.hdr[0].header['HIERARCH ESO DET SEQ1 DIT']
        NDIT   = self.hdr[0].header['HIERARCH ESO DET NDIT']
        NDSKIP = self.hdr[0].header['HIERARCH ESO DET NDSKIP']

        return psInMas, gain, ron, DIT, NDIT, NDSKIP


    def get_star_coordinates(self):
        return self.hdr[0].header['HIERARCH ESO OBS NAME'], float(self.hdr[0].header['RA']), float(self.hdr[0].header['DEC'])


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


    def LoadObservationData(self):
        TELAZ, TELALT, airmass = self.get_telescope_pointing()
        wvl, bw = self.get_wavelength()
        magnitudes, OB_NAME, RA, DEC = self.get_star_magnitudes()
        tau0, wDir, wSpeed, RHUM, pressure, fwhm = self.get_ambi_parameters()
        SPARTA_data, MASSDIMM_data = self.read_DTTS_data()
        psInMas, gain, ron, DIT, NDIT, NDSKIP = self.get_detector_config()
        
        data = {
            'tau0': {
                'header': tau0,
                'SPARTA': SPARTA_data['tau0'],
                'MASSDIMM': MASSDIMM_data['MASSDIMM_tau0'],
                'MASS': MASSDIMM_data['MASS_tau0']
            },

            'r0': SPARTA_data['r0'],
            'seeing': {
                'SPARTA': SPARTA_data['seeing'],
                'MASSDIMM': MASSDIMM_data['MASSDIMM_seeing']
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
            'spectrum': { 'lambda': wvl, 'bandwidth': bw },
            'turb speed': MASSDIMM_data['MASSDIMM_turb_speed'],

            'Detector': {
                'psInMas': psInMas,
                'gain': gain,
                'ron': ron
            },

            'WFS': {
                'Nph vis': SPARTA_data['n_ph'],
                'rate':    SPARTA_data['rate']
            },

            'observation': {
                'name': OB_NAME,
                'date': self.hdr[0].header['DATE-OBS'],
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


    def Load(self, fits_filename=None):
        if fits_filename is not None:
            self.filename = fits_filename
        self.hdr = fits.open(self.filename)
        self.data = self.LoadObservationData()
        self.im, self.var, self.bg_median, self.n_frames = self.LoadImage()
        self.hdr.close()
        self.data['image'] = self.im
        self.data['variance'] = self.var
        self.data['bg_median'] = self.bg_median
        self.data['n_frames'] = self.n_frames


    def __init__(self, path_dtts, fits_filename=None) -> None:
        self.path_dtts = path_dtts
        self.data = None
        self.filename = fits_filename
        if self.filename is not None:
            self.Load(fits_filename)


    def save(self, path, prefix=''):
        path = os.path.normpath(path)
        if os.path.exists(path):
            file = os.path.split(self.filename)[-1][:-5]
            path_new = os.path.join(path, prefix+file+'.pickle')
            with open(path_new, 'wb') as handle:
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError('Wrong path is specified!')

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

#%%
# Save massively reduced data as pickle files with dictionaries
def ProcessRawData(save=True):
    #data_path = ROOT + 'SPHERE_DC_DATA/'
    #path_dtts = ROOT + 'SPHERE/DATA/DTTS/'
    data_path = 'C:/Users/akuznets/Data/SPHERE/SPHERE_DC_DATA/'
    path_dtts = 'H:/Data/SPHERE/DATA/DTTS/'

    folders = os.listdir(data_path)
    fits_files = []
    for folder in folders:
        files = os.listdir(path.join(data_path,folder))
        for file in files:
            fits_files.append(path.join(data_path,folder,file))

    loader = SPHERE_loader(path_dtts)

    corrupted_ids = []
    wrong_key_ids = []
    error_ids = []

    for id in tqdm(range(0,len(fits_files))):
        fits_filename = fits_files[id]
        try:
            loader.Load(fits_filename)
            if save: loader.save(ROOT + 'test', prefix=str(id)+'_')
        except KeyError:
            print('Ooops! Wrong key encountered while reading .fits file')
            wrong_key_ids.append(id)
        except OSError:
            print('Ooops! Corrupted one')
            corrupted_ids.append(id)
        except ValueError:
            print('Ooops! Something wrong for this one')
            error_ids.append(id)


# Visualize PSFs and save them into temporary folder
# so that it will be possible to look at bthem and filter them manualy
def SaveDatasetImages():
    import matplotlib
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='red')

    dir_test = ROOT + 'test'
    dir_save = ROOT + 'temp'

    files = os.listdir(dir_test)
    crop = slice(128-32, 128+32)
    crop = (crop,crop)

    for file in tqdm(files):
        with open(os.path.join(dir_test, file), 'rb') as handle:
            data = pickle.load(handle)
        plt.imshow(np.log(data['image'][crop]))
        plt.savefig(dir_save + file.split('_')[0]+'.png')


# Ones bad files are selected with pictures, move them to another folder
# This code reads the names of PSFs present in folders with pictures
def FilterSelectedDatasamples():
    import shutil
    valid_ids = set( [int(file.split('.')[0]) for file in os.listdir(ROOT + 'temp/')] )
    all_ids   = set( [int(file.split('_')[0]) for file in os.listdir(ROOT + 'test/')] )
    invalid_ids = all_ids-valid_ids

    files_dir  = ROOT + 'test/'
    target_dir = ROOT + 'test_invalid/'

    files = os.listdir(files_dir)
    for file in files:
        id = int(file.split('_')[0])
        if id in invalid_ids: shutil.move(os.path.join(files_dir, file), target_dir)

#%%

ProcessRawData(save=False)

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
