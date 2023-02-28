#%%
import sys
sys.path.insert(0, '..')
# sys.path.insert(0, 'C:/Users/akuznets/Projects/Datalab/libraries/dlab_config/dlab_config/')
# sys.path.insert(0, 'C:/Users/akuznets/Projects/Datalab/libraries/dlab_config/dlt/dlt')

import os
from os import path
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import pickle5 as pickle
from skimage.restoration import inpaint
# from tools.utils import GetROIaroundMax, wavelength_to_rgb

from datetime import datetime

initial_time = dlt.get_datetime('2022-12-01T00:00:00')
final_time = dlt.get_datetime('2022-12-31T00:00:00')

import dlt

lgs1seeing = dlt.query_ts('wmfsgw', 'AOS.LGS1.SEEING', initial_time, final_time)
lgs2seeing = dlt.query_ts('wmfsgw', 'AOS.LGS2.SEEING', initial_time, final_time)
lgs3seeing = dlt.query_ts('wmfsgw', 'AOS.LGS3.SEEING', initial_time, final_time)
lgs4seeing = dlt.query_ts('wmfsgw', 'AOS.LGS4.SEEING', initial_time, final_time)

#%%
def ReduceMUSEcube(hdul):
    def GetSpectrum(white, radius=5):
        _, ids = GetROIaroundMax(white, radius*2)
        return np.nansum(hdul[1].data[:, ids[0], ids[1]], axis=(1,2))

    find_closest = lambda λ, λs: np.argmin(np.abs(λs-λ)).astype('int')

    # Extract spectral range information
    start_spaxel = hdul[1].header['CRPIX3']
    num_λs = int(hdul[1].header['NAXIS3']-start_spaxel+1)
    Δλ = hdul[1].header['CD3_3' ] / 10.0
    λ_min = hdul[1].header['CRVAL3'] / 10.0
    λs = np.arange(num_λs)*Δλ + λ_min
    λ_max = λs.max()

    # 1 [erg] = 10^−7 [J];  1 [Angstrom] = 10 [nm];  1 [cm^2] = 0.0001 [m^2]
    # [10^-20 * erg / s / cm^2 / Angstrom] = [10^-22 * W/m^2 / nm] = [10^-22 * J/s / m^2 / nm]
    units = hdul[1].header['BUNIT'] # flux units

    # polychromatic image
    white = np.nan_to_num(hdul[1].data).sum(axis=0)
    spectrum = GetSpectrum(white)

    # Pre-defined bad wavelengths ranges
    bad_wvls = np.array([[450, 478], [577, 606]])
    bad_ids = np.array([find_closest(wvl, λs) for wvl in bad_wvls.flatten()], dtype='int').reshape(bad_wvls.shape)

    valid_λs = np.ones_like(λs)
    for i in range(len(bad_ids)):
        valid_λs[bad_ids[i,0]:bad_ids[i,1]+1] = 0
        bad_wvls[i,:] = np.array([λs[bad_ids[i,0]], λs[bad_ids[i,1]+1]])
    #λs_new = λs[np.where(valid_λs==1)]

    # Bin data cubes
    # Before the sodium filter
    λ_bin = (bad_wvls[1][0]-bad_wvls[0][1])/3.0
    λ_bins_before = bad_wvls[0][1] + np.arange(4)*λ_bin
    bin_ids_before = [find_closest(wvl, λs) for wvl in λ_bins_before]

    #After the sodium filter
    λ_bins_num = (λ_max-bad_wvls[1][1]) / np.diff(λ_bins_before).mean()
    λ_bins_after = bad_wvls[1][1] + np.arange(λ_bins_num+1)*λ_bin
    bin_ids_after = [find_closest(wvl, λs) for wvl in λ_bins_after]
    bins_smart = bin_ids_before + bin_ids_after
    λ_bins_smart = λs[bins_smart]

    Rs = np.zeros_like(λs)
    Gs = np.zeros_like(λs)
    Bs = np.zeros_like(λs)

    for i,λ in enumerate(λs): Rs[i], Gs[i], Bs[i] = wavelength_to_rgb(λ, show_invisible=True)

    Rs = Rs * valid_λs * spectrum / np.median(spectrum)
    Gs = Gs * valid_λs * spectrum / np.median(spectrum)
    Bs = Bs * valid_λs * spectrum / np.median(spectrum)
    colors = np.dstack([np.vstack([Rs, Gs, Bs])]*600).transpose(2,1,0)

    show_plots = True

    if show_plots:
        fig_handler = plt.figure(dpi=200)
        plt.imshow(colors, extent=[λs.min(), λs.max(), 0, 120])
        plt.vlines(λ_bins_smart, 0, 120, color='white') #draw bins borders
        plt.plot(λs, spectrum/spectrum.max()*120, linewidth=2.0, color='white')
        plt.plot(λs, spectrum/spectrum.max()*120, linewidth=0.5, color='blue')
        plt.xlabel(r"$\lambda$, [nm]")
        plt.ylabel(r"$\left[ 10^{-20} \frac{erg}{s \cdot cm^2 \cdot \AA} \right]$")
        ax = plt.gca()
        # ax.get_yaxis().set_visible(False)
        # plt.show()

    _, ROI = GetROIaroundMax(white, win=100)

    # Generate reduced cubes
    data_reduced = np.zeros([len(λ_bins_smart)-1, white[ROI].shape[0], white[ROI].shape[1]])
    std_reduced  = np.zeros([len(λ_bins_smart)-1, white[ROI].shape[0], white[ROI].shape[1]])
    wavelengths  = np.zeros(len(λ_bins_smart)-1)
    flux         = np.zeros(len(λ_bins_smart)-1)

    bad_layers = []

    for bin in tqdm( range(len(bins_smart)-1) ):
        chunk = hdul[1].data[ bins_smart[bin]:bins_smart[bin+1], ROI[0], ROI[1] ]
        wvl_chunck = λs[bins_smart[bin]:bins_smart[bin+1]]
        flux_chunck = spectrum[bins_smart[bin]:bins_smart[bin+1]]

        for i in range(chunk.shape[0]):
            layer = chunk[i,:,:]
            if np.isnan(layer).sum() > layer.size//2: # if more than 50% of the pixela on image are NaN
                bad_layers.append((bin, i))
                wvl_chunck[i] = np.nan
                flux_chunck[i] = np.nan
                continue
            else:
                mask_inpaint = np.zeros_like(layer, dtype=np.int8)
                mask_inpaint[np.where(np.isnan(layer))] = 1
                chunk[i,:,:] = inpaint.inpaint_biharmonic(layer, mask_inpaint)  # Fix bad pixels per spectral slice

        data_reduced[bin,:,:] = np.nansum(chunk, axis=0)
        std_reduced [bin,:,:] = np.nanstd(chunk, axis=0)
        wavelengths[bin] = np.nanmean(np.array(wvl_chunck)) # central wavelength for this bin
        flux[bin] = np.nanmean(np.array(flux_chunck))

    print(str(len(bad_layers))+'/'+str(hdul[1].data.shape[0]), '('+str(np.round(len(bad_layers)/hdul[1].data.shape[0],2))+'%)', 'slices are corrupted')

    # Collect the telemetry from the header
    misc_info = {
        'spectrum': spectrum,
        'wvl range': [λ_min, λ_max, Δλ], # num = int((λ_max-λ_min)/Δλ)+1
        'filtered wvls': bad_wvls,
        'wvl bins': λ_bins_smart,
        'flux units': units,
        'spectrum fig': fig_handler
    }

    h = 'HIERARCH ESO '

    telemetry = {
        'pixel scale': hdul[0].header[h+'OCS IPS PIXSCALE']*1000, #[mas/pixel]
        'target':      hdul[0].header[h+'OBS TARG NAME'],
        'date':        hdul[0].header['DATE'],
        'date-obs':    hdul[0].header['DATE-OBS'],
        'RA':          hdul[0].header['RA'],
        'DEC':         hdul[0].header['DEC'],
        'exp time':    hdul[0].header['EXPTIME'],
        'target':      hdul[0].header[h+'OBS TARG NAME'],
        'airmass':     (hdul[0].header[h+'TEL AIRM START'] + hdul[0].header[h+'TEL AIRM END']) / 2.0,
        'altitude':    hdul[0].header[h+'TEL ALT'],
        'azimuth':     hdul[0].header[h+'TEL AZ'],
        'seeing':      (hdul[0].header[h+'TEL AMBI FWHM START'] + hdul[0].header[h+'TEL AMBI FWHM END']) / 2.0,
        'tau0':        hdul[0].header[h+'TEL AMBI TAU0'],
        'temperature': hdul[0].header[h+'TEL AMBI TEMP'],
        'wind dir':    hdul[0].header[h+'TEL AMBI WINDDIR'],
        'wind speed':  hdul[0].header[h+'TEL AMBI WINDSP']
    }

    try:
        telemetry['Strehl'] = hdul[0].header[h+'OBS STREHLRATIO']
    except KeyError:
        telemetry['Strehl'] = hdul[0].header[h+'DRS MUSE RTC STREHL']

    data = {
        'cube':        data_reduced,
        'std':         std_reduced,
        'white':       white[ROI],
        'flux':        flux,
        'telemetry':   telemetry,
        'wavelengths': wavelengths,
        'misc info':   misc_info
    }
    return data

#%%
hdul = fits.open('E:/ESO/Data/MUSE/NFM-AO-Jan29/MUSE.2023-01-30T07%3A49%3A55.721/DATACUBE_FINAL.fits')
data1 = ReduceMUSEcube(hdul)
hdul.close()

# with open('C:/Users/akuznets/Data/MUSE/DATA_raw_binned/'+files[file_id].split('.fits')[0]+'.pickle', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
%matplotlib qt


import matplotlib.pyplot as plt
import numpy as np

data = np.log(data1['cube'].mean(axis=0))
# Get the extent of the data
extent = np.array([-data.shape[0]/2, data.shape[0]/2, -data.shape[1]/2, data.shape[1]/2]) * data1['telemetry']['pixel scale']


# Display the data with the extent centered at zero
plt.imshow(data, extent=extent.tolist(), origin='lower')
plt.colorbar()
plt.show()

#%%

#AOS LGSi FLUX: [ADU] Median flux in subapertures / Median of the LGSi flux in all subapertures with 0.17 photons/ADU when the camera gain is 100.

# [pix] slopes RMS / Median of the Y slopes std dev measured in LGS WFSi subap (0.83 arcsec/pixel).
HO_slopes_RMS_keywords = ['AOS.LGS'+str(i)+'.SLOPERMS'+coord for i,coord in zip([1,1,2,2,3,3,4,4], ['X', 'Y']*4)]

# [ADU / frame / subaperture] Median flux per subaperture with 0.17 photons/ADU when the camera gain is 100.
HO_flux_keywords = ['AOS.LGS'+str(i+1)+'.FLUX' for i in range(4)] 

#%%
def GetLGSphotons(flux_LGS, HO_gain):
    conversion_factor = 18 #[e-/ADU]
    GALACSI_transmission = 0.31 #(with VLT) / 0.46 (without VLT)
    detector_DIT = 1-0.01982 # [ms] 0.01982 [ms] are for transfer
    HO_rate = 1000 #[Hz] 
    QE = 0.9 # [e-/photons]
    # gain = 100

    num_subapertures = 1240
    M1_area = (8-1.12)**2 * np.pi / 4 # [m^2]

    return flux_LGS * conversion_factor * num_subapertures  \
           / HO_gain / detector_DIT / QE / GALACSI_transmission \
           * HO_rate / M1_area # [photons/m^2/s]

#%%
h = 'HIERARCH ESO '

# J+H magnitude was 7.1

HO_gains  = np.array([hdul[0].header[h+'AOS LGS'+str(i+1)+' DET GAIN'] for i in range(4)]).mean().item()
LO_regime = hdul[0].header[h+'AOS MAIN MODES IRLOS'] #'64x64_FullPupil_200Hz_HighGain'

if LO_regime == '64x64_FullPupil_200Hz_HighGain':
    LO_rate = 200 # [Hz]
    LO_gain = 98
else:
    raise ValueError('Unknown LO regime')

#[ADU/frame/sub aperture] * 4 sub apertures
IRLOS_flux_ADU = sum([hdul[0].header[h+'AOS NGS'+str(i+1)+' FLUX'] for i in range(4)]) # [ADU/frame]

def GetIRLOSphotons(flux_ADU, LO_gain, LO_rate):
    QE = 1.0 # [photons/e-]
    convert_factor = 9.8 # [e-/ADU]
    M1_area = (8-1.12)**2 * np.pi / 4 # [m^2]
    return flux_ADU / QE * convert_factor / LO_gain * LO_rate / M1_area # [photons/s/m^2]

IRLOS_flux = GetIRLOSphotons(IRLOS_flux_ADU, LO_gain, LO_rate) # [photons/s/m^2]

#%%

import dlt

layers_altitude_keywords = ['AOS.CNSQ.H'+str(i+1) for i in range(10)]
layers_CN2_keywords      = ['AOS.CNSQ.CNSQ_'+str(i+1) for i in range(10)]
layers_faction_keywords  = ['AOS.CNSQ.FRACCNSQ_'+str(i+1) for i in range(10)]

def GetValueByKeyword(keyword, initial_time, final_time):
    inquiry = dlt.query_ts('wmfsgw', keyword, initial_time, final_time)

    return inquiry


#%%



wspeed = dlt.query_ts('wt4tcs', 'TEL.AMBI.WINDSP', initial_time, final_time)
wdir = dlt.query_ts('wt4tcs', 'TEL.AMBI.WINDDIR', initial_time, final_time)

aofpseeing = dlt.query_ts('wmfsgw', 'AOS.CNSQ.SEEINGTOT', initial_time, final_time)
aofpL0    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0TOT', initial_time, final_time)
aofpR0    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.R0TOT', initial_time, final_time)
aofpH1    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.H1', initial_time, final_time)
aofpH2    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.H2', initial_time, final_time)
aofpH3    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.H3', initial_time, final_time)
aofpH4    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.H4', initial_time, final_time)
aofpH5    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.H5', initial_time, final_time)
aofpH6    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.H6', initial_time, final_time)
aofpH7    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.H7', initial_time, final_time)
aofpH8    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.H8', initial_time, final_time)
aofpH9    = dlt.query_ts('wmfsgw', 'AOS.CNSQ.H9', initial_time, final_time)
aofpH10   = dlt.query_ts('wmfsgw', 'AOS.CNSQ.H10', initial_time, final_time)
aofpCn21  = dlt.query_ts('wmfsgw', 'AOS.CNSQ.CNSQ_1', initial_time, final_time)
aofpCn22  = dlt.query_ts('wmfsgw', 'AOS.CNSQ.CNSQ_2', initial_time, final_time)
aofpCn23  = dlt.query_ts('wmfsgw', 'AOS.CNSQ.CNSQ_3', initial_time, final_time)
aofpCn24  = dlt.query_ts('wmfsgw', 'AOS.CNSQ.CNSQ_4', initial_time, final_time)
aofpCn25  = dlt.query_ts('wmfsgw', 'AOS.CNSQ.CNSQ_5', initial_time, final_time)
aofpCn26  = dlt.query_ts('wmfsgw', 'AOS.CNSQ.CNSQ_6', initial_time, final_time)
aofpCn27  = dlt.query_ts('wmfsgw', 'AOS.CNSQ.CNSQ_7', initial_time, final_time)
aofpCn28  = dlt.query_ts('wmfsgw', 'AOS.CNSQ.CNSQ_8', initial_time, final_time)
aofpCn29  = dlt.query_ts('wmfsgw', 'AOS.CNSQ.CNSQ_9', initial_time, final_time)
aofpCn210 = dlt.query_ts('wmfsgw', 'AOS.CNSQ.CNSQ_10', initial_time, final_time)
aofpL01   = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0_1', initial_time, final_time)
aofpL02   = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0_2', initial_time, final_time)
aofpL03   = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0_3', initial_time, final_time)
aofpL04   = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0_4', initial_time, final_time)
aofpL05   = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0_5', initial_time, final_time)
aofpL06   = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0_6', initial_time, final_time)
aofpL07   = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0_7', initial_time, final_time)
aofpL08   = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0_8', initial_time, final_time)
aofpL09   = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0_9', initial_time, final_time)
aofpL010  = dlt.query_ts('wmfsgw', 'AOS.CNSQ.L0_10', initial_time, final_time)ut4alt = dlt.query_ts('wt4tcs', 'TEL.ALT.POS', initial_time, final_time)
ut4az     = dlt.query_ts('wt4tcs', 'TEL.AZ.POS', initial_time, final_time)
ut4fwhm   = dlt.query_ts('wt4tcs', 'TEL.AMBI.FWHM', initial_time, final_time)



# %%
from elasticsearch import Elasticsearch
es = Elasticsearch("datalab.pl.eso.org:9200")

import elasticsearch_dsl as es_dsl
es_dsl.connections.create_connection(hosts='datalab.pl.eso.org')

system = 'galacsi'

# file_location = '/datalake/rawdata/aof/' + system.upper() + '_HEALTHCHECK/'
# s = es_dsl.Search(using = es, index='dev_galacsi_health_checks')
s = es_dsl.Search(using = es, index='galacsi_health_checks')
# all folders
# file_location = '/datalake/rawdata/aof/'
# folders = os.listdir(file_location)
s = s.sort('-timestamp')
# s = s.query('match', keywname = "VIS_WFS_temperature")
s = s.query()

indices = es.cat.indices(format='json')

def print_result(search):
    result = search.execute()['hits']['hits']
    for i in result:
        for j in i['_source']:
            print(j, ': ', i['_source'][j])
        print()

print_result(s)

#%%

# AUTREP Keywords
#•	wmfsgw
#AOS ATM TIMESTAMPSEC: [UTC] Timestamp of the data in seconds(o).
#AOS CNSQ CNSQ_i: [m^-2/3] Absolute Cn2 at altitude i.
#AOS CNSQ FRACCNSQ_i: [%] Fraction of Cn2 at altitude i.
#AOS CNSQ Hi: [km] Altitude of turbulent layer i.
#AOS CNSQ L0TOT: [m] Total outerscale.
#AOS CNSQ L0_i: [m] Outerscale at altitude i.
#AOS CNSQ R0TOT: [m] Total fried parameter.
#AOS CNSQ SEEINGTOT: [acrsec] Total seeing.
#AOS CNSQ TIMESTAMPSEC: [UTC] Timestamp of the data in seconds(o).
#AOS DSM MODE1: [nm] Amount of focus in the DSM commands offloaded to the M2 hexapod
#AOS DSM MODE2: [nm] Amount of coma 1 in the DSM commands offloaded to the M2 hexapod
#AOS DSM MODE3: [nm] Amount of coma 2 in the DSM commands offloaded to the M2 hexapod
#AOS DSM MODEi: [nm] Amount of the elastic mode i in the DSM commands offloaded to the M1
#AOS IR SAi FLUX: [ADU] IRLOS subaperture flux
#AOS IR SAi SPOTSIZEi: [pix] Fwhm of the major axis of the spot in the IRLOS subaperture.
#AOS IR TTRMS X: [pix] IRLOS residual tip/tilt RMS - x axis
#AOS IR TTRMS Y: [pix] IRLOS residual tip/tilt RMS - y axis
#AOS IRCTR SUBSTATE: LO/TTS Controller Loop State / 41 idle, 44 open, 45 closed, 46 suspend
#AOS JITCTR SUBSTATE: LGS Jitter Controller Loop State / 41 idle, 44 open, 45 closed, 46 suspend
#AOS LGSi L0: [m] Outer scale of the turbulence, as measured by the WFSi
#AOS LGSi R0: [m] Fried parameter along the line of sight and at 0.5 microns, as measured by the WFSi.
#AOS LGSi SPOTSIZE1: [pix] Fwhm of the small axis of the LGSi (0.83 arcsec/pixel)
#AOS LGSi SPOTSIZE2: [pix] Elongation of LGS image vs distance between the WFSi subaperture and the launch telescope pupil of the LGSUi.
#AOS LGSi TAU0: [s] Coherence time of the turbulence (tau0) along the line of sight and at 0.5 microns.
#AOS LGSCTR SUBSTATE: HO Controller Loop State / 41 idle, 44 open, 45 closed, 46 suspend.
#AOS LGSi DARKi: [ADU] Quadrant i dark / Dark value, as measured in the quadrant i of detector i median value (0.17 photons/ADU at gain 100).
#AOS LGSi EEIMP: Encircled Energy Improvement.
#AOS LGSi FLUX: [ADU] Median flux in subapertures / Median of the LGSi flux in all subapertures with 0.17 photons/ADU when the camera gain is 100.
#AOS LGSi FSM X: [SPARTA unit] FSM X-axis / LGSUi Field Steering Mirror command - X axis.
#AOS LGSi FSM Y: [SPARTA unit] FSM Y-axis / LGSUi Field Steering Mirror command - Y axis.
#AOS LGSi FWHM GAIN: Ratio between the fwhm of the uncorrected PSF and of the corrected one as measured by LGS WFSi.
#AOS LGSi FWHMIMP: FWHM improvement.
#AOS LGSi ROTATION: [deg] DSM clocking / Clocking of the DSM w.r.t. the WFSi.
#AOS LGSi SEEING: [arcsec] Seeing / Seeing along the line of sight and at 0.5 microns, as measured by the WFSi.
#AOS LGSi SLOPERMSX: [pix] X slopes rms / Median of the X slopes std dev measured in LGS WFSi subap (0.83 arcsec/pixel).
#AOS LGSi SLOPERMSY: [pix] Y slopes rms / Median of the Y slopes std dev measured in LGS WFSi subap (0.83 arcsec/pixel).
#AOS LGSi STREHL: [%] Strehl ratio / Strehl Ratio as computed from LGS WFSi measurements at science wavelength.
#AOS LGSi TUR ALT: Altitude turb fraction / Fraction of turbulence not corrected by AO loop - LGS WFSi, (high altitude layer)
#AOS LGSi TUR GND: Ground turb fraction / Fraction of turbulence corrected by AO loop - LGS WFSi, (ground layer).
#AOS LGSi TURVAR RES: [rad^2] Residual variance / Residual variance of aberrations along the line of sight and at 0.5 microns - LGS WFSi.
#AOS LGSi TURVAR TOT: [rad^2] Turb total variance / Total variance of the aberrations along the line of sight and at 0.5 microns - LGS WFSi.
#AOS LGSi VWIND: [m/s] Turb wind speed / Average wind speed along the line of sight, integrated on the turbulence layers.
#AOS LGSi XPUP: X Pupil Shift
#AOS LGSi XSHIFT: [subap %] DSM X shift / Lateral motion of the DSM w.r.t. the WFSi along the X axis - unit is % of one subaperture.
#AOS LGSi XSTRETCH: [subap %] DSM X stretch
#AOS LGSi YPUP: Y Pupil Shift
#AOS LGSi YSHIFT: [subap %] DSM Y shift / Lateral motion of the DSM w.r.t. the WFSi along the Y axis - unit is % of one subaperture.
#AOS LGSi YSTRETCH: [subap %] DSM Y stretch
#AOS MCMCTR SUBSTATE: MCM Controller Loop State / 41 idle, 44 open, 45 closed, 46 suspend.
#AOS NA HEIGHT: [m] Na layer altitude.
#AOS NGCIR TEMP: [deg] NGC Head temp.
#AOS NGCLGSi TEMP: [deg] NGC Head temp.
#AOS NGCVIS TEMP: [deg] NGC Head temp.
#AOS NGS FLUX: [ADU] NGS flux / NGS flux (0.17 photons/ADU at gain 100).
#AOS NGS SPOTANGLE: [rad] NGS spot orientation angle
#AOS NGS SPOTSIZE1: [pix] NGS minor axis fwhm / Fwhm of the minor axis of the NGS image as measured in the tip-tilt sensor (0.167 arcsec/pixel).
#AOS NGS SPOTSIZE2: [pix] NGS major axis fwhm / Fwhm of the major axis of the NGS image as measured in the tip-tilt sensor (0.167 arcsec/pixel).
#AOS NGS TTRMS X: [pix] NGS residual tip/tilt RMS - x axis
#AOS NGS TTRMS Y: [pix] NGS residual tip/tilt RMS - y axis
#AOS VISCTR SUBSTATE: LO/TTS Controller Loop State / 41 idle, 44 open, 45 closed, 46 suspend.
#
#•	wasm:
#INS DIMM AG DIT: [s] Autoguider base exposure time. This value is logged once at startup and on any change.
#INS DIMM AG ENABLE: Autoguider enable status. This value is logged once at startup and on any change.
#INS DIMM AG KIX: Autoguider integral gain X. This value is logged once at startup and on any change.
#INS DIMM AG KIY: Autoguider integral gain Y. This value is logged once at startup and on any change.
#INS DIMM AG KPX: Autoguider proportional gain X. This value is logged once at startup and on any change.
#INS DIMM AG KPY: Autoguider proportional gain Y. This value is logged once at startup and on any change.
#INS DIMM AG LOCKED: Locked on target. This action is logged when the target star first enters the field (asmdim).
#INS DIMM AG RADIUS: [arcsec] Autoguider in-target radius. Autoguider in-target radius for centering.
# This value is logged once at startup and on any change.
#INS DIMM AG RATE: Autoguider update rate. Autoguider update rate for averaging exposures.
# This value is logged once at startup and on any change.
#INS DIMM AG REFX: [pixel] Autoguider reference pixel X.Autoguider TCCD reference pixel for centroiding.
# This value is logged once at startup and on any change.
#INS DIMM AG REFY: [pixel] Autoguider reference pixel Y.Autoguider TCCD reference pixel for centroiding.
# This value is logged once at startup and on any change.
#INS DIMM AG START: Initiating setup to center target star. This action is logged at start of centering (asmdim).
#INS DIMM AG STOP: Star is inside target radius. This action is logged at end of centering (asmdim).
#INS DIMM CONST AF: CCD noise floor (explicit ad-count if >0, implicit n-sigma if <0). This value is logged once at startup and on any change.
#INS DIMM CONST CCDX: [pixel] X CCD total size. This value is logged once at startup and on any change.
#INS DIMM CONST CCDY: [pixel] Y CCD total size. This value is logged once at startup and on any change.
#INS DIMM CONST FL: [m] Objective Focal Length. This value is logged once at startup and on any change.
#INS DIMM CONST FR: Telescope F Ratio. This value is logged once at startup and on any change.
#INS DIMM CONST INX: [pixel^2] X Instrumental Noise (measured). This value is logged once at startup and on any change.
#INS DIMM CONST INY: [pixel^2] Y Instrumental Noise (measured). This value is logged once at startup and on any change.
#INS DIMM CONST LAMBDA: [m] Wavelength. This value is logged once at startup and on any change.
#INS DIMM CONST MD: [m] Mask Hole Diameter. This value is logged once at startup and on any change.
#INS DIMM CONST MS: [m] Mask Hole Separation. This value is logged once at startup and on any change.
#INS DIMM CONST PA: [deg] Pupil Mask Prism Angle. This value is logged once at startup and on any change.
#INS DIMM CONST PD: [m] Pupils Diameter. This value is logged once at startup and on any change.
#INS DIMM CONST PHI: [m] Telescope Diameter. This value is logged once at startup and on any change.
#INS DIMM CONST PPX: [arcsec] X Pixel Angular Pitch. This value is logged once at startup and on any change.
#INS DIMM CONST PPY: [arcsec] Y Pixel Angular Pitch. This value is logged once at startup and on any change.
#INS DIMM CONST PS: [m] Pupils Separation. This value is logged once at startup and on any change.
#INS DIMM CONST PSX: [m] X Pixel Size. This value is logged once at startup and on any change.
#INS DIMM CONST PSY: [m] Y Pixel Size. This value is logged once at startup and on any change.
#INS DIMM CONST W0: [pixel] Spot separation (nominal). This value is logged once at startup and on any change.
#INS DIMM LIMIT KC: Gaussian Clipping Limit (rms of distribution). This value is logged once at startup and on any change.
#INS DIMM LIMIT KE: [%] Spot Elongation Limit. This value is logged once at startup and on any change.
#INS DIMM LIMIT KF: Spot Decentering Limit (times of FWHM). This value is logged once at startup and on any change.
#INS DIMM LIMIT KN1: [%] Rejected Exposures Limit. This value is logged once at startup and on any change.
#INS DIMM LIMIT KN2: [%] Clipped Exposures Limit. This value is logged once at startup and on any change.
#INS DIMM LIMIT KO: Polynomial order for error-curve fitting (0=off). This value is logged once at startup and on any change.
#INS DIMM LIMIT KP: Valid Pixels Limit. This value is logged once at startup and on any change.
#INS DIMM LIMIT KR: Saturation Limit (ad count). This value is logged once at startup and on any change.
#INS DIMM LIMIT KS: Statistical Error Limit 1/sqrt(N-1). This value is logged once at startup and on any change.
#INS DIMM LIMIT KS1: Max number of consecutive rejected sequences for a star (0=off). If the number is reached then a change of star is requested.
# This value is logged once at startup and on any change.
#INS DIMM LIMIT KS2: Max total number of rejected sequences for a star (0=off). If the number is reached then a change of star is requested.
# This value is logged once at startup and on any change.
#INS DIMM LIMIT SNMIN: S/N Ratio Limit. This value is logged once at startup and on any change.
#INS DIMM MODE N: Frames per sequence. This value is logged once at startup and on any change.
#INS DIMM MODE NAME: Executing mode name (DIMM or WFCE). This value is logged once at startup and on any change.
#INS DIMM MODE ND: Exposures per frame. This value is logged once at startup and on any change.
#INS DIMM MODE NX: [pixel] X CCD sub-array size. This value is logged once at startup and on any change.
#INS DIMM MODE NY: [pixel] Y CCD sub-array size. This value is logged once at startup and on any change.
#INS DIMM MODE TD: [s] Exposure time. his value is logged once at startup and on any change.
#INS DIMM SEQ START: Seeing measurement sequence started. This action is logged when a measurement sequence is started.
#INS DIMM SEQ STOP: Seeing measurement sequence completed. This action is logged when a measurement sequence is completed.
#INS DIMM SEQIN AIRMASS: Airmass of target star. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN CCDTEMP: CCD chip temperature before sequence. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN DEC: [deg] Position in DEC of target star. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN RA: [deg] Position in RA of target star. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN MAG: [mag] Magnitude of target star. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN N: Number of requested exposures for sequence. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN SKYBG: Sky background input. This value is logged at the beginning of each sequence.
#INS DIMM SEQIN ZD: [deg] Zenith distance of target star. This value is logged at the beginning of each sequence.
#INS DIMM SEQOUT ALPHAG: [deg] Spot Axis Rotation vs CCD-x (measured). This value is logged at the end of each sequence.
#INS DIMM SEQOUT CCDTEMP: CCD chip temperature after sequence. This value is logged at the end of each sequence.
#INS DIMM SEQOUT DIT: [s] Average Exposure Time (measured). This value is logged at the end of each sequence.
#INS DIMM SEQOUT DITVAR: [s^2] Variance on Exposure Time. This value is logged at the end of each sequence.
#INS DIMM SEQOUT DN1: Number of exposures rejected in the frame validation. This value is logged at the end of each sequence.
#INS DIMM SEQOUT DN2: Number of exposures rejected in the statistical validation. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FLUXi: Flux of window 1/2. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FWHM: [arcsec] Seeing average. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FWHMPARA: [arcsec] Seeing parallel. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FWHMPERP: [arcsec] Seeing perpendicular. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FXi: [pixel] Average FWHM(x) of spot 1/2. This value is logged at the end of each sequence.
#INS DIMM SEQOUT FYi: [pixel] Average FWHM(y) of spot 1/2. This value is logged at the end of each sequence.
#INS DIMM SEQOUT NL: Long exposure validated exposures. This value is logged at the end of each sequence.
#INS DIMM SEQOUT NVAL: Number of validated exposures for sequence. This value is logged at the end of each sequence.
#INS DIMM SEQOUT RFLRMS: [ratio] Relative rms flux variation. This value is logged at the end of each sequence.
#INS DIMM SEQOUT SI: [%] Scintillation Index avg. This value is logged at the end of each sequence.
#INS DIMM SEQOUT SKYBGVAR: Variance on Sky background. This value is logged at the end of each sequence.
#INS DIMM SEQOUT SKYBGi: Sky background of window 1/2. This value is logged at the end of each sequence.
#INS DIMM SEQOUT SNR: Signal-to-Noise ratio. This value is logged at the end of each sequence.
#INS DIMM SEQOUT Si: [%] Scintillation Index of spot 1/2. This value is logged at the end of each sequence.
#INS DIMM SEQOUT TAU0: [s] Average coherence time. This value is logged at the end of each sequence.
#INS DIMM SEQOUT THETA0: [arcsec] Isoplanatic angle. This value is logged at the end of each sequence
#INS DIMM SEQOUT TS: [s] Long exposure time. This value is logged at the end of each sequence
#INS DIMM SEQOUT VARPARA: [arcsec^2] Variance of differential motion parallel. This value is logged at the end of each sequence
#INS DIMM SEQOUT VARPARAN: [arcsec^2] Variance of differential motion parallel (without fitting). This value is logged at the end of each sequence
#INS DIMM SEQOUT VARPERP: [arcsec^2] Variance of differential motion perpendicular. This value is logged at the end of each sequence
#INS DIMM SEQOUT VARPERPN: [arcsec^2] Variance of differential motion perpendicular (without fitting). This value is logged at the end of each sequence
#INS DIMM SEQOUT WG: [pixel] Spot Separation (measured). This value is logged at the end of each sequence
#INS DIMM SEQOUT XG: [pixel] Image cdg X (measured). This value is logged at the end of each sequence
#INS DIMM SEQOUT YG: [pixel] Image cdg Y (measured). This value is logged at the end of each sequence
#INS METEO START: Start recording of meteorological data. This action is logged when the interface to Meteo is started (asmmet).
#INS METEO STOP: Start recording of meteorological data. This action is logged when the interface to Meteo is stopped (asmmet).
#TEL POS DEC: [deg] %HOURANG: actual telescope position DEC J2000 from backward calculation. This position is logged at regular intervals.
#TEL POS RA: [deg] %HOURANG: actual telescope position RA J2000 from backward calculation. This position is logged at regular intervals.
#TEL POS THETA: [deg] actual telescope position in THETA E/W. This position is logged at intervals of typically 1 minute.
#TEL PRESET NAME: Name of the target star. This action is logged when presetting to a new target star (asmws).

#%%


from elasticsearch import Elasticsearch
es = Elasticsearch("datalab.pl.eso.org:9200")

import elasticsearch_dsl as es_dsl
es_dsl.connections.create_connection(hosts='datalab.pl.eso.org')

# import dlt
import hashlib
import pandas as pd

def print_result(search):
    result = search.execute()['hits']['hits']
    for i in result:
        for j in i['_source']:
            print(j, ': ', i['_source'][j])
        print()

s = es_dsl.Search(using = es, index = 'dev_galacsi_health_checks')
s = s.sort('-timestamp')
s = s.query('match', keywname = "LGS1_JIT_YREF")
print_result(s)# s = es_dsl.Search(using = es, index = 'dev_galacsi_health_checks')
# s = s.sort('-timestamp')
# s = s.query('match', keywname = "LGS1_JIT_YREF")
# print_result(s)
