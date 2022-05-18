#%%
import os
from os import path
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import path

dir_path = 'C:\\Users\\akuznets\\Data\\MUSE\\DATA_raw\\'

# M.MUSE.2018-06-22T08 23 22.730.fits

files = os.listdir(dir_path)
#with fits.open(path.join(dir_path,files[0])) as hdul:

file_id = 4

hdul = fits.open(path.join(dir_path,files[file_id]))

start_spaxel = hdul[1].header['CRPIX3']
num_λs = hdul[1].header['NAXIS3']-start_spaxel+1
Δλ = hdul[1].header['CD3_3' ] / 10.0
λ_min = hdul[1].header['CRVAL3'] / 10.0
λs = np.arange(num_λs)*Δλ + λ_min
λ_max = λs.max()

def CroppedROI(im, point, win):
    ids = np.zeros(4)
    ids[0] = np.max([point[0]-win//2, 0.0])
    ids[1] = np.min([point[0]+win//2, im.shape[0]])
    ids[2] = np.max([point[1]-win//2, 0.0])
    ids[3] = np.min([point[1]+win//2, im.shape[1]])
    ids = ids.astype('uint').tolist()
    return (slice(ids[0], ids[1]), slice(ids[2], ids[3])) #TODO: check, if axis are realy meaningful y=y_im, x=x_im

def GetROIaroundMax(im, win=70):
    ROI = CroppedROI(im, np.array(im.shape)//2, win)
    return np.array(np.unravel_index(np.argmax(im[ROI]), im[ROI].shape)) + np.array([im.shape[0]//2-win//2, im.shape[1]//2-win//2])

def GetSpectrum(radius=5):
    white = hdul[3].data
    id = GetROIaroundMax(white)
    xx,yy = np.meshgrid(np.linspace(0,white.shape[1]-1,white.shape[1]), np.linspace(0,white.shape[0]-1,white.shape[0]))
    spectral_ids = np.where(np.sqrt((xx-id[1])**2 + (yy-id[0])**2) < radius)

    spectrum = np.nansum(hdul[1].data[:,spectral_ids[0],spectral_ids[1]], axis=1)
    return spectrum

spectrum = GetSpectrum()

# Remove bad wavelengths ranges
bad_wvls = np.array([[450, 478], [577, 606]])
def find_closest(wvl,λs):
    return np.argmin(np.abs(λs-wvl)).astype('int')

bad_ids = np.zeros_like(bad_wvls.flatten(), dtype='int')
for i,wvl in enumerate(bad_wvls.flatten()):
    bad_ids[i] = find_closest(wvl, λs)
bad_ids = bad_ids.reshape(bad_wvls.shape)

valid_λs = np.ones_like(λs)
for i in range(len(bad_ids)):
    valid_λs[bad_ids[i,0]:bad_ids[i,1]+1] = 0
    bad_wvls[i,:] = np.array([λs[bad_ids[i,0]], λs[bad_ids[i,1]+1]])
#λs_new = λs[np.where(valid_λs==1)]

# Bin data cubes
#Before the sodium filter
λ_bin = (bad_wvls[1][0]-bad_wvls[0][1])/3.0
λ_bins_before = bad_wvls[0][1] + np.arange(4)*λ_bin
bin_ids_before = [find_closest(wvl, λs) for wvl in λ_bins_before]

#After the sodium filter
λ_bins_num = (λ_max-bad_wvls[1][1]) / np.diff(λ_bins_before).mean()
λ_bins_after = bad_wvls[1][1] + np.arange(λ_bins_num+1)*λ_bin
bin_ids_after = [find_closest(wvl, λs) for wvl in λ_bins_after]
bins_smart = bin_ids_before + bin_ids_after
λ_bins_smart = [λ for λ in λs[bins_smart]]

#TODO: mb sodium line calibration of wavelengths grid?

def wavelength_to_rgb(wavelength, gamma=0.8, show_invisible=False):
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        a = 0.4
        R = a
        G = a
        B = a
        if not show_invisible:
            R = 0.0
            G = 0.0
            B = 0.0
    return (R,G,B)

Rs = np.zeros_like(λs)
Gs = np.zeros_like(λs)
Bs = np.zeros_like(λs)

for i,λ in enumerate(λs): Rs[i],Gs[i],Bs[i] = wavelength_to_rgb(λ, show_invisible=True)

Rs = Rs * valid_λs * spectrum/np.median(spectrum)
Gs = Gs * valid_λs * spectrum/np.median(spectrum)
Bs = Bs * valid_λs * spectrum/np.median(spectrum)
colors = np.dstack([np.vstack([Rs, Gs, Bs])]*600).transpose(2,1,0)

fig_handler = plt.figure(dpi=200)
plt.imshow(colors, extent=[λs.min(), λs.max(), 0, 120])
plt.vlines(λ_bins_smart, 0, 120, color='white') #draw bins borders
plt.plot(λs, spectrum/spectrum.max()*120, linewidth=2.0, color='white')
plt.plot(λs, spectrum/spectrum.max()*120, linewidth=0.5, color='blue')
plt.xlabel(r"$\lambda$, [nm]")
ax = plt.gca()
ax.get_yaxis().set_visible(False)

misc_info = {
    'spectrum': spectrum,
    'wvl range': [λ_min, λ_max, Δλ], # num = int((λ_max-λ_min)/Δλ)+1
    'filtered wvls': bad_wvls,
    'wvl bins': λ_bins_smart,
}

#%%
white = hdul[3].data
max_id = GetROIaroundMax(white)
ROI = CroppedROI(white, max_id, 100)

data_reduced = np.zeros([white[ROI].shape[0], white[ROI].shape[1], len(λ_bins_smart)-1])
std_reduced  = np.zeros([white[ROI].shape[0], white[ROI].shape[1], len(λ_bins_smart)-1])
wavelengths  = np.zeros(len(λ_bins_smart)-1)

def ChunkMean(frame, row, col, win=1, exclude=True):
    # Prevent from going outside frame's dimensions
    row1 = max(0, row-win)
    row2 = min(row+win, frame.shape[0]-1)
    col1 = max(0, col-win)
    col2 = min(col+win, frame.shape[1]-1)

    chunck = np.copy(frame[row1:row2+1, col1:col2+1])
    if np.all(np.isnan(chunck)):
        return np.nan
    if exclude:
        chunck[(row-row1, col-col1)] = np.nan # block the pixel under consideration
    return np.nanmean(chunck) # averaged value

def HealPixels(frame):
    indexes = np.array(np.where(np.isnan(frame))).T
    while len(indexes) > 0:
        for idx in indexes:
            row,col = idx
            frame[row,col] = ChunkMean(frame, row,col)
        indexes = np.array(np.where(np.isnan(frame))).T
    return frame

for i in range(len(bins_smart)-1):
    buf1 = np.nansum(hdul[1].data[bins_smart[i]:bins_smart[i+1],ROI[0],ROI[1]], axis=0)
    buf2 = np.nanstd(hdul[1].data[bins_smart[i]:bins_smart[i+1],ROI[0],ROI[1]], axis=0)
    data_reduced[:,:,i] = HealPixels(buf1)
    std_reduced [:,:,i] = HealPixels(buf2)
    
    wavelengths[i] = (λ_bins_smart[i]+λ_bins_smart[i+1])/2.0

plt.imshow(np.log(data_reduced[:,:,0]))

#%%
# Collect important data from the header
data = {}
data['date']          = hdul[0].header['DATE']
data['date-obs']      = hdul[0].header['DATE-OBS']
data['RA']            = hdul[0].header['RA']
data['DEC']           = hdul[0].header['DEC']
data['exposure']      = hdul[0].header['EXPTIME ']
data['Strehl' ]       = hdul[0].header['HIERARCH ESO OBS STREHLRATIO']
data['target']        = hdul[0].header['HIERARCH ESO OBS TARG NAME']
data['airmass start'] = hdul[0].header['HIERARCH ESO TEL AIRM START']
data['airmass end']   = hdul[0].header['HIERARCH ESO TEL AIRM END']
data['altitude']      = hdul[0].header['HIERARCH ESO TEL ALT']
data['azimuth']       = hdul[0].header['HIERARCH ESO TEL AZ']
data['seeing start']  = hdul[0].header['HIERARCH ESO TEL AMBI FWHM START']
data['seeing end']    = hdul[0].header['HIERARCH ESO TEL AMBI FWHM END']
data['tau0']          = hdul[0].header['HIERARCH ESO TEL AMBI TAU0']
data['temperature']   = hdul[0].header['HIERARCH ESO TEL AMBI TEMP']
data['wind dir']      = hdul[0].header['HIERARCH ESO TEL AMBI WINDDIR']
data['wind speed']    = hdul[0].header['HIERARCH ESO TEL AMBI WINDSP']
hdul.close()

#%%
import pickle

packet = {
    'cube': data_reduced,
    'std': std_reduced,
    'data': data,
    'wavelengths': wavelengths,
    'misc info': misc_info
}
with open('C:/Users/akuznets/Data/MUSE/DATA_raw_binned/'+files[file_id].split('.fits')[0]+'.pickle', 'wb') as handle:
    pickle.dump(packet, handle, protocol=pickle.HIGHEST_PROTOCOL)

#import zlib
#
#compressed_packet = {
#    'cube': zlib.compress(data_reduced),
#    'std': zlib.compress(std_reduced),
#    'wavelengths': zlib.compress(wavelengths),
#    'data': data,
#    'misc info': misc_info
#}
#
#with open('C:/Users/akuznets/Data/MUSE/DATA_raw_binned/'+files[file_id].split('.fits')[0]+'_compressed.pickle', 'wb') as handle:
#    pickle.dump(compressed_packet, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#with open('C:/Users/akuznets/Data/MUSE/DATA_raw_binned/'+files[file_id].split('.fits')[0]+'_compressed.pickle', 'rb') as handle:
#    test = pickle.load(handle)
#
#a = np.array(zlib.decompress(test['cube']))
#b = np.array(zlib.decompress(test['std']))
#c = np.array(zlib.decompress(test['wavelengths']), dtype=np.float32)
