#%%
import sys
sys.path.insert(0, '..')

import numpy as np
import torch
from torch import fft
from torch.nn.functional import interpolate
import scipy.special as spc
from astropy.io import fits
import time

from tools.utils import rad2mas, rad2arc, deg2rad, r0_new, pdims


class TipToy(torch.nn.Module):
    def InitValues(self):
        # Reading parameters from the file
        num_src = self.config['NumberSources']
        self.Nsrc = num_src.int().item()
        self.wvl = self.config['sources_science']['Wavelength']
        if not torch.all(self.wvl  == self.wvl[0]).item():
            raise ValueError('All wavelength must be the same for all samples')
        self.wvl = self.wvl[0].item()
        
        # Setting internal parameters
        self.D       = self.config['telescope']['TelescopeDiameter']
        self.psInMas = self.config['sensor_science']['PixelScale'] #[mas]
        self.nPix    = self.config['sensor_science']['FieldOfView'].int().item()
        self.pitch   = self.config['DM']['DmPitchs'] #[m]
        #self.h_DM    = self.AO_config['DM']['DmHeights'] # ????? what is h_DM?
        #self.nDM     = 1
        self.kc      = 1/(2*self.pitch)

        #self.zenith_angle  = torch.tensor(self.AO_config['telescope']['ZenithAngle'], device=self.device) # [deg] #TODO: telescope zenith != sample zenith?
        self.zenith_angle  = self.config['telescope']['ZenithAngle']
        self.airmass       = 1.0 / torch.cos(self.zenith_angle * deg2rad)

        self.GS_wvl     = self.config['sources_HO']['Wavelength'][0] #[m]
        #self.GS_height  = self.AO_config['sources_HO']['Height'] * self.airmass #[m]
        min_2d = lambda x: x if x.dim() == 2 else x.unsqueeze(1)
        self.wind_speed  = self.config['atmosphere']['WindSpeed']
        self.wind_dir    = self.config['atmosphere']['WindDirection']
        self.Cn2_weights = min_2d(self.config['atmosphere']['Cn2Weights'])
        self.Cn2_heights = min_2d(self.config['atmosphere']['Cn2Heights']) * self.airmass.unsqueeze(1) # [m]
        #self.stretch     = 1.0 / (1.0-self.Cn2_heights/self.GS_height)
        self.h           = self.Cn2_heights #* self.stretch
        self.nL          = self.Cn2_heights.size(0)


        self.WFS_d_sub = self.config['sensor_HO']['SizeLenslets'] #TODO: seems like it's absent
        self.WFS_n_sub = self.config['sensor_HO']['NumberLenslets']

        self.WFS_det_clock_rate = self.config['sensor_HO']['ClockRate'].flatten() # [(?)]
        self.WFS_FOV = self.config['sensor_HO']['FieldOfView']
        self.WFS_RON = self.config['sensor_HO']['SigmaRON']
        self.WFS_psInMas = self.config['sensor_HO']['PixelScale']
        self.WFS_wvl = self.make_tensor(self.GS_wvl) #TODO: clarify this
        self.WFS_spot_FWHM = self.make_tensor(self.config['sensor_HO']['SpotFWHM'][0])
        self.WFS_excessive_factor = self.config['sensor_HO']['ExcessNoiseFactor']
        self.WFS_Nph = self.config['sensor_HO']['NumberPhotons']

        self.HOloop_rate  = self.config['RTC']['SensorFrameRate_HO'] # [Hz] (?)
        self.HOloop_delay = self.config['RTC']['LoopDelaySteps_HO'] # [ms] (?)
        self.HOloop_gain  = self.config['RTC']['LoopGain_HO']


    def InitGrids(self):
        pixels_per_l_D = self.wvl*rad2mas / (self.psInMas*self.D)

        self.sampling_factor = int(np.ceil(2.0/pixels_per_l_D)) # check how much it is less than Nyquist
        self.sampling = self.sampling_factor * pixels_per_l_D
        self.nOtf = self.nPix * self.sampling_factor

        self.dk = 1/self.D/self.sampling # PSD spatial frequency step
        self.cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2/(2*np.pi**(11/3)))

        # Initialize spatial frequencies
        self.kx, self.ky = torch.meshgrid(
            torch.linspace(-self.nOtf/2, self.nOtf/2-1, self.nOtf, device=self.device)*self.dk + 1e-10,
            torch.linspace(-self.nOtf/2, self.nOtf/2-1, self.nOtf, device=self.device)*self.dk + 1e-10,
            indexing = 'ij')

        self.k2 = self.kx**2 + self.ky**2
        self.k = torch.sqrt(self.k2)

        self.mask = torch.ones_like(self.k2, device=self.device)
        self.mask[self.k2 <= self.kc**2] = 0
        self.mask_corrected = 1.0-self.mask

        self.nOtf_AO = int(2*self.kc/self.dk)
        self.nOtf_AO += self.nOtf_AO % 2

        # Comb samples involved in antialising
        n_times = min(4,max(2,int(np.ceil(self.nOtf/self.nOtf_AO/2))))
        ids = []
        for mi in range(-n_times,n_times):
            for ni in range(-n_times,n_times):
                if mi or ni: #exclude (0,0)
                    ids.append([mi,ni])
        ids = np.array(ids)

        m = self.make_tensor(ids[:,0])
        n = self.make_tensor(ids[:,1])
        self.N_combs = m.shape[0]

        corrected_ROI = slice(self.nOtf//2-self.nOtf_AO//2, self.nOtf//2+self.nOtf_AO//2)
        corrected_ROI = (corrected_ROI,corrected_ROI)

        self.mask_AO = self.mask[corrected_ROI]
        self.mask_corrected_AO = pdims( self.mask_corrected[corrected_ROI], -1 )
        self.mask = pdims( self.mask, -1 )

        self.kx_AO = pdims( self.kx[corrected_ROI], -1 )
        self.ky_AO = pdims( self.ky[corrected_ROI], -1 )
        self.k_AO  = pdims( self.k [corrected_ROI], -1 )
        self.k2_AO = pdims( self.k2[corrected_ROI], -1 )

        self.kx = pdims( self.kx, -1 )
        self.ky = pdims( self.ky, -1 )
        self.k  = pdims( self.k,  -1 )
        self.k2 = pdims( self.k2, -1 )

        # For NGS-like alising 0th dimension is used to store shifted spatial frequency
        self.km = self.kx_AO.repeat([self.N_combs,1,1,1]) - pdims(m/self.WFS_d_sub,2).unsqueeze(3)
        self.kn = self.ky_AO.repeat([self.N_combs,1,1,1]) - pdims(n/self.WFS_d_sub,2).unsqueeze(3)
        
        # Initialize OTF frequencines
        self.U, self.V = torch.meshgrid(
            torch.linspace(0, self.nOtf-1, self.nOtf, device=self.device),
            torch.linspace(0, self.nOtf-1, self.nOtf, device=self.device),
            indexing = 'ij')

        self.U = pdims( (self.U-self.nOtf/2) * 2/self.nOtf, -1 )
        self.V = pdims( (self.V-self.nOtf/2) * 2/self.nOtf, -1 )

        self.U2  = self.U**2
        self.V2  = self.V**2
        self.UV  = self.U*self.V
        self.UV2 = self.U**2 + self.V**2

        pupil_path = self.config['telescope']['PathPupil']
        pupil_apodizer = self.config['telescope']['PathApodizer']

        pupil    = self.make_tensor(fits.getdata(pupil_path).astype('float'))
        apodizer = self.make_tensor(fits.getdata(pupil_apodizer).astype('float'))

        pupil_pix  = pupil.shape[0]
        padded_pix = int(pupil_pix*self.sampling)

        pupil_padded = torch.zeros([padded_pix, padded_pix], device=self.device)
        pupil_padded[
            padded_pix//2-pupil_pix//2 : padded_pix//2+pupil_pix//2,
            padded_pix//2-pupil_pix//2 : padded_pix//2+pupil_pix//2
        ] = pupil*apodizer

        def fftAutoCorr(x):
            x_fft = fft.fft2(x)
            return fft.fftshift( fft.ifft2(x_fft*torch.conj(x_fft))/x.size(0)*x.size(1) )

        self.OTF_static = pdims(torch.real(fftAutoCorr(pupil_padded)), -2)
        self.OTF_static = interpolate(self.OTF_static, size=(self.nOtf,self.nOtf), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        self.OTF_static = pdims( self.OTF_static / self.OTF_static.max(), -1 )

        self.PSD_padder = torch.nn.ZeroPad2d((self.nOtf-self.nOtf_AO)//2)

        # Piston filter
        def PistonFilter(f):
            x = (np.pi*self.D*f).cpu().numpy() #TODO: find Bessel analog for pytorch
            R = spc.j1(x)/x
            piston_filter = self.make_tensor(1.0-4*R**2)
            piston_filter[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 0.0
            return piston_filter

        self.piston_filter = PistonFilter(self.k_AO)
        self.PR = PistonFilter(torch.hypot(self.km,self.kn))

        # Diffraction-limited PSF
        self.PSF_DL = interpolate( \
            torch.abs(fft.fftshift(fft.ifft2(fft.fftshift(self.OTF_static)))).unsqueeze(0), \
            size=(self.nPix,self.nPix), mode='area' ).squeeze(0).squeeze(0)
        

    def Update(self, reinit_grids=True):
        self.InitValues()
        if reinit_grids: self.InitGrids()


    def __init__(self, AO_config, norm_regime='sum', device=torch.device('cpu')):
        self.device = device
        self.make_tensor = lambda x: torch.tensor(x, device=self.device)

        if self.device.type is not 'cpu':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end   = torch.cuda.Event(enable_timing=True)

        super().__init__()

        self.norm_regime = norm_regime
        self.norm_scale  = self.make_tensor(1.0)

        # Read data and initialize AO system
        self.config = AO_config
        self.InitValues()
        self.InitGrids()

        self.PSDs = {}


    def Controller(self, nF=1000):
        #nTh = 1
        Ts = 1.0 / self.HOloop_rate # samplingTime
        delay = self.HOloop_delay #latency
        loopGain = self.HOloop_gain

        def TransferFunctions(freq, Ts, delay, loopGain):
            z = torch.exp(-2j*np.pi*freq*Ts)
            hInt = loopGain/(1.0 - z**(-1.0))
            rtfInt = 1.0/(1 + hInt*z**(-delay))
            atfInt = hInt * z**(-delay)*rtfInt
            ntfInt = atfInt / z
            return hInt, rtfInt, atfInt, ntfInt

        #f = torch.logspace(-3, torch.log10(torch.tensor([0.5/Ts])).item(), nF)
        f = torch.zeros([self.Nsrc,nF], device=self.device)
        for i in range(self.Nsrc): 
            f[i,:] = torch.logspace(-3, torch.log10(0.5/Ts[i]), nF)

        _, _, _, ntfInt = TransferFunctions(f, Ts.unsqueeze(1), delay.unsqueeze(1), loopGain.unsqueeze(1))
        self.noise_gain = pdims( torch.trapz(torch.abs(ntfInt)**2, f, dim=1)*2*Ts, 2 )

        thetaWind = self.make_tensor(0.0) #torch.linspace(0, 2*np.pi-2*np.pi/nTh, nTh)
        costh = torch.cos(thetaWind) #TODO: what is thetaWind?
        
        fi = -self.vx*self.kx_AO*costh - self.vy*self.ky_AO*costh
        _, _, atfInt, ntfInt = TransferFunctions(fi, pdims(Ts,2),
                                                     pdims(delay,2),
                                                     pdims(loopGain,2))

        # AO transfer function
        self.h1 = pdims(self.Cn2_weights.T,2) * atfInt.unsqueeze(0) #/nTh
        self.h2 = pdims(self.Cn2_weights.T,2) * abs(atfInt.unsqueeze(0))**2 #/nTh
        self.hn = pdims(self.Cn2_weights.T,2) * abs(ntfInt.unsqueeze(0))**2 #/nTh

        self.h1 = torch.sum(self.h1, axis=0) #summing along the weights dimension (4D->3D)
        self.h2 = torch.sum(self.h2, axis=0) 
        self.hn = torch.sum(self.hn, axis=0) 


    def ReconstructionFilter(self, r0, L0, WFS_noise_var):
        Av = torch.sinc(self.WFS_d_sub*self.kx_AO)*torch.sinc(self.WFS_d_sub*self.ky_AO) * torch.exp(1j*np.pi*self.WFS_d_sub*(self.kx_AO+self.ky_AO))
        self.SxAv = ( 2j*np.pi*self.kx_AO*self.WFS_d_sub*Av ).repeat([self.Nsrc,1,1])
        self.SyAv = ( 2j*np.pi*self.ky_AO*self.WFS_d_sub*Av ).repeat([self.Nsrc,1,1])

        MV = 0
        Wn = WFS_noise_var / (2*self.kc)**2
        # TODO: isn't this one computed for the WFSing wvl?
        #self.W_atm = self.cte*r0.**(-5/3) * (self.k2_AO + 1/L0**2)**(-11/6) * \
        #    (self.wvl/self.GS_wvl)**2 #TODO: check for SPHERE

        self.W_atm = self.VonKarmanSpectrum(r0, L0, self.k2_AO) * (self.wvl/self.GS_wvl)**2
        
        gPSD = torch.abs(self.SxAv)**2 + torch.abs(self.SyAv)**2 + MV*Wn/self.W_atm
        self.Rx = torch.conj(self.SxAv) / gPSD
        self.Ry = torch.conj(self.SyAv) / gPSD
        self.Rx[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 0
        self.Ry[..., self.nOtf_AO//2, self.nOtf_AO//2] *= 0

    
    def SpatioTemporalPSD(self):
        A = torch.ones([self.W_atm.shape[0], self.nOtf_AO, self.nOtf_AO], device=self.device) #TODO: fix it. A should be initialized differently
        Ff = self.Rx*self.SxAv + self.Ry*self.SyAv
        psd_ST = (1+abs(Ff)**2 * self.h2 - 2*torch.real(Ff*self.h1*A)) * self.W_atm * self.mask_corrected_AO
        return psd_ST


    def NoisePSD(self, WFS_noise_var):
        noisePSD = abs(self.Rx**2 + self.Ry**2) / (2*self.kc)**2
        noisePSD = noisePSD * self.piston_filter * self.noise_gain * WFS_noise_var * self.mask_corrected_AO
        return noisePSD


    def AliasingPSD(self, r0, L0):
        T = pdims(self.WFS_det_clock_rate / self.HOloop_rate, 2)
        td = T * pdims(self.HOloop_delay, 2)

        Rx1 = pdims(2j*np.pi*self.WFS_d_sub * self.Rx, -1)
        Ry1 = pdims(2j*np.pi*self.WFS_d_sub * self.Ry, -1)

        W_mn = (self.km**2 + self.kn**2 + 1/pdims(L0,-1)**2)**(-11/6)
        Q = (Rx1*self.km + Ry1*self.kn) * torch.sinc(self.WFS_d_sub*self.km) * torch.sinc(self.WFS_d_sub*self.kn)
        tf = pdims(self.h1,-1)

        avr = ( (self.Cn2_weights.T).unsqueeze(1).unsqueeze(3).unsqueeze(4) * \
            pdims( torch.sinc(self.km*self.vx*T) * torch.sinc(self.kn*self.vy*T) * \
            torch.exp(2j*np.pi*self.km*self.vx*td) * \
            torch.exp(2j*np.pi*self.kn*self.vy*td) * tf,-1) ).sum(axis=0)

        aliasing_PSD = torch.sum(self.PR*W_mn*abs(Q*avr)**2, axis=0) * self.cte*r0**(-5/3) * self.mask_corrected_AO
        return aliasing_PSD


    def VonKarmanSpectrum(self, r0, L0, freq2):
        return self.cte*r0**(-5/3) * (freq2 + 1/L0**2)**(-11/6)


    def VonKarmanPSD(self, r0, L0):
        return self.VonKarmanSpectrum(r0, L0, self.k2) * self.mask


    #TODO: polychromatic support
    def ChromatismPSD(self, r0, L0):
        wvlRef = self.wvl
        W_atm = self.VonKarmanSpectrum(r0, L0, self.k2_AO) * self.piston_filter
        IOR = lambda lmbd: 23.7+6839.4/(130-(lmbd*1.e6)**(-2))+45.47/(38.9-(lmbd*1.e6)**(-2))
        n2 = IOR(self.GS_wvl)
        n1 = IOR(wvlRef)
        chromatic_PSD = ((n2-n1)/n2)**2 * W_atm
        return chromatic_PSD


    def JitterCore(self, Jx, Jy, Jxy): #TODO: wavelength dependency!
        u_max = self.sampling*self.D/self.wvl/(3600*180*1e3/np.pi)
        norm_fact = u_max**2 * (2*np.sqrt(2*np.log(2)))**2
        Djitter = norm_fact * (Jx**2 * self.U2 + Jy**2 * self.V2 + 2*Jxy*self.UV)
        return torch.exp(-0.5*Djitter) #TODO: cover Nyquist sampled case


    def NoiseVariance(self, r0):
        r0_WFS = r0_new(r0.abs(), self.GS_wvl, 0.5e-6).flatten() #from (Nsrc x 1 x 1) to (Nsrc)
        WFS_nPix = self.WFS_FOV / self.WFS_n_sub
        WFS_pixelScale = self.WFS_psInMas / 1e3 # [arcsec]
       
        # Read-out noise calculation
        nD = torch.maximum( rad2arc*self.wvl/self.WFS_d_sub/WFS_pixelScale, self.make_tensor(1.0) ) #spot FWHM in pixels and without turbulence
        # Photon-noise calculation
        nT = torch.maximum( torch.hypot(self.WFS_spot_FWHM.max()/1e3, rad2arc*self.WFS_wvl/r0_WFS) / WFS_pixelScale, self.make_tensor(1.0) )

        varRON  = np.pi**2/3 * (self.WFS_RON**2/self.WFS_Nph**2) * (WFS_nPix**2/nD)**2
        varShot = np.pi**2/(2*self.WFS_Nph) * (nT/nD)**2
        
        # Noise variance calculation
        varNoise = self.WFS_excessive_factor * (varRON+varShot)

        return varNoise

    
    def DLPSF(self):
        if len(self.norm_scale.flatten()) > 1:
            return self.PSF_DL.unsqueeze(0) / self.norm_scale
        else:
            return self.PSF_DL / self.norm_scale


    def PSD2PSF(self, r0, L0, F, dx, dy, bg, dn, Jx, Jy, Jxy):
        r0  = pdims(r0,  2)
        L0  = pdims(L0,  2)
        F   = pdims(F,   2)
        dx  = pdims(dx,  2)
        dy  = pdims(dy,  2)
        bg  = pdims(bg,  2)
        dn  = pdims(dn,  2)
        Jx  = pdims(Jx,  2)
        Jy  = pdims(Jy,  2)
        Jxy = pdims(Jxy, 2)

        WFS_noise_var = torch.abs( dn + pdims( self.NoiseVariance(r0.abs()), 2) )
        self.vx = pdims( self.wind_speed*torch.cos(self.wind_dir*np.pi/180.0), 2 )
        self.vy = pdims( self.wind_speed*torch.sin(self.wind_dir*np.pi/180.0), 2 )

        self.Controller()
        self.ReconstructionFilter(r0.abs(), L0.abs(), WFS_noise_var)

        dk = 2*self.kc/self.nOtf_AO

        # All PSD components are computed in radians and here normalized to [nm^2]
        PSD_norm = lambda wvl: (dk*wvl*1e9/2/np.pi)**2
       
        # Most contributors are computed for 500 [nm], wavelengths-dependant inputs as well are assumed for 500 [nm]
        # But WFS operates on another wavelength, so this PSD components is normalized for WFSing wvl 
        # Put all contributiors together and sum up the resulting PSD
        self.PSDs = {
            'fitting':         self.VonKarmanPSD(r0.abs(),L0.abs()) * PSD_norm(500e-9),
            'WFS noise':       self.NoisePSD(WFS_noise_var) * PSD_norm(self.GS_wvl),
            'spatio-temporal': self.SpatioTemporalPSD() * PSD_norm(500e-9),
            'aliasing':        self.AliasingPSD(r0.abs(), L0) * PSD_norm(500e-9),
            'chromatism':      self.ChromatismPSD(r0.abs(), L0.abs()) * PSD_norm(500e-9)
        }

        PSD = self.PSDs['fitting'] + self.PSD_padder(
                    self.PSDs['WFS noise'] + \
                    self.PSDs['spatio-temporal'] + \
                    self.PSDs['aliasing'] + \
                    self.PSDs['chromatism'] )

        # Computing OTF from PSD
        cov = 2*fft.fftshift(fft.fft2(fft.fftshift(PSD))) # FFT axes are set to 1,2 by PyTorch by default
        # Computing the Structure Function from the covariance
        SF = pdims(torch.abs(cov).amax(dim=(1,2)),2) - cov
        # Phasor to shift the PSF with the subpixel accuracy
        fftPhasor  = torch.exp(-np.pi*1j*self.sampling_factor * (self.U*dx + self.V*dy))
        OTF_turb   = torch.exp(-0.5*SF*(2*np.pi*1e-9/self.wvl)**2)
        # Compute the residual tip/tilt kernel
        OTF_jitter = self.JitterCore(Jx.abs(), Jy.abs(), Jxy.abs())
        # Resulting OTF
        OTF = OTF_turb * self.OTF_static * fftPhasor * OTF_jitter

        PSF = torch.abs( fft.fftshift(fft.ifft2(fft.fftshift(OTF))) ).unsqueeze(0) # add 1 extra dimension to use PyTorch's interpolation function
        PSF_out = interpolate(PSF, size=(self.nPix,self.nPix), mode='area').squeeze(0) # shrink extra dimension back

        if self.norm_regime == 'max':
            self.norm_scale = torch.amax(PSF_out, dim=(1,2), keepdim=True)
        elif self.norm_regime == 'sum':
            self.norm_scale = PSF_out.sum(dim=(1,2), keepdim=True)

        # return PSF_out/self.norm_scale * F + bg #TODO: to put norm inside or not?
        return (PSF_out*F+bg)/self.norm_scale #TODO: to put norm inside or not?


    def forward(self, x):
        return self.PSD2PSF(*[x[:,i] for i in range(x.shape[1])])


    def StartTimer(self):
        if self.device.type == 'cpu':
            self.start = time.time()
        else:
            self.start.record()


    def EndTimer(self):
        if self.device.type == 'cpu':
            self.end = time.time()
            return (self.end-self.start)*1000.0 # in [ms]
        else:
            self.end.record()
            torch.cuda.synchronize()
            return self.start.elapsed_time(self.end)

# %%
