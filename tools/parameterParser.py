#%%
# IMPORTING PYTHON LIBRAIRIES
from ntpath import join
import os.path as ospath
from configparser import ConfigParser
from sys import path
import numpy as np
import os


class parameterParser():
    """
    Class to parse all the information about an AO system given in a .ini file.
    The parsed data is stored as dictionary that can be modified from inside the code
    """
    def __init__(self, path_root, path_ini):
        self.params = None

        # Root directory is detected automatically (one down the aoSystem dir)
        #path_root = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) 
        #path_ini  = os.path.join(path_root, 'aoSystem', 'parFiles', file_ini)

        # verify if the file exists
        if ospath.isfile(path_ini) == False:
            raise ValueError('The .ini file does not exist\n')

        # open the .ini file
        config = ConfigParser()
        config.optionxform = str
        config.read(path_ini)

        #Read config file and convert strings to actual values
        params = { s:dict(config.items(s)) for s in config.sections() }
        for section in params:
            for entry in params[section]:
                params[section][entry] = eval(params[section][entry])

        # Auxillary functions
        # Checks if an entry is present in the file. If no, the specified default value is set
        def checkValue(section, entry, default):
            if entry not in params[section]:
                params[section][entry] = default

        # Checks if path is present in the file. If no, path is set to None by default
        def checkPath(section, entry):
            if entry not in params[section]:
                params[section][entry] = None
            else:
                if params[section][entry][0] == '\\' or params[section][entry][0] == '/':
                    params[section][entry] = params[section][entry][1:]
                params[section][entry] = os.path.join(path_root, os.path.normpath(params[section][entry]))

        # -------- Grabbing main parameters -----------
        #%% Telescope
        if 'TelescopeDiameter' not in params['telescope']:
            raise ValueError('You must provide a value for the telescope diameter\n')

        ####### params['telescope']['airmass'] = 1.0 / np.cos(params['telescope']['ZenithAngle']*np.pi / 180.0)
        checkValue('telescope','ZenithAngle', 0.0)

        if 'Resolution' not in params['telescope']:
            raise ValueError('You must provide a value for the pupil resolution\n')

        # Pupil
        checkValue('telescope','ObscurationRatio', 0.0)
        checkValue('telescope','PupilAngle', 0.0)
        checkPath('telescope', 'PathPupil')
        checkPath('telescope', 'PathStaticOn')
        checkPath('telescope', 'PathStaticOff')
        checkPath('telescope', 'PathStaticPos')
        checkPath('telescope', 'PathApodizer')
        checkPath('telescope', 'PathStatModes') # Telescope aberrations

        # Atmosphere
        checkValue('atmosphere','Wavelength', 500e-9) #[m]

        if 'Seeing' not in params['atmosphere']:
            raise ValueError('You must provide a value for the seeing\n')

        checkValue('atmosphere','L0', 25) #[m]
        checkValue('atmosphere','Cn2Weights', [1.0]) #[m]
        checkValue('atmosphere','Cn2Heights', [0.0]) #[m]
        checkValue('atmosphere','WindSpeed', [10.0]) #[m]
        checkValue('atmosphere','WindDirection', [0.0]) #[m]

        if not len(params['atmosphere']['Cn2Weights']) == \
            len(params['atmosphere']['Cn2Heights']) == \
            len(params['atmosphere']['WindSpeed']) == \
            len(params['atmosphere']['WindDirection']):
            raise ValueError("The number of atmospheric layers is not consistent in the parameters file\n")

        # Guide stars
        if 'sources_HO' in params:
            if 'Wavelength' not in params['sources_HO']:
                raise ValueError("You must provide a value for the wavelength of the science source\n")

            params['sources_HO']['Wavelength'] = np.unique(np.array(params['sources_HO']['Wavelength']))

            checkValue('sources_HO','Zenith', [0.0])
            checkValue('sources_HO','Azimuth', [0.0])
            checkValue('sources_HO','Height', 0.0)

            if not len(params['sources_HO']['Zenith']) == len(params['sources_HO']['Azimuth']):
                raise ValueError("The number of guide stars for high-order sensing is not consistent in the parameters file\n")

        # Science sources
        if 'Wavelength' not in params['sources_science']:
            raise ValueError("You must provide a value for the wavelength of the science source\n")

        checkValue('sources_science','Zenith',  [0.0])
        checkValue('sources_science','Azimuth', [0.0])

        # High-order wavefront sensor
        if 'PixelScale' not in params['sensor_HO']:
            raise ValueError('You must provide a value for the HO detector pixel scale\n')

        if 'FieldOfView' not in params['sensor_HO']:
            raise ValueError("You must provide a value for the science detector field of view\n")

        if 'NumberLenslets' not in params['sensor_HO']:
            raise ValueError('You must provide a list of number of lenslets for the HO WFS\n')

        checkValue('sensor_HO','SizeLenslets', list(params['telescope']['TelescopeDiameter'] / np.array(params['sensor_HO']['NumberLenslets'])))
        checkValue('sensor_HO','Binning', 1)
        checkValue('sensor_HO','SpotFWHM', [[0.0, 0.0]])
        checkValue('sensor_HO','NumberPhotons', [np.inf])
        checkValue('sensor_HO','SigmaRON', 0.0)
        checkValue('sensor_HO','Gain', 1)
        checkValue('sensor_HO','SkyBackground', 0.0)
        checkValue('sensor_HO','Dark', 0.0)
        checkValue('sensor_HO','SpectralBandwidth', 0.0)
        checkValue('sensor_HO','Transmittance', [1.0])
        checkValue('sensor_HO','Dispersion', [[0.0], [0.0]])
        checkValue('sensor_HO','WfsType', 'Shack-Hartmann')
        checkValue('sensor_HO','Modulation', None)
        checkValue('sensor_HO','NoiseVariance', [None])
        checkValue('sensor_HO','ClockRate', [1,]*len(params['sensor_HO']['NumberLenslets']))
        checkValue('sensor_HO','Algorithm', 'wcog')
        checkValue('sensor_HO','WindowRadiusWCoG', 5.0)
        checkValue('sensor_HO','ThresholdWCoG', 0.0) # (?) = 0.0 or it's a mistake?
        checkValue('sensor_HO','NewValueThrPix', 0.0)
        checkValue('sensor_HO','ExcessNoiseFactor', 1.0)

        # Tip-tilt sensors
        if 'sensor_LO' in params:
            if 'PixelScale' not in params['sensor_LO']:
                raise ValueError("You must provide a value for the LO detector pixel scale\n")

            if 'FieldOfView' not in params['sensor_LO']:
                raise ValueError("You must provide a value for the science detector field of view\n")

            checkValue('sensor_LO','NumberLenslets', [1])
            checkValue('sensor_LO','SizeLenslets', list(params['telescope']['TelescopeDiameter'] / np.array(params['sensor_LO']['NumberLenslets'])))
            checkValue('sensor_LO','Binning', 1)
            checkValue('sensor_LO','SpotFWHM', [[0.0, 0.0]])
            checkValue('sensor_LO','NumberPhotons', np.inf)
            checkValue('sensor_LO','SigmaRON', 0.0)
            checkValue('sensor_LO','Gain', 1)
            checkValue('sensor_LO','SkyBackground', 0.0)
            checkValue('sensor_LO','Dark', 0.0)
            checkValue('sensor_LO','SpectralBandwidth', 0.0)
            checkValue('sensor_LO','Transmittance', [1.0])
            checkValue('sensor_LO','Dispersion', [[0.0], [0.0]])
            checkValue('sensor_LO','NoiseVariance', [None])
            checkValue('sensor_LO','WfsType', 'Shack-Hartmann')
            checkValue('sensor_LO','ClockRate', [1,]*len(params['sensor_LO']['NumberLenslets']))
            checkValue('sensor_LO','Algorithm', 'wcog')
            checkValue('sensor_LO','WindowRadiusWCoG', 5.0)
            checkValue('sensor_LO','ThresholdWCoG', 0.0)
            checkValue('sensor_LO','NewValueThrPix', 0.0)
            checkValue('sensor_LO','ExcessNoiseFactor', 1.0)

        # Real-time-computer
        checkValue('RTC','LoopGain_HO', 0.5)
        checkValue('RTC','SensorFrameRate_HO', 500.0)
        checkValue('RTC','LoopDelaySteps_HO', 2)
        checkValue('RTC','LoopGain_LO', None)
        checkValue('RTC','SensorFrameRate_LO', None)
        checkValue('RTC','LoopDelaySteps_LO', None)
        checkValue('RTC','ResidualError', None)

        # Deformable mirrors
        if 'NumberActuators' not in params['DM']:
            raise ValueError("You must provide a value for the DM actuators pitch\n")

        if 'DmPitchs' not in params['DM']:
            raise ValueError("You must provide a value for the Dm actuators pitch\n")

        checkValue('DM','InfModel', 'gaussian')
        checkValue('DM','InfCoupling', [0.2])
        checkValue('DM','DmHeights', [0.0])
        checkValue('DM','OptimizationWeight', [0.0])
        checkValue('DM','OptimizationAzimuth', [0.0])
        checkValue('DM','OptimizationZenith', [0.0])

        if len(params['DM']['OptimizationZenith']) != len(params['DM']['OptimizationAzimuth']) != len(params['DM']['OptimizationWeight']):
            raise ValueError("The number of optimization directions is not consistent in the parameters file\n")

        checkValue('DM','OptimizationConditioning', 100.0)
        checkValue('DM','NumberReconstructedLayers', 10)
        checkValue('DM','AoArea', 'circle')

        # Science detector
        if 'PixelScale' not in params['sensor_science']:
            raise ValueError("You must provide a value for the science detector pixel scale\n")
                                
        if 'FieldOfView' not in params['sensor_science']:
            ValueError("You must provide a value for the science detector field of view\n")

        checkValue('sensor_science','Name', 'SCIENCE CAM')
        checkValue('sensor_science','Binning', 1)
        checkValue('sensor_science','SpotFWHM', [[0.0, 0.0, 0.0]])
        checkValue('sensor_science','SpectralBandwidth', 0.0)
        checkValue('sensor_science','Transmittance', [1.0])
        checkValue('sensor_science','Dispersion', [[0.0], [0.0]])
        checkValue('sensor_science','NumberPhotons', np.inf)
        checkValue('sensor_science','Saturation', np.inf)
        checkValue('sensor_science','SigmaRON', 0.0)
        checkValue('sensor_science','Gain', 1)
        checkValue('sensor_science','SkyBackground', 0.0)
        checkValue('sensor_science','Dark', 0.0)
        checkValue('sensor_science','ExcessNoiseFactor', 1.0)

        self.params = params

# %%