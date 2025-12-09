from project_settings import PROJECT_PATH, default_torch_type

import os.path as ospath
import warnings
from configparser import ConfigParser
import numpy as np
import os

"""
Class to parse all the information about an AO system given in a .ini file.
The parsed data is stored as dictionary that can be modified from inside the code
"""
class ParameterParser():
    # Check for the existance of the file in the directory. If it is not found, initializes top-down search for it
    def find_file(self, path_input):       
        path_check = str(path_input)
        if path_check.startswith('$PROJECT_PATH$'):
            path_check = str(PROJECT_PATH / path_check.replace('$PROJECT_PATH$/', ''))
                    
        if os.path.isfile(path_check):
            return path_check
        
        else:
            file = os.path.basename(path_check)
            warnings.warn(f'Warning: file "{file}" is not found! Initializing the search...')

            # Walking top-down from the root
            for root, _, files in os.walk(PROJECT_PATH):
                if file in files:
                    return os.path.join(root, file)
                
                warnings.warn(f'Warning: file "{file}" is not found!')
                return None


    def __init__(self, path_ini):
        # if not isinstance(path_ini, str):
        #     raise TypeError("The input must be a string representing a file path")
        
        self.params = None

        # verify if the file exists
        path_ini = self.find_file(path_ini)
        
        if path_ini is None:
            raise FileNotFoundError('File not found!')
        
        if ospath.isfile(path_ini) == False:
            raise FileNotFoundError(f'Cannot open \"{path_ini}\"!')

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
        def check_value(section, entry, default):
            if entry not in params[section]:
                params[section][entry] = default

        # Checks if path is present in the file. If no, path is set to None by default
        def check_path(section, entry):
            if entry not in params[section]:
                params[section][entry] = None
            else:
                params[section][entry] = self.find_file(params[section][entry])

        # -------- Grabbing main parameters -----------
        if 'TelescopeDiameter' not in params['telescope']:
            raise ValueError('You must provide a value for the telescope diameter\n')

        ####### params['telescope']['airmass'] = 1.0 / np.cos(params['telescope']['ZenithAngle']*np.pi / 180.0)
        check_value('telescope','ZenithAngle', 0.0)

        if 'Resolution' not in params['telescope']:
            raise ValueError('You must provide a value for the pupil resolution\n')

        # Pupil
        check_value('telescope', 'ObscurationRatio', 0.0)
        check_value('telescope', 'PupilAngle', 0.0)
        check_path('telescope',  'PathPupil')
        check_path('telescope',  'PathStaticOn')
        check_path('telescope',  'PathStaticOff')
        check_path('telescope',  'PathStaticPos')
        check_path('telescope',  'PathApodizer')
        check_path('telescope',  'PathStatModes') # Telescope aberrations

        # Atmosphere
        check_value('atmosphere','Wavelength', 500e-9) #[m]

        if 'Seeing' not in params['atmosphere']:
            raise ValueError('You must provide a value for the seeing\n')

        check_value('atmosphere', 'L0', 25) #[m]
        check_value('atmosphere', 'Cn2Weights', [1.0]) #[m]
        check_value('atmosphere', 'Cn2Heights', [0.0]) #[m]
        check_value('atmosphere', 'WindSpeed', [10.0]) #[m]
        check_value('atmosphere', 'WindDirection', [0.0]) #[m]

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

            check_value('sources_HO', 'Zenith', [0.0])
            check_value('sources_HO', 'Azimuth', [0.0])
            check_value('sources_HO', 'Height', 0.0)

            if not len(params['sources_HO']['Zenith']) == len(params['sources_HO']['Azimuth']):
                raise ValueError("The number of guide stars for high-order sensing is not consistent in the parameters file\n")

        # Science sources
        if 'Wavelength' not in params['sources_science']:
            raise ValueError("You must provide a value for the wavelength of the science source\n")

        check_value('sources_science','Zenith',  [0.0])
        check_value('sources_science','Azimuth', [0.0])

        # High-order wavefront sensor
        if 'PixelScale' not in params['sensor_HO']:
            raise ValueError('You must provide a value for the HO detector pixel scale\n')

        if 'FieldOfView' not in params['sensor_HO']:
            raise ValueError("You must provide a value for the science detector field of view\n")

        if 'NumberLenslets' not in params['sensor_HO']:
            raise ValueError('You must provide a list of number of lenslets for the HO WFS\n')

        check_value('sensor_HO', 'SizeLenslets', list(params['telescope']['TelescopeDiameter'] / np.array(params['sensor_HO']['NumberLenslets'])))
        check_value('sensor_HO', 'Binning', 1)
        check_value('sensor_HO', 'SpotFWHM', [[0.0, 0.0]])
        check_value('sensor_HO', 'NumberPhotons', [np.inf])
        check_value('sensor_HO', 'SigmaRON', 0.0)
        check_value('sensor_HO', 'Gain', 1)
        check_value('sensor_HO', 'SkyBackground', 0.0)
        check_value('sensor_HO', 'Dark', 0.0)
        check_value('sensor_HO', 'SpectralBandwidth', 0.0)
        check_value('sensor_HO', 'Transmittance', [1.0])
        check_value('sensor_HO', 'Dispersion', [[0.0], [0.0]])
        check_value('sensor_HO', 'WfsType', 'Shack-Hartmann')
        check_value('sensor_HO', 'Modulation', None)
        check_value('sensor_HO', 'NoiseVariance', [None])
        check_value('sensor_HO', 'ClockRate', [1,]*len(params['sensor_HO']['NumberLenslets']))
        check_value('sensor_HO', 'Algorithm', 'wcog')
        check_value('sensor_HO', 'WindowRadiusWCoG', 5.0)
        check_value('sensor_HO', 'ThresholdWCoG', 0.0) # (?) = 0.0 or it's a mistake?
        check_value('sensor_HO', 'NewValueThrPix', 0.0)
        check_value('sensor_HO', 'ExcessNoiseFactor', 1.0)

        # Tip-tilt sensors
        if 'sensor_LO' in params:
            if 'PixelScale' not in params['sensor_LO']:
                raise ValueError("You must provide a value for the LO detector pixel scale\n")

            if 'FieldOfView' not in params['sensor_LO']:
                raise ValueError("You must provide a value for the science detector field of view\n")

            check_value('sensor_LO', 'NumberLenslets', [1])
            check_value('sensor_LO', 'SizeLenslets', list(params['telescope']['TelescopeDiameter'] / np.array(params['sensor_LO']['NumberLenslets'])))
            check_value('sensor_LO', 'Binning', 1)
            check_value('sensor_LO', 'SpotFWHM', [[0.0, 0.0]])
            check_value('sensor_LO', 'NumberPhotons', np.inf)
            check_value('sensor_LO', 'SigmaRON', 0.0)
            check_value('sensor_LO', 'Gain', 1)
            check_value('sensor_LO', 'SkyBackground', 0.0)
            check_value('sensor_LO', 'Dark', 0.0)
            check_value('sensor_LO', 'SpectralBandwidth', 0.0)
            check_value('sensor_LO', 'Transmittance', [1.0])
            check_value('sensor_LO', 'Dispersion', [[0.0], [0.0]])
            check_value('sensor_LO', 'NoiseVariance', [None])
            check_value('sensor_LO', 'WfsType', 'Shack-Hartmann')
            check_value('sensor_LO', 'ClockRate', [1,]*len(params['sensor_LO']['NumberLenslets']))
            check_value('sensor_LO', 'Algorithm', 'wcog')
            check_value('sensor_LO', 'WindowRadiusWCoG', 5.0)
            check_value('sensor_LO', 'ThresholdWCoG', 0.0)
            check_value('sensor_LO', 'NewValueThrPix', 0.0)
            check_value('sensor_LO', 'ExcessNoiseFactor', 1.0)

        # Real-time controller
        check_value('RTC', 'LoopGain_HO', 0.5)
        check_value('RTC', 'SensorFrameRate_HO', 500.0)
        check_value('RTC', 'LoopDelaySteps_HO', 2)
        check_value('RTC', 'LoopGain_LO', None)
        check_value('RTC', 'SensorFrameRate_LO', None)
        check_value('RTC', 'LoopDelaySteps_LO', None)
        check_value('RTC', 'ResidualError', None)

        # Deformable mirrors
        if 'NumberActuators' not in params['DM']:
            raise ValueError("You must provide a value for the DM actuators pitch\n")

        if 'DmPitchs' not in params['DM']:
            raise ValueError("You must provide a value for the Dm actuators pitch\n")

        check_value('DM', 'InfModel', 'gaussian')
        check_value('DM', 'InfCoupling', [0.2])
        check_value('DM', 'DmHeights', [0.0])
        check_value('DM', 'OptimizationWeight', [0.0])
        check_value('DM', 'OptimizationAzimuth', [0.0])
        check_value('DM', 'OptimizationZenith', [0.0])

        if len(params['DM']['OptimizationZenith']) != len(params['DM']['OptimizationAzimuth']) != len(params['DM']['OptimizationWeight']):
            raise ValueError("The number of optimization directions is not consistent in the parameters file\n")

        check_value('DM', 'OptimizationConditioning', 100.0)
        check_value('DM', 'NumberReconstructedLayers', 10)
        check_value('DM', 'AoArea', 'circle')

        # Science detector
        if 'PixelScale' not in params['sensor_science']:
            raise ValueError("You must provide a value for the science detector pixel scale\n")
                                
        if 'FieldOfView' not in params['sensor_science']:
            ValueError("You must provide a value for the science detector field of view\n")

        check_value('sensor_science', 'Name', 'SCIENCE CAM')
        check_value('sensor_science', 'Binning', 1)
        check_value('sensor_science', 'SpotFWHM', [[0.0, 0.0, 0.0]])
        check_value('sensor_science', 'SpectralBandwidth', 0.0)
        check_value('sensor_science', 'Transmittance', [1.0])
        check_value('sensor_science', 'Dispersion', [[0.0], [0.0]])
        check_value('sensor_science', 'NumberPhotons', np.inf)
        check_value('sensor_science', 'Saturation', np.inf)
        check_value('sensor_science', 'SigmaRON', 0.0)
        check_value('sensor_science', 'Gain', 1)
        check_value('sensor_science', 'SkyBackground', 0.0)
        check_value('sensor_science', 'Dark', 0.0)
        check_value('sensor_science', 'ExcessNoiseFactor', 1.0)

        self.params = params
