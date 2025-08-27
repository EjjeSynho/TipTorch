import sys
sys.path.insert(0, '..')

from managers.parameter_parser import ParameterParser

import math
import torch
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np
    
from copy import deepcopy
from functools import reduce
import operator

"""
This module is used to manage the configuration files of TipTorch. Configuration files are used to set the up
the initial state of the TipTorch simulation. This module is used to parse the configuration files and manipulate
the configs, i.e., splitting and merging them, converting to different frameworks (NumPy, PyTorch, CuPy), etc.
Since TipTorch configs support for multiple targets, splitting and joining different configs is necessary.
"""

class ConfigManager():
    def __init__(self, uniqualized_values=None):
        #'uniqualized' paremeters are those are defined only one in a config file for all targets
        if uniqualized_values is None:
            self.uniqualized_values = [
                ['atmosphere', 'Wavelength'],
                ['DM', 'AoArea'],
                ['DM', 'DmHeights'],
                ['DM', 'DmPitchs'],
                ['DM', 'InfCoupling'],
                ['DM', 'InfModel'],
                ['DM', 'NumberActuators'],
                # ['DM', 'OptimizationAzimuth'],
                ['DM', 'OptimizationConditioning'],
                # ['DM', 'OptimizationWeight'],
                ['sensor_HO', 'Algorithm'],
                ['sensor_LO', 'Algorithm'],
                ['sensor_HO', 'Binning'],
                ['sensor_HO', 'Dispersion'],
                ['sensor_HO', 'ExcessNoiseFactor'],
                ['sensor_HO', 'FieldOfView'],
                ['sensor_HO', 'Modulation'],
                ['sensor_HO', 'NewValueThrPix'],
                ['sensor_HO', 'NoiseVariance'],
                ['sensor_HO', 'NumberLenslets'],
                ['sensor_HO', 'PixelScale'],
                ['sensor_HO', 'SigmaRON'],
                ['sensor_HO', 'SizeLenslets'],
                ['sensor_HO', 'ThresholdWCoG'],
                ['sensor_HO', 'WfsType'],
                ['sensor_LO', 'WfsType'],
                ['sensor_HO', 'WindowRadiusWCoG'],
                ['sensor_science', 'Binning'],
                ['sensor_science', 'ExcessNoiseFactor'],
                ['sensor_science', 'FieldOfView'],
                ['sensor_science', 'Name'],
                ['sensor_science', 'PixelScale'],
                ['sensor_science', 'Saturation'],
                ['sensor_science', 'SigmaRON'],
                ['telescope', 'ObscurationRatio'],
                ['telescope', 'PathApodizer'],
                ['telescope', 'PathPupil'],
                ['telescope', 'PathStatModes'],
                ['telescope', 'PathStaticOff'],
                ['telescope', 'PathStaticOn'],
                ['telescope', 'PathStaticPos'],
                # ['telescope', 'PupilAngle'],
                ['telescope', 'Resolution'],
                ['telescope', 'TelescopeDiameter']
            ]


    def set_value(self, root, items, value):
        """Set a value in a nested object in root by item sequence."""
        self.get_value(root, items[:-1])[items[-1]] = value


    def get_value(self, root, items):
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, items, root)


    def process_value(self, x):
        if x is None or isinstance(x, str) or isinstance(x, int):
            return x
        
        elif isinstance(x, (list, tuple)):
            if hasattr(x, '__len__') and len(x) == 1:
                return self.process_value(x[0])
            else:
                return x
            
        elif isinstance(x, (torch.Tensor, np.ndarray)):
            if x.ndim == 0:
                return float(x.real.item())
            else:
                x = x.squeeze()
                if x.ndim == 0:
                    return float(x.item())
                else:
                    return x.real
        else:
            return float(x.real)


    def process_dictionary(self, d):
        """Collapse all singleton dimensions of stored values to ensure that later addition of dimensions by TipTorch will go fine"""
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = self.process_dictionary(value)
            else:
                d[key] = self.process_value(value)
        return d


    def is_valid(self, value):
        """Checks if value contains Nones"""
        not_None = lambda x: False if x is None else True
        value = self.process_value(value)
        if hasattr(value, '__len__'):
            return np.all( np.array([self.is_valid(x) for x in value]) )
        else:
            return not_None(value)


    def Modify(self, config, modifier, conversion_table, processor_func=None):
        """
        Modifies a config file with an extrenal dictionary of values.
        Requires the mathing table to math values from extrenal source to config.
        Optionally, requires a processor function to tranform the values inside the config.
        """	
        self.process_dictionary(buf_config   := deepcopy(config))
        self.process_dictionary(buf_modifier := deepcopy(modifier))

        for config_entry, modifier_entry in conversion_table:
            self.set_value(buf_config, config_entry, self.get_value(buf_modifier, modifier_entry))
        
        if processor_func is not None:
            buf_config = processor_func(buf_config, buf_modifier)
        
        return buf_config


    def Merge(self, configs):
        """Merges config files for multiple targets into one config dictionary"""
        if not isinstance(configs, list):
            return configs
        new_config = deepcopy(configs[0]) # creating a buffer configuration

        def get_all_entries(d):
            entries = []
            def dict_path(path, my_dict):
                for k,v in my_dict.items():
                    if isinstance(v, dict):
                        dict_path(path+"|"+k, v)
                    else:
                        entries.append(path+"|"+str(k))
            dict_path('', d)
            for i,v in enumerate(entries): entries[i] = v.split('|')[1:]
            return entries

        all_entries = get_all_entries(new_config)
        for entry in all_entries:
            if entry in self.uniqualized_values:
                value_to_set = self.get_value(configs[0], entry)
            else:
                value_to_set = [self.get_value(config, entry) for config in configs]
            self.set_value(new_config, entry, value_to_set)

        new_config['NumberSources'] = int(len(configs))
        return new_config


    def Split(self, config, indexes):
        """Splits a config file based on a list of indexes into separate config parts."""
        # Deep copy the original config to avoid modifying it
        original_config = deepcopy(config)

        # Function to get all paths (entries) in the config dictionary
        def get_all_entries(d):
            entries = []
            def dict_path(path, my_dict):
                for k, v in my_dict.items():
                    new_path = f"{path}|{k}" if path else k
                    if isinstance(v, dict):
                        dict_path(new_path, v)
                    else:
                        entries.append(new_path)
            dict_path('', d)
            return entries

        all_entries = get_all_entries(original_config)

        # Function to create a blank config structure
        def create_blank_config_from_original(original):
            if isinstance(original, dict):
                return {k: create_blank_config_from_original(v) for k, v in original.items()}
            else:
                return None  # Placeholder for non-dict values

        # Initialize split configs based on indexes
        split_configs = [create_blank_config_from_original(original_config) for _ in indexes]

        # Function to set values in split configs based on indexes
        def set_values_based_on_indexes(split_configs, original_config, all_entries, indexes):
            for entry in all_entries:
                path = entry.split('|')
                value = reduce(operator.getitem, path, original_config)

                # Decide how to split the value based on its type and whether it's uniqualized
                if path in self.uniqualized_values or not isinstance(value, (list, tuple)):
                    # Set the same value for all configs if it's uniqualized or not a list/tuple
                    for split_config in split_configs:
                        self.set_value(split_config, path, deepcopy(value))
                else:
                    # For list/tuple values, distribute them according to indexes
                    for i, index in enumerate(indexes):
                        if 0 <= index < len(value):
                            self.set_value(split_configs[i], path, deepcopy(value[index]))
                        else:
                            # Handle the case where the index is out of range
                            self.set_value(split_configs[i], path, None)

        set_values_based_on_indexes(split_configs, original_config, all_entries, indexes)

        return split_configs


    def Convert(self, config, framework='pytorch', device=None):
        """Converts all values in a config file to a specified framework"""
        if framework.lower() == 'pytorch' or framework.lower() == 'torch':
            
            if device is None:
                device = torch.device('cpu')
            
            def convert_value(x):
                if isinstance(x, torch.Tensor):
                    return x.float().to(device) if x.device != device else x.float()
                
                elif isinstance(x, list) and all(isinstance(i, torch.Tensor) for i in x):
                    return torch.stack(x).float().to(device)
                
                elif isinstance(x, str):
                    pass
                
                else:
                    return torch.as_tensor(x, dtype=torch.float32, device=device)
                            
            
        elif framework.lower() == 'numpy':
            convert_value = lambda x: np.array(x.cpu()) if isinstance(x, torch.Tensor) else np.array(x)
        
        elif framework.lower() == 'cupy':
            convert_value = lambda x: cp.array(x.cpu()) if isinstance(x, torch.Tensor) else cp.array(x)
        
        elif framework.lower() == 'list':
            def convert_value(x):
                if isinstance(x, torch.Tensor):
                    return x.cpu().tolist()
                elif isinstance(x, np.ndarray):
                    return x.tolist()
                elif isinstance(x, cp.ndarray):
                    return cp.asnumpy(x).tolist()
                else:
                    return x
        else:
            raise NotImplementedError(f'Unsupported framework "{framework}"!')

        zero_d = lambda x: x if type(x) == float else convert_value(x)
        
        for entry in config:
            value = config[entry]
            if isinstance(value, dict):
                self.Convert(value, framework, device)  
            elif isinstance(value, str) or not self.is_valid(value):
                pass
            else:
                value = zero_d(value)
                
            config[entry] = value


'''
    def Convert(self, config, framework='pytorch', device=None, convert_to_float=True):
        """Converts all values in a config file to a specified framework"""
        if framework.lower() == 'pytorch' or framework.lower() == 'torch':
            if device is None: device = torch.device('cpu')

            if getattr(torch.backends.mps, "is_available", lambda: False)():
                convert_to_float = True  # MPS only supports float32

            # Define a helper function to convert values to arraays of the correct type and framework
            def convert_value(x):
                if isinstance(x, torch.Tensor):
                    if convert_to_float:
                        return x.to(device=device, dtype=torch.float32)
                    else:
                        return x.to(device=device)
                    
                elif isinstance(x, list) and all(isinstance(i, torch.Tensor) for i in x):
                    if convert_to_float:
                        return torch.stack(x).to(device=device, dtype=torch.float32)
                    else:
                        return torch.stack(x).to(device=device)
                    
                elif isinstance(x, str):
                    return x
                
                else:
                    if convert_to_float:
                        return torch.as_tensor(x, device=device, dtype=torch.float32)
                    else:
                        return torch.as_tensor(x, device=device)

        elif framework.lower() == 'numpy':
            def convert_value(x):
                if isinstance(x, torch.Tensor):
                    arr = x.cpu().numpy()
                    return arr.astype(np.float32) if convert_to_float else arr
                else:
                    arr = np.array(x)
                    return arr.astype(np.float32) if convert_to_float else arr

        elif framework.lower() == 'cupy':
            def convert_value(x):
                if isinstance(x, torch.Tensor):
                    arr = cp.array(x.cpu().numpy())
                    return arr.astype(cp.float32) if convert_to_float else arr
                else:
                    arr = cp.array(x)
                    return arr.astype(cp.float32) if convert_to_float else arr

        elif framework.lower() == 'list':
            def convert_value(x):
                if   isinstance(x, torch.Tensor): return x.cpu().tolist()
                elif isinstance(x, np.ndarray):   return x.tolist()
                elif isinstance(x, cp.ndarray):   return cp.asnumpy(x).tolist()+
                else: return x
        else:
            raise NotImplementedError(f'Unsupported framework "{framework}"!')

        zero_d = lambda x: x if isinstance(x, float) else convert_value(x)

        for entry in config:
            value = config[entry]
            if isinstance(value, dict):
                self.Convert(value, framework, device, convert_to_float) # use recursion to convert nested dictionaries
                
            elif isinstance(value, str) or not self.is_valid(value):
                pass
            
            else:
                config[entry] = zero_d(value)
'''


def GetSPHEREonsky():
    '''
    This function composes the SPHERE config understandable by the TipTorch model. It converts the data from an external source
    (modifier) and puts iit into the config. NOTE: to be refactored for clarity!
    '''
    
    conversion_table = [
        (['atmosphere','Seeing'],          ['seeing','SPARTA']        ),
        (['atmosphere','WindSpeed'],       ['Wind speed','header']    ),
        (['atmosphere','WindDirection'],   ['Wind direction','header']),
        (['sensor_science','Zenith'],      ['telescope','altitude']   ),
        (['telescope','ZenithAngle'],      ['telescope','altitude']   ), #TODO: difference between zenith and zenithAngle?
        (['sensor_science','Azimuth'],     ['telescope','azimuth']    ),
        (['sensor_science','SigmaRON'],    ['Detector','ron']         ),
        (['sensor_science','Gain'],        ['Detector','gain']        ),
        (['sources_HO', 'Wavelength'],     ['WFS', 'wavelength']      ),
        (['sensor_HO','NumberPhotons'],    ['WFS','Nph vis']          ),
        (['sensor_HO','Jitter X'],         ['WFS','TT jitter X']      ),
        (['sensor_HO','Jitter Y'],         ['WFS','TT jitter Y']      ),
        (['RTC','SensorFrameRate_HO'],     ['WFS','rate']             ),
        (['sensor_science','PixelScale'],  ['Detector', 'psInMas']    ),
        (['sensor_science','SigmaRON'],    ['Detector', 'ron']        ),
        (['sources_science','Wavelength'], ['spectra']                )
    ]
    
    def processor_func(config, modifier):
        config['sources_science']['Zenith'] = 90.0 - modifier['telescope']['altitude']
        config['telescope']['ZenithAngle']  = 90.0 - modifier['telescope']['altitude']
        
        def frame_delay(loop_freq):
            if not isinstance(loop_freq, torch.Tensor):
                loop_freq_ = torch.tensor(loop_freq)
            return torch.clamp(loop_freq_/1e3 * 2.3, min=1.0)
        
        config['RTC']['LoopDelaySteps_HO'] = frame_delay(config['RTC']['SensorFrameRate_HO'])

        return config
    
    return conversion_table, processor_func


def GetSPHEREsynth():
    conversion_table = [
        (['atmosphere','Cn2Weights'],      ['Cn2','profile']     ),
        (['atmosphere','Cn2Heights'],      ['Cn2','heights']      ),
        (['atmosphere','Seeing'],          ['seeing']             ),
        (['atmosphere','WindSpeed'],       ['Wind speed']         ),
        (['atmosphere','WindDirection'],   ['Wind direction']     ),
        (['telescope','Zenith'],           ['telescope','zenith'] ),
        (['telescope','ZenithAngle'],      ['telescope','zenith'] ),
        (['sensor_science','SigmaRON'],    ['Detector','ron']     ),
        (['sensor_science','Gain'],        ['Detector','gain']    ),
        (['sensor_HO','NumberPhotons'],    ['WFS','Nph vis']      ),
        (['sources_HO','Wavelength'],      ['WFS','wavelength']   ),
        (['RTC','SensorFrameRate_HO'],     ['RTC','loop rate']    ),
        (['RTC','LoopGain_HO'],            ['RTC','loop gain']    ),
        (['RTC','LoopDelaySteps_HO'],      ['RTC','frames delay'] ),
        (['sensor_science','PixelScale'],  ['Detector', 'psInMas']),
        (['sensor_science','SigmaRON'],    ['Detector','ron']     ),
        (['sources_science','Wavelength'], ['spectra']            ),
    ]
    return conversion_table, None

    
def ConfigFromFile(ini_file_path, device):
    config_manager = ConfigManager()
    config = ParameterParser(ini_file_path).params
    config_manager.Convert(config, framework='pytorch', device=device)

    config['NumberSources'] = 1

    config['telescope']['ZenithAngle'] = torch.tensor([config['telescope']['ZenithAngle']], device=device)

    config['atmosphere']['Cn2Weights'] = config['atmosphere']['Cn2Weights'].unsqueeze(0)
    config['atmosphere']['Cn2Heights'] = config['atmosphere']['Cn2Heights'].unsqueeze(0)

    config['RTC']['SensorFrameRate_HO'] = torch.tensor(config['RTC']['SensorFrameRate_HO'], device=device)
    config['RTC']['LoopDelaySteps_HO'] = torch.tensor(config['RTC']['LoopDelaySteps_HO'], device=device)
    config['RTC']['LoopGain_HO'] = torch.tensor(config['RTC']['LoopGain_HO'], device=device)

    config['atmosphere']['L0'] = torch.tensor(config['atmosphere']['L0'], device=device)

    config['sensor_science']['FieldOfView'] = config['sensor_science']['FieldOfView'].int().item()

    config['sensor_HO']['SizeLenslets'] = config['sensor_HO']['SizeLenslets'] [0]
    config['sensor_HO']['ClockRate']    = config['sensor_HO']['ClockRate'][0]

    return config


def are_equal(a, b, tolerance):
    """Check if two values are equal within the given tolerance, handling tensors and nested structures."""
    # Handle dictionaries
    if isinstance(a, dict) and isinstance(b, dict):
        return len(CompareConfigs(a, b, tolerance)) == 0
    # Handle lists and tuples
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not are_equal(x, y, tolerance):
                return False
        return True
    # Handle tensors and numbers
    elif torch.is_tensor(a) or torch.is_tensor(b):
        a_tensor = torch.is_tensor(a)
        b_tensor = torch.is_tensor(b)
        if a_tensor and b_tensor:
            a_cpu = a.cpu()
            b_cpu = b.cpu()
            return torch.allclose(a_cpu, b_cpu, atol=tolerance, rtol=tolerance)
        elif a_tensor:
            if a.numel() != 1:
                return False
            a_val = a.item()
            return isinstance(b, (int, float)) and math.isclose(a_val, b, abs_tol=tolerance)
        elif b_tensor:
            if b.numel() != 1:
                return False
            b_val = b.item()
            return isinstance(a, (int, float)) and math.isclose(a, b_val, abs_tol=tolerance)
        else:
            return False
    # Handle numbers (int/float)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, abs_tol=tolerance)
    # Handle other types (strings, None, etc.)
    else:
        return a == b


def CompareConfigs(dict1, dict2, tolerance=1e-6, path=""):
    """Recursively compare two dictionaries and return a list of differences."""
    differences = []
    # Check for missing keys
    for key in dict1:
        if key not in dict2:
            differences.append(f"Key {path}.{key} not in dict2")
    for key in dict2:
        if key not in dict1:
            differences.append(f"Key {path}.{key} not in dict1")
    # Compare common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())
    for key in common_keys:
        current_path = f"{path}.{key}" if path else key
        val1 = dict1[key]
        val2 = dict2[key]
        # Check if both are dictionaries
        if isinstance(val1, dict) and isinstance(val2, dict):
            sub_diffs = CompareConfigs(val1, val2, tolerance, current_path)
            differences.extend(sub_diffs)
        else:
            # Check for lists or tuples
            if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                if len(val1) != len(val2):
                    differences.append(f"Mismatch in {current_path}: lengths {len(val1)} vs {len(val2)}")
                else:
                    for i, (x, y) in enumerate(zip(val1, val2)):
                        elem_path = f"{current_path}[{i}]"
                        if isinstance(x, dict) and isinstance(y, dict):
                            sub_diffs = CompareConfigs(x, y, tolerance, elem_path)
                            differences.extend(sub_diffs)
                        else:
                            if not are_equal(x, y, tolerance):
                                differences.append(f"Mismatch in {elem_path}: {x} vs {y}")
            else:
                if not are_equal(val1, val2, tolerance):
                    differences.append(f"Mismatch in {current_path}: {val1} vs {val2}")
    return differences


def MultipleTargetsInOneObservation(config_file, N_srcs):
    '''
    Initializing the config file this way allows to avoid computing tomographic reconstructor
    multiple times for the same atmospheric conditions. This involves initializing the proper dimensions
    for the input arrays. Given the right dimensionality, the PSF model will automatically understand what to do.
    '''

    config_manager = ConfigManager()
    config  = config_manager.Merge([config_file,]*N_srcs)

    if hasattr(config_file['atmosphere']['Cn2Weights'], 'device'):
        config_manager.Convert(config, framework='pytorch', device=config_file['atmosphere']['Cn2Weights'].device)
    else:
        config_manager.Convert(config, framework='numpy')
    
    config['NumberSources'] = N_srcs
    
    config['sources_science']['Wavelength'] = config['sources_science']['Wavelength'][0]
    config['sources_HO']['Height']          = config['sources_HO']['Height'].unsqueeze(-1)
    config['sources_HO']['Wavelength']      = config['sources_HO']['Wavelength'].squeeze()
    config['sensor_science']['FieldOfView'] = config['sensor_science']['FieldOfView'].int().item()
    config['atmosphere']['Cn2Weights']      = config['atmosphere']['Cn2Weights'].squeeze()
    config['atmosphere']['Cn2Heights']      = config['atmosphere']['Cn2Heights'].squeeze()
    config['sensor_HO']['NumberPhotons']    = config['sensor_HO']['NumberPhotons'].squeeze()
    
    config['telescope']['ZenithAngle']    = config['telescope']['ZenithAngle'][0,...]

    config['atmosphere']['L0']            = config['atmosphere']['L0'][0]
    config['atmosphere']['Seeing']        = config['atmosphere']['Seeing'][0]
    config['atmosphere']['Cn2Weights']    = config['atmosphere']['Cn2Weights'][0,...].unsqueeze(0)
    config['atmosphere']['Cn2Heights']    = config['atmosphere']['Cn2Heights'][0,...].unsqueeze(0)
    config['atmosphere']['WindSpeed']     = config['atmosphere']['WindSpeed'][0,...].unsqueeze(0)
    config['atmosphere']['WindDirection'] = config['atmosphere']['WindDirection'][0,...].unsqueeze(0)

    config['sources_HO']['Height']        = config['sources_HO']['Height'][0,...].unsqueeze(0)
    config['sources_HO']['Zenith']        = config['sources_HO']['Zenith'][0,...]
    config['sources_HO']['Azimuth']       = config['sources_HO']['Azimuth'][0,...]

    config['RTC']['SensorFrameRate_HO']   = config['RTC']['SensorFrameRate_HO'][0]
    config['RTC']['LoopDelaySteps_HO']    = config['RTC']['LoopDelaySteps_HO'][0]
    config['RTC']['LoopGain_HO']          = config['RTC']['LoopGain_HO'][0]

    config['DM']['OptimizationZenith' ]   = config['DM']['OptimizationZenith'][0]
    config['DM']['OptimizationAzimuth']   = config['DM']['OptimizationAzimuth'][0]
    config['DM']['OptimizationWeight']    = config['DM']['OptimizationWeight'][0]
    
    config['sensor_HO']['NumberPhotons']  = config['sensor_HO']['NumberPhotons'][0,...].unsqueeze(0)
    config['sensor_HO']['ClockRate']      = config['sensor_HO']['ClockRate'][0]
    
    return config
