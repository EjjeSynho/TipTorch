import sys
sys.path.insert(0, '..')

from managers.parameter_parser import ParameterParser, default_torch_type

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


    def Convert(self, config, framework='pytorch', device=torch.device('cpu'), dtype=default_torch_type):
        """Converts all values in a config file to a specified framework"""
        
        if framework.lower() == 'pytorch' or framework.lower() == 'torch':
            def convert_value(x):
                if isinstance(x, torch.Tensor):
                    return x.to(device=device, dtype=dtype)
                
                elif isinstance(x, list) and all(isinstance(i, torch.Tensor) for i in x):
                    return torch.stack([t.to(device=device, dtype=dtype) for t in x])

                elif isinstance(x, list) and all(isinstance(i, np.ndarray) for i in x):
                    return torch.stack([torch.from_numpy(i) for i in x]).to(device, dtype=dtype)

                else:
                    return torch.as_tensor(x, dtype=dtype, device=device)

        elif framework.lower() == 'numpy':
            def convert_value(x): 
                if isinstance(x, torch.Tensor):
                    return np.array(x.cpu())
                else:
                    return np.array(x)

        elif framework.lower() == 'cupy':
            def convert_value(x):
                if isinstance(x, torch.Tensor):
                    return cp.array(x.cpu())
                else:
                    return cp.array(x)

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

        # Handle scalar types
        def process_value(x):
            if isinstance(x, (str, type(None))) or not self.is_valid(x):
                return x
            elif isinstance(x, (int, float, complex)) and not hasattr(x, '__len__'):
                return x  # Keep scalar values as-is
            else:
                return convert_value(x)
        
        for entry in config:
            value = config[entry]
            if isinstance(value, dict):
                self.Convert(value, framework, device)  
            else:
                config[entry] = process_value(value)  # Assign the converted value back

    
'''
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
'''

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


def MultipleTargetsInOneObservation(config_file, N_srcs, device=None):
    '''
    Restructures config file in such a way that multiple sources are observed in one observation.
    Initializing the config file this way allows to avoid computing tomographic reconstructor
    multiple times for the same atmospheric conditions. This involves initializing the proper dimensions
    for the input arrays. Given the right dimensionality, the PSF model will automatically understand what to do.
    '''

    config_manager = ConfigManager()
    config  = config_manager.Merge([config_file,]*N_srcs)

    # Determine framework based on device parameter or existing data
    if device is not None:
        config_manager.Convert(config, framework='pytorch', device=device)
    else:
        config_manager.Convert(config, framework='numpy')
    
    config['NumberSources'] = N_srcs

    if device is not None:
        # PyTorch operations
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
        
    else:
        # NumPy operations
        config['sources_science']['Wavelength'] = config['sources_science']['Wavelength'][0]
        config['sources_HO']['Height']          = np.expand_dims(config['sources_HO']['Height'], axis=-1)
        config['sources_HO']['Wavelength']      = config['sources_HO']['Wavelength'].squeeze()
        config['sensor_science']['FieldOfView'] = int(config['sensor_science']['FieldOfView'])
        config['atmosphere']['Cn2Weights']      = config['atmosphere']['Cn2Weights'].squeeze()
        config['atmosphere']['Cn2Heights']      = config['atmosphere']['Cn2Heights'].squeeze()
        config['sensor_HO']['NumberPhotons']    = config['sensor_HO']['NumberPhotons'].squeeze()
        
        config['telescope']['ZenithAngle']    = config['telescope']['ZenithAngle'][0,...]
        
        config['atmosphere']['L0']            = config['atmosphere']['L0'][0]
        config['atmosphere']['Seeing']        = config['atmosphere']['Seeing'][0]
        config['atmosphere']['Cn2Weights']    = np.expand_dims(config['atmosphere']['Cn2Weights'][0,...], axis=0)
        config['atmosphere']['Cn2Heights']    = np.expand_dims(config['atmosphere']['Cn2Heights'][0,...], axis=0)
        config['atmosphere']['WindSpeed']     = np.expand_dims(config['atmosphere']['WindSpeed'][0,...], axis=0)
        config['atmosphere']['WindDirection'] = np.expand_dims(config['atmosphere']['WindDirection'][0,...], axis=0)

        config['sources_HO']['Height']        = np.expand_dims(config['sources_HO']['Height'][0,...], axis=0)
        config['sources_HO']['Zenith']        = config['sources_HO']['Zenith'][0,...]
        config['sources_HO']['Azimuth']       = config['sources_HO']['Azimuth'][0,...]

        config['RTC']['SensorFrameRate_HO']   = config['RTC']['SensorFrameRate_HO'][0]
        config['RTC']['LoopDelaySteps_HO']    = config['RTC']['LoopDelaySteps_HO'][0]
        config['RTC']['LoopGain_HO']          = config['RTC']['LoopGain_HO'][0]

        config['DM']['OptimizationZenith' ]   = config['DM']['OptimizationZenith'][0]
        config['DM']['OptimizationAzimuth']   = config['DM']['OptimizationAzimuth'][0]
        config['DM']['OptimizationWeight']    = config['DM']['OptimizationWeight'][0]
        
        config['sensor_HO']['NumberPhotons']  = np.expand_dims(config['sensor_HO']['NumberPhotons'][0,...], axis=0)
        config['sensor_HO']['ClockRate']      = config['sensor_HO']['ClockRate'][0]
    
    return config


def MultipleTargetsInDifferentObservations(configs, device=None):
    '''
    Merges multiple config files into one config file for multiple targets observed in different observations.
    Useful when training calibrator NN on multiple targets.
    '''
    config_manager = ConfigManager()
    merged_config  = config_manager.Merge(configs)

    N_src = len(configs)
    
    if device is None:
        config_manager.Convert(merged_config, framework='numpy')
        
        # All stacked sources must have the same wavelengths bins
        merged_config['sources_HO']['Height']        = np.expand_dims(merged_config['sources_HO']['Height'], axis=-1)
        merged_config['atmosphere']['Cn2Weights']    = merged_config['atmosphere']['Cn2Weights'].reshape(N_src, -1)
        merged_config['atmosphere']['Cn2Heights']    = merged_config['atmosphere']['Cn2Heights'].reshape(N_src, -1)
        merged_config['atmosphere']['WindSpeed']     = merged_config['atmosphere']['WindSpeed'].reshape(N_src, -1)
        merged_config['atmosphere']['WindDirection'] = merged_config['atmosphere']['WindDirection'].reshape(N_src, -1)
        merged_config['atmosphere']['Seeing']        = merged_config['atmosphere']['Seeing'].reshape(N_src)
        merged_config['sensor_HO']['NumberPhotons']  = merged_config['sensor_HO']['NumberPhotons'].reshape(N_src, -1)
    else:
        config_manager.Convert(merged_config, framework='pytorch', device=device)
        
        merged_config['sources_HO']['Height']        = merged_config['sources_HO']['Height'].unsqueeze(-1)
        merged_config['atmosphere']['Cn2Weights']    = merged_config['atmosphere']['Cn2Weights'].view(N_src, -1)
        merged_config['atmosphere']['Cn2Heights']    = merged_config['atmosphere']['Cn2Heights'].view(N_src, -1)
        merged_config['atmosphere']['WindSpeed']     = merged_config['atmosphere']['WindSpeed'].view(N_src, -1)
        merged_config['atmosphere']['WindDirection'] = merged_config['atmosphere']['WindDirection'].view(N_src, -1)
        merged_config['atmosphere']['Seeing']        = merged_config['atmosphere']['Seeing'].view(N_src)
        merged_config['sensor_HO']['NumberPhotons']  = merged_config['sensor_HO']['NumberPhotons'].view(N_src, -1)

    merged_config['NumberSources'] = N_src
    # All stacked sources must have the same wavelengths bins
    merged_config['sources_science']['Wavelength'] = merged_config['sources_science']['Wavelength'].view(N_src, -1)[0,...].unsqueeze(0)
    merged_config['sources_HO']['Wavelength']      = merged_config['sources_HO']['Wavelength'].squeeze()
    merged_config['sensor_science']['FieldOfView'] = int(merged_config['sensor_science']['FieldOfView'])

    return merged_config