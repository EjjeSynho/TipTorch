import sys
sys.path.insert(0, '..')

import math
import torch
import numpy as np
from pathlib import Path
import yaml

try:
    import cupy as cp
except ImportError:
    cp = np
    
from copy import deepcopy
from functools import reduce
import operator

from managers.parameter_parser import ParameterParser, default_torch_type

"""
This module is used to manage the configuration files of TipTorch. Configuration files are used to set the up
the initial state of the TipTorch simulation. This module is used to parse the configuration files and manipulate
the configs, i.e., splitting and merging them, converting to different frameworks (NumPy, PyTorch, CuPy), etc.
Since TipTorch configs support for multiple targets, splitting and joining different configs is necessary.
"""

# These values remain the same for the entire config, regardless of the number of targets
SINGLETON_VALUES = [
    ['atmosphere', 'Wavelength'],
    ['DM', 'AoArea'],
    ['DM', 'DmHeights'],
    ['DM', 'DmPitchs'],
    ['DM', 'InfCoupling'],
    ['DM', 'InfModel'],
    ['DM', 'NumberActuators'],
    ['DM', 'OptimizationConditioning'],
    # ['DM', 'OptimizationAzimuth'],
    # ['DM', 'OptimizationWeight'],
    ['sensor_HO', 'Algorithm'],
    ['sensor_LO', 'Algorithm'],
    ['sensor_HO', 'Binning'],
    ['sensor_HO', 'Dispersion'],
    # ['sensor_HO', 'ExcessNoiseFactor'],
    ['sensor_HO', 'FieldOfView'],
    ['sensor_HO', 'Modulation'],
    ['sensor_HO', 'NewValueThrPix'],
    ['sensor_HO', 'NoiseVariance'],
    ['sensor_HO', 'NumberLenslets'],
    ['sensor_HO', 'PixelScale'],
    # ['sensor_HO', 'SigmaRON'],
    ['sensor_HO', 'SizeLenslets'],
    ['sensor_HO', 'ThresholdWCoG'],
    ['sensor_HO', 'WfsType'],
    ['sensor_LO', 'WfsType'],
    ['sensor_HO', 'WindowRadiusWCoG'],
    ['sensor_science', 'Binning'],
    # ['sensor_science', 'ExcessNoiseFactor'],
    ['sources_science', 'Wavelength'],
    ['sensor_science', 'FieldOfView'],
    ['sensor_science', 'Name'],
    ['sensor_science', 'PixelScale'],
    ['sensor_science', 'Saturation'],
    # ['sensor_science', 'SigmaRON'],
    ['telescope', 'ObscurationRatio'],
    ['telescope', 'PathApodizer'],
    ['telescope', 'PathPupil'],
    ['telescope', 'PupilAngle'],
    ['telescope', 'PathStatModes'],
    ['telescope', 'PathStaticOff'],
    ['telescope', 'PathStaticOn'],
    ['telescope', 'PathStaticPos'],
    ['telescope', 'Resolution'],
    ['telescope', 'TelescopeDiameter']
]

# Thee values remain scalars and must not be converted to arrays/lists when modifying configs
SCALAR_VALUES = [
    ['sensor_HO', 'FieldOfView'],
    ['sensor_HO', 'PixelScale'],
    ['sensor_science', 'PixelScale'],
    ['sensor_science', 'FieldOfView'],
    ['telescope', 'PupilAngle'],
    ['telescope', 'TelescopeDiameter']
]

class ConfigManager():
    def __init__(self):
        # Load required fields specification
        required_fields_path = Path(__file__).parent.parent / 'data' / 'parameter_files' / 'required_fields.yaml'
        
        with open(required_fields_path, 'r') as f:
            self.required_fields = yaml.safe_load(f)

    def Load(self, path_ini):
        """ Loads and preprocesses a config file from the given path. """
        if not isinstance(path_ini, str): path_ini = str(path_ini)
        
        config_manager = ConfigManager()
        config = ParameterParser(path_ini).params

        config['NumberSources'] = 1 # Assumes by default that it's a single-source config
        config = config_manager.select_required_fields(config)
        config = config_manager.wrap_scalars_to_lists(config)
        config = config_manager.ensure_dimensions(config, 1)
        return config
    

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
            if entry in SINGLETON_VALUES:
                value_to_set = self.get_value(configs[0], entry)
            else:
                value_to_set = [self.get_value(config, entry) for config in configs]
            self.set_value(new_config, entry, value_to_set)

        N_src = 0
        for config in configs:
            N_src += config['NumberSources']
            
        new_config['NumberSources'] = N_src
        new_config = self.ensure_dimensions(new_config, N_src)
        
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

                # Decide how to split the value based on its type and whether it's a singleton
                if path in SINGLETON_VALUES or not isinstance(value, (list, tuple)):
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
                    return x.to(device=device, dtype=dtype, non_blocking=True)
                
                elif isinstance(x, list) and all(isinstance(i, torch.Tensor) for i in x):
                    return torch.stack([t.to(device=device, dtype=dtype, non_blocking=True) for t in x])

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
                
        return config


    def CompareConfigs(self, dict1, dict2, tolerance=1e-6, path=""):
        """
        Recursively compare two config dictionaries and return a list of differences.
        Handles nested structures, arrays, tensors, and numeric comparisons with tolerance.
        
        Parameters:
        -----------
        dict1 : dict
            First configuration dictionary
        dict2 : dict
            Second configuration dictionary
        tolerance : float
            Absolute tolerance for numeric comparisons (default: 1e-6)
        path : str
            Current path in nested structure (for error reporting)
            
        Returns:
        --------
        list
            List of difference descriptions. Empty list means configs are equal.
        """
        def _are_equal(a, b):
            """Check if two values are equal within tolerance."""
            # Handle dictionaries
            if isinstance(a, dict) and isinstance(b, dict):
                return len(self.CompareConfigs(a, b, tolerance)) == 0
            
            # Handle lists and tuples
            elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                if len(a) != len(b):
                    return False
                return all(_are_equal(x, y) for x, y in zip(a, b))
            
            # Handle tensors and tensor-scalar comparisons
            elif torch.is_tensor(a) or torch.is_tensor(b):
                a_tensor = torch.is_tensor(a)
                b_tensor = torch.is_tensor(b)
                
                # Both tensors: compare on CPU with tolerance
                if a_tensor and b_tensor:
                    return torch.allclose(a.cpu(), b.cpu(), atol=tolerance, rtol=tolerance)        
                # Tensor vs scalar: check if tensor is single element
                elif a_tensor and isinstance(b, (int, float)):
                    return a.numel() == 1 and math.isclose(a.item(), b, abs_tol=tolerance)
                # Scalar vs tensor: check if tensor is single element
                elif b_tensor and isinstance(a, (int, float)):
                    return b.numel() == 1 and math.isclose(a, b.item(), abs_tol=tolerance)
                else:
                    return False
            
            # Handle NumPy arrays
            elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                try:
                    return np.allclose(a, b, atol=tolerance, rtol=tolerance)
                except (TypeError, ValueError):
                    return False
            
            # Handle numeric types
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return math.isclose(a, b, abs_tol=tolerance)
            
            # Handle strings, None, and other types
            else:
                return a == b
        
        differences = []
        
        # Check for missing keys
        for key in dict1:
            if key not in dict2:
                differences.append(f"Key '{path}.{key}' exists in dict1 but not in dict2" if path else f"Key '{key}' exists in dict1 but not in dict2")
        
        for key in dict2:
            if key not in dict1:
                differences.append(f"Key '{path}.{key}' exists in dict2 but not in dict1" if path else f"Key '{key}' exists in dict2 but not in dict1")
        
        # Compare common keys
        common_keys = set(dict1.keys()) & set(dict2.keys())
        for key in common_keys:
            current_path = f"{path}.{key}" if path else key
            val1 = dict1[key]
            val2 = dict2[key]
            
            # Recursively compare nested dictionaries
            if isinstance(val1, dict) and isinstance(val2, dict):
                sub_diffs = self.CompareConfigs(val1, val2, tolerance, current_path)
                differences.extend(sub_diffs)
            
            # Compare list/tuple elements
            elif isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                if len(val1) != len(val2):
                    differences.append(f"Length mismatch at '{current_path}': {len(val1)} vs {len(val2)}")
                else:
                    for i, (x, y) in enumerate(zip(val1, val2)):
                        elem_path = f"{current_path}[{i}]"
                        if isinstance(x, dict) and isinstance(y, dict):
                            sub_diffs = self.CompareConfigs(x, y, tolerance, elem_path)
                            differences.extend(sub_diffs)
                        elif not _are_equal(x, y):
                            differences.append(f"Value mismatch at '{elem_path}': {x} vs {y}")
            
            # Compare scalar values
            else:
                if not _are_equal(val1, val2):
                    differences.append(f"Value mismatch at '{current_path}': {val1} vs {val2}")
        
        return differences


    def select_required_fields(self, config):
        """
        Filters a config dictionary to keep only the required fields specified in a YAML file.
        Modifies the dictionary in place.
        """
        # Remove unwanted top-level keys (except NumberSources)
        for key in list(config.keys()):
            if key not in self.required_fields and key != 'NumberSources':
                config.pop(key)

        # Filter fields within each allowed category
        for category, allowed_fields in self.required_fields.items():
            if category not in config:
                continue

            for field in list(config[category].keys()):
                if field not in allowed_fields:
                    config[category].pop(field)

        return config


    def wrap_scalars_to_lists(self, config):
        """
        Walks through a config dictionary and wraps all scalar numeric values (int, float)
        into single-element lists, except for SCALAR_VALUES which remain as scalars.
        Modifies the dict in place.
        """
        # Convert SCALAR_VALUES to set of tuples for efficient lookup
        exception_set = {tuple(path) for path in SCALAR_VALUES}
        
        def walk(node, path):
            if not isinstance(node, dict):
                return

            for key, value in node.items():
                new_path = (*path, key)

                if isinstance(value, dict):
                    walk(value, new_path)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    if new_path not in exception_set:
                        node[key] = [value]

        # Start the recursive walk, skipping 'NumberSources' at top level
        for key, value in config.items():
            if key == 'NumberSources':
                continue
            if isinstance(value, dict):
                walk(value, (key,))

        return config


    def ensure_dims(self, x, N, flatten=False):
        """ Ensures a selected tensor has correct dimensions. """
        if type(x) is np.ndarray:  # NumPy array
            x = np.atleast_1d(x).squeeze()
            result = np.atleast_2d(x).reshape(N, -1)
            return result.flatten() if flatten else result
        
        elif hasattr(x, 'is_cuda'):  # PyTorch tensor
            x = torch.atleast_1d(x).squeeze()
            result = torch.atleast_2d(x).view(N, -1)
            return result.flatten() if flatten else result
        
        elif hasattr(x, 'get'):  # CuPy array
            x = cp.atleast_1d(x).squeeze()
            result = cp.atleast_2d(x).reshape(N, -1)
            return result.flatten() if flatten else result
        
        else:  # Fallback to NumPy
            x = np.atleast_1d(np.array(x)).squeeze()
            result = np.atleast_2d(x).reshape(N, -1)
            return result.flatten() if flatten else result


    def ensure_dimensions(self, config, N_src):
        """ Ensures that all relevant config entries have correct dimensions. """
        
        # NOTE: requires that config entries are already converted to the target framework (NumPy, PyTorch, CuPy).
        # Check only one known field to accelerate the process
        sample_value = config['atmosphere']['Seeing']
        
        if not isinstance(sample_value, (np.ndarray, torch.Tensor, cp.ndarray)):
            config = self.Convert(config, framework='numpy')
        
        entries_2d = [\
            ['atmosphere', 'Cn2Weights'],
            ['atmosphere', 'Cn2Heights'],
            ['atmosphere', 'WindSpeed'],
            ['atmosphere', 'WindDirection'],
            ['sensor_HO', 'NumberPhotons'],
            ['sensor_HO', 'SpotFWHM'],
            ['telescope', 'ZenithAngle'],
            ['sources_HO', 'Height'],
            ['sources_HO', 'Zenith'],
            ['sources_HO', 'Azimuth']
        ]

        entries_flatten = [\
            ['atmosphere', 'Seeing'],
            ['atmosphere', 'L0'],
            ['sources_science', 'Zenith'],
            ['sources_science', 'Azimuth']
        ]
        
        entries_singleton = [\
            ['sources_HO', 'Wavelength'],
            ['sensor_HO', 'ClockRate'],
            ['sensor_HO', 'NumberLenslets'],
            ['sensor_HO', 'SizeLenslets']
        ]

        for section, entry in entries_2d:
            config[section][entry] = self.ensure_dims(config[section][entry], N_src, flatten=False)

        for section, entry in entries_flatten:
            config[section][entry] = self.ensure_dims(config[section][entry], N_src, flatten=True)

        # Doing this assumes that these parameters are the same for all WFSs
        for section, entry in entries_singleton:
            config[section][entry] = self.ensure_dims(config[section][entry], 1, flatten=False)
            config[section][entry] = config[section][entry][:,0]

        return config


    def debug_dims(self, config):
        # Print all shapes for verification
        for section, subdict in config.items():
            if isinstance(subdict, dict):
                for entry, value in subdict.items():
                    if isinstance(value, (np.ndarray, torch.Tensor, cp.ndarray)):
                        print(f"{section} -> {entry}: {value.shape}")
                    else:
                        print(f"{section} -> {entry}: {type(value)}")
                        

def MultipleTargetsInOneObservation(config_file, N_srcs, device=None):
    '''
    Restructures config file in such a way that multiple sources are observed in one observation.
    Initializing the config file this way allows to avoid computing tomographic reconstructor
    multiple times for the same atmospheric conditions. This involves initializing the proper dimensions
    for the input arrays. Given the right dimensionality, TipTorch will automatically understand what to do.
    '''

    config_manager = ConfigManager()
    config  = config_manager.Merge([config_file,] * N_srcs) # First, copy the same config for all sources in the field

    # Determine framework based on device parameter or existing data
    if device is not None:
        config_manager.Convert(config, framework='pytorch', device=device)
    else:
        config_manager.Convert(config, framework='numpy')
    
    config['NumberSources'] = N_srcs

    # The entries which are the same for all sources in the field
    shared_entries = [\
        ['telescope',  'ZenithAngle'],
        ['atmosphere', 'L0'],
        ['atmosphere', 'Seeing'],
        ['atmosphere', 'Cn2Weights'],
        ['atmosphere', 'Cn2Heights'],
        ['atmosphere', 'WindSpeed'],
        ['atmosphere', 'WindDirection'],
        ['sources_HO', 'Height'],
        ['sources_HO', 'Zenith'],
        ['sources_HO', 'Azimuth'],
        ['RTC', 'SensorFrameRate_HO'],
        ['RTC', 'LoopDelaySteps_HO'],
        ['RTC', 'LoopGain_HO'],
        ['DM',  'OptimizationZenith'],
        ['DM',  'OptimizationAzimuth'],
        ['DM',  'OptimizationWeight'],
        ['sensor_HO',  'NumberPhotons'],
        ['sensor_HO',  'SpotFWHM'],
        ['sensor_HO',  'SigmaRON'],
        ['sensor_HO',  'ExcessNoiseFactor'],
        ['sources_HO', 'Wavelength']
    ]
    
    entries_flatten = [\
        ['atmosphere', 'Seeing'],
        ['atmosphere', 'L0'],
        ['sources_HO', 'Wavelength'],
        ['sources_science', 'Zenith'],
        ['sources_science', 'Azimuth']
    ]
    
    # In theory, the only remaining per-source values, in this case, are sources_science Azimuth and Zenith.
    # All other entries must be shared among all sources. But just in case, the full merge of configs is done
    for section, entry in shared_entries:
        config[section][entry] = config[section][entry][0,...]
        to_flatten = [section, entry] in entries_flatten
        config[section][entry] = config_manager.ensure_dims(config[section][entry], 1, flatten=to_flatten)

    return config


def MultipleTargetsInDifferentObservations(configs, device=None):
    '''
    A wrapper unction that merges multiple config files into one for multiple targets observed in different conditions.
    Useful when training calibrator NN on multiple targets.
    '''
    config_manager = ConfigManager()
    merged_config  = config_manager.Merge(configs)

    # Check if any configs were provided
    if not configs:
        raise ValueError("No configuration files provided for merging.")
    
    # Check if any of configs is converted to pytorch. If yes, deconvert all to lists first
    for config in configs:
        sample_value = config['atmosphere']['Seeing']
        if isinstance(sample_value, torch.Tensor):
            config = config_manager.Convert(config, framework='list') 
            
    N_src = len(configs)
    
    # Convert to target framework
    if device is not None:
        config_manager.Convert(merged_config, framework='pytorch', device=device)
    else:
        config_manager.Convert(merged_config, framework='numpy')
    
    # Use ensure_dimensions to properly reshape all arrays
    merged_config['NumberSources'] = N_src

    return merged_config