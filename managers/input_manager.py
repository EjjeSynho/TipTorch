from dataclasses import dataclass
from typing import Any, Optional, Union
from tabulate import tabulate
from copy import deepcopy
import torch
import pickle
import numpy as np
from tools.normalizers import TransformSequence, DataTransform, Identity, LoadTransforms

"""
This module contains classes to manage the input data for the TipTorch PSF model. 
"""

class InputsTransformer:
    """
    A class to manage the input data for the TipTorch PSF model. It allows for the stacking of input dictionaries
    of into a single tensor and and unstacking it back. This is useful for non-linear optimization routines and when
    packing data to input into a NN. It also allow to standartize/normalize/transform model inputs. The class also
    provides methods for saving and loading the transformer.
    """
    def __init__(self, transforms: dict = {}):
        # Store the transforms provided as a dictionary
        self.transforms = transforms
        self.slices = {}

    def __len__(self):
        return len(self.transforms)

    def stack(self, args_dict, no_transform=False):
        if len(self.transforms) == 0:
            raise ValueError("No transforms provided for stacking.")
        """
        Constructs a joint tensor from provided keyword arguments, applying the corresponding transforms to each.
        Keeps track of each tensor's size for later unstacking by storing ciorresponding slices for each variable.
        NOTE: As consequence, the order of the arguments is important.
        NOTE: The 0th dimension is allocated to sources. Meaning that the tensor is of shape (n_sources, n_features)
        """
        self.slices = {}
        self._packed_size = None
        tensors = []
        current_index = 0

        for key, value in args_dict.items():
            if not no_transform:
                if key not in self.transforms:
                    raise ValueError(f"Transform for {key} not found.")

                transformed_value = self.transforms[key](value.unsqueeze(-1) if value.dim() == 1 else value)
            else:
                transformed_value = value.unsqueeze(-1) if value.dim() == 1 else value

            next_index = current_index + transformed_value.shape[1]
            self.slices[key] = slice(current_index, next_index)
            current_index = next_index
            tensors.append(transformed_value)

        # Concatenate all transformed tensors
        joint_tensor = torch.hstack(tensors)
        self._packed_size = joint_tensor.shape[1]
        return joint_tensor

    # Getter
    def get_stacked_size(self):
        return self._packed_size

    def unstack(self, x_stacked):
        if self.slices is None:
            raise ValueError("No cache of decomposing slices found. Stack the tensors first.")
        """
        Decomposes the joint tensor back into a dictionary of variables,
        inversely transforming each back to its original space.
        Uses the slices tracked during the stack operation.
        """
        decomposed = {}
        for key, sl in self.slices.items():
            val = self.transforms[key].inverse(x_stacked[:, sl])
            decomposed[key] = val.squeeze(-1) if sl.stop-sl.start<2 else val
            # expects that the 0th dimension is allocated to sources. Meaning that the tensor is of shape (n_sources, n_features)

        return decomposed

    def save(self, filename):
        """
        Serializes and saves the InputsTransformer to a file, including slice information.

        Args:
            filename (str): Path to the file where the transformer will be saved
        """
        save_data = {
            'transforms': {},
            'slices': {},
            'packed_size': self._packed_size
        }

        # Store transform data
        for key, transform in self.transforms.items():
            # For each transform, store its type and parameters
            if hasattr(transform, 'get_params'):
                save_data['transforms'][key] = transform.get_params()
            else:
                # If the transform is a sequence, handle specially
                if isinstance(transform, TransformSequence):
                    save_data['transforms'][key] = {
                        'type': 'TransformSequence',
                        'transforms': [t.get_params() for t in transform.transforms]
                    }
                else:
                    save_data['transforms'][key] = {'type': type(transform).__name__}

        # Store slice information
        for key, sl in self.slices.items():
            save_data['slices'][key] = {'start': sl.start, 'stop': sl.stop, 'step': sl.step}

        with open(filename, 'wb') as handle:
            pickle.dump(save_data, handle)

    @classmethod
    def load(cls, filename):
        """
        Loads an InputsTransformer from a file, including slice information if available.

        Args:
            filename (str): Path to the file containing the saved transformer

        Returns:
            InputsTransformer: A new instance with the loaded transforms and slices
        """
        with open(filename, 'rb') as handle:
            save_data = pickle.load(handle)

        # Handle backward compatibility with old format
        if not isinstance(save_data, dict) or 'transforms' not in save_data:
            transform_data = save_data
            save_data = {'transforms': transform_data, 'slices': {}}

        transforms = {}
        for key, params in save_data['transforms'].items():
            if isinstance(params, dict) and 'type' in params and params['type'] == 'TransformSequence':
                # Handle TransformSequence
                sub_transforms = []
                for t_params in params['transforms']:
                    transform_type = t_params.pop('type', None)
                    if transform_type:
                        sub_transforms.append(globals()[transform_type](**t_params))
                transforms[key] = TransformSequence(sub_transforms)
            else:
                # Use LoadTransforms for other transform types
                transforms[key] = LoadTransforms(params)
                
        # Create the instance
        instance = cls(transforms)

        # Restore slices if available
        if 'slices' in save_data and save_data['slices']:
            for key, slice_data in save_data['slices'].items():
                instance.slices[key] = slice(slice_data['start'], slice_data['stop'], slice_data['step'])

        # Restore packed size if available
        if 'packed_size' in save_data:
            instance._packed_size = save_data['packed_size']

        return instance


@dataclass
class InputParameter:
    value: Any
    transform: Optional[Any] = None
    optimizable: bool = True

class InputsManager:
    """
    The wrapper over the InputsTransformer class. It allows for more convenient management of the model inputs,
    enabling to add or delete them, as well as to control which inputs are stacker into a single tensor and which
    are not. It also allows for easy displaying of stored model inputs.
    """
    def __init__(self):
        self.parameters = {}
        self.inputs_transformer = InputsTransformer()

    def add(self,
            name: str,
            default_val: Any,
            transform: Union[DataTransform, TransformSequence] = Identity(),
            optimizable: bool = True):

        self.parameters[name] = InputParameter(
            value       = default_val,
            transform   = TransformSequence(transform) if not isinstance(transform, TransformSequence) else transform,
            optimizable = optimizable
        )

        self.inputs_transformer.transforms.update({name: transform})
        # self.inputs_transformer.transforms = dict(sorted(self.inputs_transformer.transforms.items(), key=lambda x: x[0]))
        # self.parameters = dict(sorted(self.parameters.items(), key=lambda x: x[0]))

    def get_stacked_size(self):
        """Get the size of the stacked tensor."""
        return self.inputs_transformer.get_stacked_size()

    def get_transformer(self) -> InputsTransformer:
        """Get the underlying InputsTransformer."""
        return self.inputs_transformer

    def get_value(self, name: str) -> Any:
        """Get the value of a parameter."""
        return self.parameters[name].value

    def get_transform(self, name: str) -> Any:
        """Get the transform of a parameter."""
        return self.parameters[name].transform

    def is_optimizable(self, name: str) -> bool:
        """Check if parameter is optimizable."""
        return self.parameters[name].optimizable

    def set_optimizable(self, names: Union[str, list], optimizable: bool):
        """Set optimizable status for a parameter or list of parameters.

        Args:
            names: Either a single parameter name (str) or a list of parameter names
            optimizable: Boolean flag to set optimizable status
        """
        if isinstance(names, str):
            self.parameters[names].optimizable = optimizable
        else:
            for name in names:
                self.parameters[name].optimizable = optimizable
        
        optimizable_names = [name for name, param in self.parameters.items() if param.optimizable]
    
        if set(optimizable_names) != set(self.inputs_transformer.slices.keys()):
            self.stack()
    
    def update(self, other: dict, selected_ids: Any = None):
        """Update the parameters with a new dictionary of values."""
        for name, value in other.items():
            if name in self.parameters:
                if selected_ids is not None:
                    self.parameters[name].value[selected_ids] = value[selected_ids]
                else:
                    self.parameters[name].value = value

    def delete(self, name: str):
        """Delete a parameter by name."""
        if name in self.parameters:
            del self.parameters[name]
            if name in self.inputs_transformer.slices:
                del self.inputs_transformer.slices[name]
                self.stack()
                
    def stack(self):
        if self.inputs_transformer.transforms == {} or self.parameters == {}:
            return None

        """Stack the parameters into a single tensor."""
        args_dict = {name: param.value for name, param in self.parameters.items() if param.optimizable}
        if len(args_dict) == 0:
            return None
        
        return self.inputs_transformer.stack(args_dict)

        
    def unstack(self, x: torch.Tensor, include_all=True, update=True):
        if self.inputs_transformer.transforms == {} or self.parameters == {}:
            return None

        """Unstack a tensor into the parameters."""
        args_dict = self.inputs_transformer.unstack(x)
        
        if update:     
            self.update(args_dict)
            # for name, value in args_dict.items():
            #     self.parameters[name].value = value
            
        # return all parameters, not just optimizable ones
        if include_all:
            for name, param in self.parameters.items():
                if name not in args_dict:
                    args_dict[name] = param.value
                    
        return args_dict

    def __len__(self):
        """Return number of parameters."""
        return len(self.parameters)
    
    def get_names(self, optimizable_only: bool = True, flattened: bool = False) -> list:
        """Get the list of parameter names.

        Args:
            optimizable_only (bool): Whether to return only optimizable parameter names.
            flattened (bool): Whether to return flattened parameter names for vector parameters
                              that map 1-to-1 to elements of flattened stacked tensor.
        Returns:
            list: List of parameter names.
        """
        if not flattened:
            if optimizable_only:
                return [name for name, param in self.parameters.items() if param.optimizable]
            else:
                return list(self.parameters.keys())
        else:
            param_names = []
            for key, sl in self.inputs_transformer.slices.items():
                size = sl.stop - sl.start
                if size == 1:
                    param_names.append(key)
                else:
                    # For vector parameters, create indexed names
                    for i in range(size):
                        param_names.append(f"{key}_{i}")
            return param_names
    
    
    def to_dict(self) -> dict:
        """Explicit method to convert to dictionary."""
        return {name: param.value for name, param in self.parameters.items()}

    def __getitem__(self, item):
        return self.parameters[item].value

    def __setitem__(self, key, value):
        self.parameters[key].value = value

    def to(self, device: torch.device):
        for name, param in self.parameters.items():
            self.parameters[name].value = param.value.to(device)

    def to_float(self):
        """Convert all parameter values to float32."""
        for name, param in self.parameters.items():
            if hasattr(param.value, 'float'):
                self.parameters[name].value = param.value.float()

    def to_double(self):
        """Convert all parameter values to float64."""
        for name, param in self.parameters.items():
            if hasattr(param.value, 'double'):
                self.parameters[name].value = param.value.double()

    def copy(self):
        """Return a new InputsManager instance with copied parameters and transforms."""
        new_manager = InputsManager()
        # Rebuild parameters dictionary
        for name, param in self.parameters.items():
            new_manager.parameters[name] = InputParameter(
                # value = deepcopy(param.value),
                value = param.value.clone() if isinstance(param.value, torch.Tensor) else deepcopy(param.value),
                transform = deepcopy(param.transform),
                optimizable = param.optimizable
            )
        # Copy the inputs_transformer
        new_manager.inputs_transformer = deepcopy(self.inputs_transformer)
        return new_manager
    
    def clone(self):
        return self.copy()

    def __str__(self) -> str:
        """Pretty print the InputsManager contents."""
        if not self.parameters:
            return "InputsManager: No parameters defined"

        # Prepare table headers and data
        headers = ["Parameter", "Shape", "Device", "Dtype", "Optimizable", "Transform"]
        rows = []

        for name, param in self.parameters.items():
            value = param.value
            # Get shape, device, and dtype info
            shape = tuple(value.shape) if hasattr(value, 'shape') else 'N/A'
            device = value.device if hasattr(value, 'device') else 'N/A'
            dtype = value.dtype if hasattr(value, 'dtype') else 'N/A'
            # Get transform info
            transform_name = type(param.transform.transforms[0]).__name__ if param.transform and param.transform.transforms else 'None'

            rows.append([
                name,
                str(shape),
                str(device),
                str(dtype),
                '✓' if param.optimizable else '✗',
                transform_name
            ])

        # Create the table
        table = tabulate(rows, headers=headers, tablefmt="pretty")

        # Add header and footer
        total_params = len(self.parameters)
        optimizable_params = sum(1 for param in self.parameters.values() if param.optimizable)

        header = "InputsManager Summary\n" + "="*len(table.split('\n')[0]) + "\n"
        footer = f"\nTotal parameters: {total_params} (Optimizable: {optimizable_params})"

        return header + table + footer


class InputsManagersUnion:
    """
    This class provides similar functionality to InputsManager for accessing and manipulating parameters across
    multiple InputsManager instances. Using multiple InputsManager objects can be useful for example when working with
    multiple models or when you want to keep track of different sets of parameters separately.
    """
    
    def __init__(self, input_managers: Union[dict, list] = {}) -> None:
        """
        Initialize with a dictionary of InputsManager objects.

        Args:
            input_managers (Union[dict, list]): Dictionary of InputsManager objects where keys
                                              are identifiers and values are InputsManager instances.
                                              Can also accept a list for backward compatibility.
                                              Order matters - the latter input manager will
                                              overwrite the former values during unstacking.
        """
        # Convert list to dictionary if needed
        if isinstance(input_managers, list):
            self.input_managers = {i: manager for i, manager in enumerate(input_managers)}
        else:
            self.input_managers = input_managers

        self.slices = {}
        self.shapes = {}  # Store original shapes
        self._stacked_size = None
        
    
    def stack(self) -> torch.Tensor:
        """
        Stack all InputsManager objects into a single tensor.

        Returns:
            torch.Tensor: Combined tensor of all InputsManager stacked values.
        """
        self.slices = {}
        self.shapes = {}  # Reset shapes
        
        vectors = []
        current_idx = 0
        
        for key, manager in self.input_managers.items():
            stacked = manager.stack()
            if stacked is None:
                continue
                
            # Store original shape
            self.shapes[key] = stacked.shape
            stacked_flat = stacked.flatten()

            # Store slice for this manager
            next_idx = current_idx + stacked_flat.shape[-1]
            self.slices[key] = slice(current_idx, next_idx)
            current_idx = next_idx
            
            vectors.append(stacked_flat)
        
        if not vectors:
            return None
            
        # Concatenate all vectors
        joint_vector = torch.cat(vectors, dim=0)
        self._stacked_size = len(joint_vector)
        return joint_vector

    def unstack(self, vector: torch.Tensor, include_all: bool = True, update: bool = True) -> dict:
        """
        Unstack the combined tensor back into a dictionary of parameters.

        Args:
            vector (torch.Tensor): Stacked tensor to unstack.
            include_all (bool): Whether to include all parameters stored in the manager(s) or just optimizable ones.
            update (bool): Whether to update InputsManager internal values.

        Returns:
            dict: Dictionary containing unstacked parameters.

        Raises:
            ValueError: If slices are not defined (stack() must be called first).
        """
        if not self.slices:
            raise ValueError("No slices defined. Call stack() first.")
            
        result = {}
        
        for key, sl in self.slices.items():
            manager = self.input_managers[key]
            original_shape = self.shapes[key]

            # Extract the flat vector related to the current manager and reshape to original dimensions
            manager_vector = vector[..., sl]
            if len(original_shape) > 1:
                manager_vector = manager_vector.reshape(original_shape)
            
            unstacked = manager.unstack(manager_vector, include_all=include_all, update=update)
            if unstacked is not None:
                result.update(unstacked)
                
        # If not all managers were included in slices due to absence of optimizable parameters in them,
        # add their contents into the resulting dictionary anyway
        if include_all and (len(self.slices) != len(self.input_managers)):
            for key, manager in self.input_managers.items():
                if key not in self.slices:
                    result.update(manager.to_dict())
            
        return result

    def get_value(self, name: str) -> Any:
        """Get the value of a parameter from the first manager that contains it.

        Args:
            name: Parameter name to look for

        Returns:
            The parameter value

        Raises:
            KeyError: If parameter is not found in any manager
        """
        for manager in self.input_managers.values():
            if name in manager.parameters:
                return manager.parameters[name].value
        raise KeyError(f"Parameter '{name}' not found in any input manager")

    def get_transform(self, name: str) -> Any:
        """Get the transform of a parameter from the first manager that contains it.

        Args:
            name: Parameter name to look for

        Returns:
            The parameter transform

        Raises:
            KeyError: If parameter is not found in any manager
        """
        for manager in self.input_managers.values():
            if name in manager.parameters:
                return manager.parameters[name].transform
        raise KeyError(f"Parameter '{name}' not found in any input manager")

    def set_optimizable(self, names: Union[str, list], optimizable: bool):
        """Set optimizable status for a parameter or list of parameters across all input managers.

        Args:
            names: Either a single parameter name (str) or a list of parameter names
            optimizable: Boolean flag to set optimizable status
        """
        name_list = [names] if isinstance(names, str) else names

        for manager in self.input_managers.values():
            for name in name_list:
                if name in manager.parameters:
                    manager.set_optimizable(name, optimizable)

    def is_optimizable(self, name: str) -> bool:
        """Check if parameter is optimizable in the first manager that contains it.

        Args:
            name: Parameter name to look for

        Returns:
            Boolean indicating if parameter is optimizable

        Raises:
            KeyError: If parameter is not found in any manager
        """
        for manager in self.input_managers.values():
            if name in manager.parameters:
                return manager.parameters[name].optimizable
        raise KeyError(f"Parameter '{name}' not found in any input manager")

    def to_dict(self) -> dict:
        """Convert all parameters from all managers to a dictionary.

        Returns:
            dict: Combined dictionary of all parameters
        """
        result = {}
        for manager in self.input_managers.values():
            result.update(manager.to_dict())
        return result

    def __getitem__(self, item):
        """Get a parameter value using dictionary-like access.

        Args:
            item: Parameter name

        Returns:
            Parameter value
        """
        return self.get_value(item)

    def __setitem__(self, key, value):
        """Set a parameter value using dictionary-like access.
        Sets the value in all managers that contain the parameter.

        Args:
            key: Parameter name
            value: New value
        """
        for manager in self.input_managers.values():
            if key in manager.parameters:
                manager.parameters[key].value = value

    def to(self, device: torch.device):
        """Move all parameters to the specified device.

        Args:
            device: Target device
        """
        for manager in self.input_managers.values():
            manager.to(device)

    def to_float(self):
        """Convert all parameter values to float32."""
        for manager in self.input_managers.values():
            manager.to_float()

    def to_double(self):
        """Convert all parameter values to float64."""
        for manager in self.input_managers.values():
            manager.to_double()

    def delete(self, name: str):
        """Delete a parameter from all managers that contain it."""
        for manager in self.input_managers.values():
            if name in manager.parameters:
                manager.delete(name)
          
    def update(self, other: dict, selected_ids: Any = None):
        """Update the parameters with a new dictionary of values."""
        for manager in self.input_managers.values():
            for name, value in other.items():
                if name in manager.parameters:
                    if selected_ids is not None:
                        manager.parameters[name].value[selected_ids] = value[selected_ids]
                    else:
                        manager.parameters[name].value = value

    def copy(self) -> "InputsManagersUnion":
        """Create a deep copy of the InputsManagersUnion."""
        new_union = InputsManagersUnion()
        for key, manager in self.input_managers.items():
            new_union.input_managers[key] = manager.copy()
        return new_union
        
    def clone(self) -> "InputsManagersUnion":
        """Create a deep copy of the InputsManagersUnion."""
        return self.copy()

    def __str__(self) -> str:
        """Pretty print the InputsManagersUnion contents."""
        if not self.input_managers:
            return "InputsManagersUnion: No input managers defined"

        # Collect all parameters from all managers
        all_params = {}
        manager_has_param = {}  # Track which managers have which parameters

        for key, manager in self.input_managers.items():
            for name, param in manager.parameters.items():
                if name not in all_params:
                    all_params[name] = param
                    manager_has_param[name] = []
                manager_has_param[name].append(key)

        if not all_params:
            return "InputsManagersUnion: No parameters defined in any manager"

        # Prepare table headers and data
        headers = ["Parameter", "Shape", "Device", "Dtype", "Optimizable", "Transform", "Managers"]
        rows = []

        for name, param in all_params.items():
            value = param.value
            # Get shape, device, and dtype info
            shape = tuple(value.shape) if hasattr(value, 'shape') else 'N/A'
            device = value.device if hasattr(value, 'device') else 'N/A'
            dtype = value.dtype if hasattr(value, 'dtype') else 'N/A'
            # Get transform info
            transform_name = type(param.transform.transforms[0]).__name__ if param.transform and param.transform.transforms else 'None'
            # Which managers have this parameter
            manager_keys = manager_has_param[name][0]

            rows.append([
                name,
                str(shape),
                str(device),
                str(dtype),
                '✓' if param.optimizable else '✗',
                transform_name,
                str(manager_keys)
            ])

        # Create the table
        table = tabulate(rows, headers=headers, tablefmt="pretty")

        # Add header and footer
        total_params = len(all_params)
        optimizable_params = sum(1 for param in all_params.values() if param.optimizable)

        header = f"InputsManagersUnion Summary ({len(self.input_managers)} managers)\n" + "="*len(table.split('\n')[0]) + "\n"
        footer = f"\nTotal unique parameters: {total_params} (Optimizable: {optimizable_params})"

        # Warning if some parameters were redefined in multiple managers
        redefined_params = [name for name, keys in manager_has_param.items() if len(keys) > 1]
        if redefined_params:
            footer += "\n\n (!) Warning: The following parameters were redefined in multiple managers:\n" + ", ".join(redefined_params) + \
                      "\nOnly the last definition will be used. The shape is displayed for the first definition."

        return header + table + footer
