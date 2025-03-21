import pickle
import torch
from torch.distributions.normal import Normal
from project_globals import device
import numpy as np
from scipy import stats
# from scipy.stats import boxcox, yeojohnson, norm
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Any, Optional, Union
from tabulate import tabulate  # For pretty table formatting
from copy import deepcopy

from collections import OrderedDict

class InputsTransformer:
    def __init__(self, transforms: Union[OrderedDict, dict] = OrderedDict()):
        # Store the transforms provided as a dictionary
        self.transforms = OrderedDict(transforms) if type(transforms) is dict else transforms
        self.slices = OrderedDict()

    def __len__(self):
        return len(self.transforms)

    def stack(self, args_dict, no_transform=False):
        if len(self.transforms) == 0:
            raise ValueError("No transforms provided for stacking.")
        """
        Constructs a joint tensor from provided keyword arguments,
        applying the corresponding transforms to each.
        Keeps track of each tensor's size for later decomposition.
        """
        self.slices = OrderedDict()
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

    def unstack(self, joint_tensor):
        if self.slices is None:
            raise ValueError("No cache of decomposing slices found. Stack the tensors first.")
        """
        Decomposes the joint tensor back into a dictionary of variables,
        inversely transforming each back to its original space.
        Uses the slices tracked during the stack operation.
        """
        decomposed = {}
        for key, sl in self.slices.items():
            val = self.transforms[key].backward(joint_tensor[:, sl])
            decomposed[key] = val.squeeze(-1) if sl.stop-sl.start<2 else val # expects the TipTorch's conventions about the tensors dimensions

        return decomposed

@dataclass
class InputParameter:
    value: Any
    transform: Optional[Any] = None
    optimizable: bool = True

class InputsManager:
    def __init__(self):
        self.parameters = OrderedDict()
        self.inputs_transformer = InputsTransformer()

    def add(self, name: str, default_val: Any, transform: Any = None, optimizable: bool = True):
        self.parameters[name] = InputParameter(
            value       = default_val,
            transform   = TransformSequence(transforms=[transform]) if type(transform) is not TransformSequence else transform,
            optimizable = optimizable
        )

        self.inputs_transformer.transforms.update({name: transform})
        # self.inputs_transformer.transforms = dict(sorted(self.inputs_transformer.transforms.items(), key=lambda x: x[0]))
        # self.parameters = dict(sorted(self.parameters.items(), key=lambda x: x[0]))

    def get_stacked_size(self):
        """Get the size of the stacked tensor."""
        return self.inputs_transformer.get_stacked_size()

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
    
    def update(self, other: Union[dict, OrderedDict]):
        """Update the parameters with a new dictionary of values."""
        for name, value in other.items():
            if name in self.parameters:
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
        return self.inputs_transformer.stack(args_dict)

    def unstack(self, x: torch.Tensor, include_all=True):
        if self.inputs_transformer.transforms == {} or self.parameters == {}:
            return None

        """Unstack a tensor into the parameters."""
        args_dict = self.inputs_transformer.unstack(x)
        
        for name, value in args_dict.items():
            self.parameters[name].value = value
        
        # return all parameters, not just optimizable ones
        if include_all:
            for name, param in self.parameters.items():
                if name not in args_dict:
                    args_dict[name] = param.value
                    
        return args_dict

    def __len__(self):
        """Return number of parameters."""
        return len(self.parameters)
        
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
                value=deepcopy(param.value),
                transform=deepcopy(param.transform),
                optimizable=param.optimizable
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

class LineModel:
    def __init__(self, x, norms):
        self.norms = norms
        self.x = x
        self.x_min = x.min().item()

    @staticmethod
    def line(x, k, y_min, x_min, A, B):
        return (A * k * (x - x_min) + y_min) * B

    def fit(self, y):
        x_ = self.x.flatten().cpu().numpy()
        y_ = y.flatten().cpu().numpy()

        func = lambda x, k, y_min: self.line(x, k, y_min, self.x_min, *self.norms)
        popt, _ = curve_fit(func, x_, y_, p0=[1e-6, 1e-6])
        return popt
    
    def __call__(self, params):
        return self.line(self.x, *params, self.x_min, *self.norms)


class QuadraticModel:
    def __init__(self, x, norms):
        self.norms = norms
        self.x = x
        self.x_min = x.min().item()

    @staticmethod
    def poly(x, a, b, c, x_min, A, B):
        y_2 = A * x-x_min
        return (a*y_2**2 + b*y_2 + c) * B
    
    def fit(self, y):
        # Flatten the arrays and move data to CPU and numpy for processing with curve_fit
        x_ = self.x.flatten().cpu().numpy()
        y_ = y.flatten().cpu().numpy()

        # Define the fitting function adjusting to the model's x_min and norms
        func = lambda x, a, b, c: self.poly(x, a, b, c, self.x_min, *self.norms)
        
        # Use curve_fit to find the optimal parameters
        popt, _ = curve_fit(func, x_, y_, p0=[1e-6, 1e-6, 1e-6])
        return popt

    def __call__(self, params):
        # Call the polynomial function with the found parameters and stored x-values
        return self.poly(self.x, *params, self.x_min, *self.norms)


class InputsCompressor:
    def __init__(self, transforms, models):
        self.transforms = transforms
        self.models = models
        self.slices = {}
        self._packed_size = None

    def stack(self, args_dict, no_transform=False):
        tensors = []
        current_index = 0

        for key, value in args_dict.items():
            if key in self.models:
                transformed_value = value.unsqueeze(-1) if value.ndim == 1 else value
            else:
                if not no_transform:
                    transformed_value = self.transforms[key](value.unsqueeze(-1) if value.dim() == 1 else value)

            next_index = current_index + transformed_value.numel()
            self.slices[key] = slice(current_index, next_index)
            current_index = next_index
            tensors.append(transformed_value.view(-1))

        joint_tensor = torch.cat(tensors)
        self._packed_size = joint_tensor.shape[0]
        return joint_tensor

    def get_stacked_size(self):
        return self._packed_size

    def unstack(self, joint_tensor):
        decomposed = {}
        for key, sl in self.slices.items():
            if key in self.models.keys():
                params = joint_tensor[:, sl]
                
                y_pred = self.models[key](*params)
                decomposed[key] = y_pred.squeeze(-1) if sl.stop - sl.start < 2 else y_pred
            else:
                val = self.transforms[key].backward(joint_tensor[:, sl])
                decomposed[key] = val.squeeze(-1) if sl.stop - sl.start < 2 else val

        return decomposed


class YeoJohnson:
    def __init__(self, data=None, lmbda=None) -> None:
        self.lmbda = lmbda
        # self.__numerical_stability = 1e-12
        
        if data is not None and self.lmbda is None:
            self.fit(data)

    def fit(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.lmbda = stats.yeojohnson_normmax(data)
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        if self.lmbda is None:
            raise ValueError("Lambda is not initialized. Call 'fit' with valid data first.")

        xp = torch if isinstance(x, torch.Tensor) else np
        # Apply transformations carefully to avoid domain issues.
        pos = x >= 0
        neg = x < 0
        
        # Compute only for valid parts of the domain.
        y = xp.zeros_like(x)  # Initialize y with the same shape and type as x.
        if self.lmbda != 0:
            y[pos] = ((x[pos] + 1) ** self.lmbda - 1) / self.lmbda
            y[neg] = -((-x[neg] + 1) ** (2 - self.lmbda) - 1) / (2 - self.lmbda)
        else:  # Handle the lambda = 0 case separately to avoid division by zero.
            y[pos] = xp.log(x[pos] + 1)
            y[neg] = -xp.log(-x[neg] + 1)
        return y

    def backward(self, y):
        if self.lmbda is None:
            raise ValueError("Lambda is not initialized. Call 'fit' with valid data first.")
        
        xp = torch if isinstance(y, torch.Tensor) else np
        x = xp.zeros_like(y)
        
        pos = y >= 0
        neg = y < 0
        
        if self.lmbda != 0:
            x[pos] = (self.lmbda * y[pos] + 1) ** (1 / self.lmbda) - 1
            x[neg] = -((2 - self.lmbda) * -y[neg] + 1) ** (1 / (2 - self.lmbda)) + 1
        else:
            x[pos] = xp.exp(y[pos]) - 1
            x[neg] = -(xp.exp(-y[neg]) - 1)
        return x

    def store(self):
        return ('YeoJohnson', self.lmbda)
    
    '''
    # This case is better differentiable
    def forward(self, x):
        if self.lmbda is None:
            raise ValueError("Lambda is not initialized. Call 'fit' with valid data first.")

        xp = torch if isinstance(x, torch.Tensor) else np

        x_safe = x + self.__numerical_stability

        pos            = x_safe >= 0
        neg            = x_safe < 0
        lmbda_zero     = self.lmbda == 0
        lmbda_not_zero = self.lmbda != 0
        lmbda_two      = self.lmbda == 2
        lmbda_not_two  = self.lmbda != 2

        y = xp.where(pos & lmbda_not_zero, ((x_safe + 1)**self.lmbda - 1) / self.lmbda, 0) + \
            xp.where(pos & lmbda_zero, xp.log(x_safe + 1), 0) + \
            xp.where(neg & lmbda_not_two, -((-x_safe + 1)**(2 - self.lmbda) - 1) / (2 - self.lmbda), 0) + \
            xp.where(neg & lmbda_two, -xp.log(-x_safe + 1), 0)

        return y   
    

    def backward(self, y):
        if self.lmbda is None:
            raise ValueError("Lambda is not initialized. Call 'fit' with valid data first.")
        
        # Dynamically select the appropriate library based on y's type
        xp = torch if isinstance(y, torch.Tensor) else np
        
        # To ensure stability in computations, especially with exp and log, adjustments are made
        y_safe = y + self.__numerical_stability

        pos            = y_safe >= 0
        neg            = y_safe < 0
        lmbda_zero     = self.lmbda == 0
        lmbda_not_zero = self.lmbda != 0
        lmbda_two      = self.lmbda == 2
        lmbda_not_two  = self.lmbda != 2

        # Computing the inverse transformation with adjusted conditions
        x = xp.where(pos & lmbda_not_zero, (self.lmbda * y_safe + 1)**(1 / self.lmbda) - 1, 0) + \
            xp.where(pos & lmbda_zero, xp.exp(y_safe) - 1, 0) + \
            xp.where(neg & lmbda_not_two, -((2 - self.lmbda) * -y_safe + 1)**(1 / (2 - self.lmbda)) + 1, 0) + \
            xp.where(neg & lmbda_two, -(xp.exp(-y_safe) - 1), 0)
            
        return x
    '''

class BoxCox:
    def __init__(self, data=None, lmbda=None) -> None:
        self.lmbda = lmbda
        if data is not None and self.lmbda is None:
            self.fit(data)

    def fit(self, data):
        self.lmbda = stats.boxcox_normmax(data)

    def forward(self, x):
        xp = torch if isinstance(x, torch.Tensor) else np
        # x = xp.abs(x+1e-12)
        return xp.log(x) if abs(self.lmbda)<1e-6 else (xp.abs(x)**self.lmbda-1)/self.lmbda

    def backward(self, y):
        xp = torch if isinstance(y, torch.Tensor) else np
        # y = xp.abs(y+1e-12) 
        return xp.exp(y) if abs(self.lmbda)<1e-6 else (self.lmbda*xp.abs(y)+1)**(1/self.lmbda)

    def __call__(self, x):
        return self.forward(x)
    
    
    def store(self):
        return ('BoxCox', self.lmbda)


class Gaussify:
    def __init__(self, data=None) -> None:
        self.normal_distribution = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

    def fit(self, data):
        pass

    def cdf(self, x):
        return self.normal_distribution.cdf(x) if isinstance(x, torch.Tensor) else  stats.norm.cdf(x)
        
    def ppf(self, q):
        return torch.sqrt(torch.tensor([2.]).to(device)) * torch.erfinv(2*q-1) if isinstance(q, torch.Tensor) else stats.norm.ppf(q)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.ppf(self.cdf(x))
    
    def backward(self, y):
        return self.cdf(self.ppf(y))
    
    def store(self):
        return ('Gaussify')


class Uniform:
    def __init__(self, data=None, a=None, b=None):
        self.a = a
        self.b = b
        self.scale = lambda: self.b-self.a
        
        self.uniform_scaler     = lambda x, a, b: 2*((x-a)/(b-a)-0.5)
        self.inv_uniform_scaler = lambda x, a, b: (x/2 + 0.5)*(b-a)+a
        
        if data is not None and a is None and b is None:
            self.fit(data)
    
    def fit(self, data):
        self.a = data.min()
        self.b = data.max()
        
    def forward(self, x):
        return self.uniform_scaler(x, self.a, self.b)
    
    def backward(self, y):
        return self.inv_uniform_scaler(y, self.a, self.b)
    
    def __call__(self, x):
        return self.forward(x)
    
    def store(self):
        return ('Uniform', (self.a, self.b))

class Uniform0_1:
    def __init__(self, data=None, a=None, b=None):
        self.a = a
        self.b = b
        
        self.uniform_scaler     = lambda x, a, b: (x-a)/(b-a)
        self.inv_uniform_scaler = lambda x, a, b: x*(b-a)+a
        
        if data is not None and a is None and b is None:
            self.fit(data)
    
    def fit(self, data):
        self.a = data.min()
        self.b = data.max()
        
    def forward(self, x):
        return self.uniform_scaler(x, self.a, self.b)
    
    def backward(self, y):
        return self.inv_uniform_scaler(y, self.a, self.b)
    
    def __call__(self, x):
        return self.forward(x)
    
    def store(self):
        return ('Uniform0_1', (self.a, self.b))


class Gauss:
    def __init__(self, data=None, mu=None, std=None):
        self.mu  = mu
        self.std = std
        if data is not None and mu is None and std is None:
            self.fit(data)
            
    def fit(self, data):
        self.mu, self.std = stats.norm.fit(data)
        
    def forward(self, x):
        return (x-self.mu) / self.std
    
    def backward(self, y):
        return y * self.std + self.mu
    
    def __call__(self, x):
        return self.forward(x)
    
    def store(self):
        return ('Gauss', (self.mu, self.std))

class Invert:
    def __init__(self, data=None):
        pass
    
    def fit(self, data):
        pass
    
    def forward(self, x):
        return 1-x
    
    def backward(self, y):
        return 1-y
    
    def __call__(self, x):
        return self.forward(x)
    
    def store(self):
        return 'Invert'
    
    
class Logify:
    def __init__(self, data=None):
        pass
    
    def fit(self, data=None):
        pass
    
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return torch.log(x)
        else:
            return np.log(x)
    
    def backward(self, y):
        if isinstance(y, torch.Tensor):
            return torch.exp(y)
        else:
            return np.exp(y)
    
    def __call__(self, x):
        return self.forward(x)
    
    def store(self):
        return 'Logify'
    
    
class TransformSequence:
    def __init__(self, transforms=None):
        self.transforms = transforms

    def add(self, transform):
        self.transforms.append(transform)

    def forward(self, x):
        for transform in self.transforms:
            x = transform.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, y):
        for transform in reversed(self.transforms):
            y = transform.backward(y)
        return y
    
    def store(self):
        return [transform.store() for transform in self.transforms]
    
    def __len__(self):
        return len(self.transforms)


def LoadTransforms(state):
    transforms = []
    for transform in state: #[name, param_1, param_2, ...]
        if type(transform) is str:
            transform = [transform]
        if   transform[0] == 'YeoJohnson': transforms.append(YeoJohnson(lmbda=transform[1]))
        elif transform[0] == 'BoxCox':     transforms.append(BoxCox(lmbda=transform[1]))
        elif transform[0] == 'Gaussify':   transforms.append(Gaussify())
        elif transform[0] == 'Uniform':    transforms.append(Uniform(a=transform[1][0], b=transform[1][1]))
        elif transform[0] == 'Uniform0_1': transforms.append(Uniform0_1(a=transform[1][0], b=transform[1][1]))
        elif transform[0] == 'Gauss':      transforms.append(Gauss(mu=transform[1][0], std=transform[1][1]))
        elif transform[0] == 'Invert':     transforms.append(Invert())
        elif transform[0] == 'Logify':     transforms.append(Logify())
        else:
            raise ValueError(f"Unknown transform \"{transform[0]}\"!") 
    return TransformSequence(transforms)


def CreateTransformSequence(entry, df, transforms_list, verbose=True):
    data      = df[entry].replace([np.inf, -np.inf], np.nan).dropna().values.astype(np.float64)
    data_init = df[entry].replace([np.inf, -np.inf], np.nan).dropna().values.astype(np.float64)

    transforms = []
    for transform in transforms_list:
        data = (trans := transform(data)).forward(data)
        transforms.append(trans)

    sequence = TransformSequence(transforms=transforms)
    test = sequence.forward(data_init)
    
    if verbose:
        recover = sequence.backward(test)
        diff = np.abs(data_init-recover).sum()
        print('Error:', diff, '\n')
    
    return sequence


def CreateTransformSequenceFromFile(filename):
    with open(filename, 'rb') as handle:
        df_transforms_stored = pickle.load(handle)
        
    df_transforms = {entry: LoadTransforms(df_transforms_stored[entry]) for entry in df_transforms_stored}
     
    return df_transforms