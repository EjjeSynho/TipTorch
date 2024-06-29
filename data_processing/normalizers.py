import pickle
import torch
from torch.distributions.normal import Normal
from project_globals import device
import numpy as np

from scipy import stats
from scipy.stats import boxcox, yeojohnson, norm
from scipy.optimize import curve_fit


class InputsTransformer:
    def __init__(self, transforms):
        # Store the transforms provided as a dictionary
        self.transforms = transforms

    def stack(self, args_dict, no_transform=False):
        """
        Constructs a joint tensor from provided keyword arguments,
        applying the corresponding transforms to each.
        Keeps track of each tensor's size for later decomposition.
        """
        self._slices = {}
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
            self._slices[key] = slice(current_index, next_index)
            current_index = next_index
            tensors.append(transformed_value)
        
        # Concatenate all transformed tensors
        joint_tensor = torch.hstack(tensors)
        self._packed_size = joint_tensor.shape[1]
        return joint_tensor

    # Getter
    def get_packed_size(self):
        return self._packed_size

    def destack(self, joint_tensor):
        """
        Decomposes the joint tensor back into a dictionary of variables,
        inversely transforming each back to its original space.
        Uses the slices tracked during the stack operation.
        """
        decomposed = {}
        for key, sl in self._slices.items():
            val = self.transforms[key].backward(joint_tensor[:, sl])
            decomposed[key] = val.squeeze(-1) if sl.stop-sl.start<2 else val # expects the TipTorch's conventions about the tensors dimensions
        
        return decomposed


class LineModel:
    def __init__(self, x, norms):
        self.norms = norms
        self.x = x
        self.x_min = x.min().item()

    @staticmethod
    def line(λ, k, y_min, λ_min, A, B):
        return (A * k * (λ - λ_min) + y_min) * B

    def fit(self, y):
        x_ = self.x.flatten().cpu().numpy()
        y_ = y.flatten().cpu().numpy()

        func = lambda x, k, y_min: self.line(x, k, y_min, self.x_min, *self.norms)
        popt, _ = curve_fit(func, x_, y_, p0=[1e-6, 1e-6])
        return popt
    
    def __call__(self, params):
        return self.line(self.x, *params, self.x_min, *self.norms)


class InputsCompressor:
    def __init__(self, transforms, models):
        self.transforms = transforms
        self.models = models
        self._slices = {}
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
            self._slices[key] = slice(current_index, next_index)
            current_index = next_index
            tensors.append(transformed_value.view(-1))

        joint_tensor = torch.cat(tensors)
        self._packed_size = joint_tensor.shape[0]
        return joint_tensor

    def get_packed_size(self):
        return self._packed_size

    def destack(self, joint_tensor):
        decomposed = {}
        for key, sl in self._slices.items():
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
    data = df[entry].values
    data_init = df[entry].values

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