import pickle
import torch
from torch.distributions.normal import Normal
from project_globals import device
import numpy as np
from scipy import stats
# from scipy.stats import boxcox, yeojohnson, norm
from scipy.optimize import curve_fit


class DataTransform:
    def __init__(self, data=None):
        if data is not None:
            self.fit(data)
    
    def fit(self, data):
        raise NotImplementedError("Subclasses must implement the 'fit' method.")
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the 'forward' method.")
    
    def backward(self, y):
        raise NotImplementedError("Subclasses must implement the 'backward' method.")
    
    def __call__(self, x):
        return self.forward(x)
    
    def get_params(self):
        # Generic get_params that returns non-callable, non-private attributes.
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)}
    
    def store(self):
        # Unified store method: return a tuple with the class name and parameters.
        return (self.__class__.__name__, self.get_params())


class YeoJohnson(DataTransform):
    def __init__(self, data=None, lmbda=None) -> None:
        self.lmbda = lmbda
        if data is not None and self.lmbda is None:
            self.fit(data)
    
    def fit(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.lmbda = stats.yeojohnson_normmax(data)
    
    def forward(self, x):
        if self.lmbda is None:
            raise ValueError("Lambda is not initialized. Call 'fit' with valid data first.")
        xp = torch if isinstance(x, torch.Tensor) else np
        pos = x >= 0
        neg = x < 0
        y = xp.zeros_like(x)
        if self.lmbda != 0:
            y[pos] = ((x[pos] + 1) ** self.lmbda - 1) / self.lmbda
            y[neg] = -((-x[neg] + 1) ** (2 - self.lmbda) - 1) / (2 - self.lmbda)
        else:
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

class BoxCox(DataTransform):
    def __init__(self, data=None, lmbda=None) -> None:
        self.lmbda = lmbda
        if data is not None and self.lmbda is None:
            self.fit(data)
    
    def fit(self, data):
        self.lmbda = stats.boxcox_normmax(data)
    
    def forward(self, x):
        xp = torch if isinstance(x, torch.Tensor) else np
        return xp.log(x) if abs(self.lmbda) < 1e-6 else (xp.abs(x)**self.lmbda - 1) / self.lmbda
    
    def backward(self, y):
        xp = torch if isinstance(y, torch.Tensor) else np
        return xp.exp(y) if abs(self.lmbda) < 1e-6 else (self.lmbda * xp.abs(y) + 1) ** (1 / self.lmbda)

class Gaussify(DataTransform):
    def __init__(self, data=None) -> None:
        # Initialize a normal distribution (assumes device and Normal are defined)
        self.normal_distribution = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
    
    def fit(self, data):
        pass
    
    def cdf(self, x):
        return self.normal_distribution.cdf(x) if isinstance(x, torch.Tensor) else stats.norm.cdf(x)
    
    def ppf(self, q):
        return (torch.sqrt(torch.tensor([2.]).to(device)) * torch.erfinv(2 * q - 1) 
                if isinstance(q, torch.Tensor) else stats.norm.ppf(q))
    
    def forward(self, x):
        return self.ppf(self.cdf(x))
    
    def backward(self, y):
        return self.cdf(self.ppf(y))

class Uniform(DataTransform):
    def __init__(self, data=None, a=None, b=None):
        self.a = a
        self.b = b
        self.uniform_scaler = lambda x, a, b: 2 * ((x - a) / (b - a) - 0.5)
        self.inv_uniform_scaler = lambda x, a, b: (x / 2 + 0.5) * (b - a) + a
        if data is not None and a is None and b is None:
            self.fit(data)
    
    def fit(self, data):
        self.a = data.min()
        self.b = data.max()
    
    def forward(self, x):
        return self.uniform_scaler(x, self.a, self.b)
    
    def backward(self, y):
        return self.inv_uniform_scaler(y, self.a, self.b)

class Uniform0_1(DataTransform):
    def __init__(self, data=None, a=None, b=None):
        self.a = a
        self.b = b
        self.uniform_scaler = lambda x, a, b: (x - a) / (b - a)
        self.inv_uniform_scaler = lambda x, a, b: x * (b - a) + a
        if data is not None and a is None and b is None:
            self.fit(data)
    
    def fit(self, data):
        self.a = data.min()
        self.b = data.max()
    
    def forward(self, x):
        return self.uniform_scaler(x, self.a, self.b)
    
    def backward(self, y):
        return self.inv_uniform_scaler(y, self.a, self.b)

class Gauss(DataTransform):
    def __init__(self, data=None, mu=None, std=None):
        self.mu = mu
        self.std = std
        if data is not None and mu is None and std is None:
            self.fit(data)
    
    def fit(self, data):
        self.mu, self.std = stats.norm.fit(data)
    
    def forward(self, x):
        return (x - self.mu) / self.std
    
    def backward(self, y):
        return y * self.std + self.mu

class Invert(DataTransform):
    def __init__(self, data=None):
        pass
    
    def fit(self, data):
        pass
    
    def forward(self, x):
        return 1 - x
    
    def backward(self, y):
        return 1 - y

class Logify(DataTransform):
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
        
class TransformSequence:
    def __init__(self, transforms=None):
        self.transforms = [transforms] if type(transforms) is DataTransform else transforms
        # self.transforms = transforms

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

    def get_params(self):
        return {'type': 'TransformSequence', 'transforms': [t.get_params() for t in self.transforms]}
    

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


class LineModel:
    """
    A PyTorch-compatible line model for non-linear optimization.

    The model function is defined as:
        f(x; k, y_intercept) = k * x + y_intercept
    """

    def __init__(self, x: torch.Tensor):
        """
        Initialize the LineModel.

        Parameters:
            x (torch.Tensor): Tensor of x-values.
        """
        self.x = x

    @staticmethod
    def line_function(x, k, y_intercept):
        """
        Compute the line function.

        Parameters:
            x (Tensor or numpy.ndarray): Input x-values.
            k (float): Slope.
            y_intercept (float): Y-intercept.

        Returns:
            Tensor or numpy.ndarray: Computed y-values.
        """
        return k * x + y_intercept

    def fit(self, y: torch.Tensor):
        """
        Fit the line model to target y-values using non-linear least squares.

        This method converts the input tensors to numpy arrays and then
        uses SciPy's curve_fit to estimate the parameters [k, y_intercept].

        Parameters:
            y (torch.Tensor): Tensor of target y-values.

        Returns:
            array: Optimal parameters [k, y_intercept].
        """
        # Convert tensors to 1D numpy arrays for curve_fit.
        x_np = self.x.flatten().cpu().numpy()
        y_np = y.flatten().cpu().numpy()

        # Define the fitting function.
        def fit_func(x, k, y_intercept):
            return self.line_function(x, k, y_intercept)

        popt, _ = curve_fit(fit_func, x_np, y_np, p0=[1e-6, 1e-6])
        return popt

    def __call__(self, params):
        """
        Evaluate the model using the provided parameters.

        Parameters:
            params (tuple): Parameters (k, y_intercept).

        Returns:
            Tensor: Computed y-values.
        """
        return self.line_function(self.x, *params)


class QuadraticModel:
    """
    A PyTorch-compatible quadratic model for non-linear optimization.

    The model function is defined as:
        f(x; a, b, c) = a * x^2 + b * x + c
    """

    def __init__(self, x: torch.Tensor):
        """
        Initialize the QuadraticModel.

        Parameters:
            x (torch.Tensor): Tensor of x-values.
        """
        self.x = x

    @staticmethod
    def quadratic_function(x, a, b, c):
        """
        Compute the quadratic function.

        Parameters:
            x (Tensor or numpy.ndarray): Input x-values.
            a (float): Quadratic coefficient.
            b (float): Linear coefficient.
            c (float): Constant term.

        Returns:
            Tensor or numpy.ndarray: Computed y-values.
        """
        return a * x ** 2 + b * x + c

    def fit(self, y: torch.Tensor, p0=[1e-6, 1e-6, 1e-6]):
        """
        Fit the quadratic model to target y-values using non-linear least squares.

        This method converts the input tensors to numpy arrays and then
        uses SciPy's curve_fit to estimate the parameters [a, b, c].

        Parameters:
            y (torch.Tensor): Tensor of target y-values.

        Returns:
            array: Optimal parameters [a, b, c].
        """
        # Convert tensors to 1D numpy arrays for curve_fit.
        x_np = self.x.flatten().cpu().numpy()
        y_np = y.flatten().cpu().numpy()

        # Define the fitting function.
        def fit_func(x, a, b, c):
            return self.quadratic_function(x, a, b, c)

        popt, _ = curve_fit(fit_func, x_np, y_np, p0=p0)
        return popt

    def __call__(self, params):
        """
        Evaluate the model using the provided parameters.

        Parameters:
            params (tuple): Parameters (a, b, c).

        Returns:
            Tensor: Computed y-values.
        """
        return self.quadratic_function(self.x, *params)
