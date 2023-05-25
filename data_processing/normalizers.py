import torch
from torch.distributions.normal import Normal
from project_globals import device
import numpy as np

from scipy import stats
from scipy.stats import boxcox, yeojohnson, norm


class DataTransformer:
    def __init__(self, data, boxcox=True, gaussian=True, uniform=False, invert=False) -> None:
        self.boxcox_flag   = boxcox
        self.gaussian_flag = gaussian
        self.uniform_flag  = uniform
        self.invert_flag   = invert
        
        self.std_scaler     = lambda x, m, s: (x-m)/s
        self.inv_std_scaler = lambda x, m, s: x*s+m

        self.uniform_scaler     = lambda x, a, b: (x-a)/(b-a)
        self.inv_uniform_scaler = lambda x, a, b: x*(b-a)+a
                        
        self.one_minus     = lambda x: 1-x if self.invert_flag else x
        self.inv_one_minus = lambda x: 1-x if self.invert_flag else x
        
        self.normal_distribution = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))

        if data is not None: self.fit(data)
    
    def boxcox(self, x, λ):
        if isinstance(x, torch.Tensor):
            return torch.log(x+1e-6) if abs(λ)<1e-6 else ((x+1e-6)**λ-1)/λ
        else:
            return np.log(x+1e-6) if abs(λ)<1e-6 else ((x+1e-6)**λ-1)/λ
    
    def inv_boxcox(self, y, λ):
        if isinstance(y, torch.Tensor) :
            return torch.exp(y) if abs(λ)<1e-6 else (λ*(y+1e-6)+1)**(1/λ)
        else:
            return np.exp(y) if abs(λ)<1e-6 else (λ*(y+1e-6)+1)**(1/λ)

    def cdf(self, x):
        return self.normal_distribution.cdf(x) if isinstance(x, torch.Tensor) else  stats.norm.cdf(x)
        
    def ppf(self, q):
        return torch.sqrt(torch.tensor([2.]).to(device)) * torch.erfinv(2*q-1) if isinstance(q, torch.Tensor) else stats.norm.ppf(q)
    
    def fit(self, data):
        if self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            data_standartized, self.lmbd = stats.boxcox(self.one_minus(data))
            self.mu, self.std = stats.norm.fit(data_standartized)
            
        elif self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            data_standartized, self.lmbd = stats.boxcox(self.one_minus(data))
            self.a = data_standartized.min()*0.99
            self.b = data_standartized.max()*1.01
        
        elif not self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            self.mu, self.std = stats.norm.fit(self.one_minus(data))

        elif not self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            self.a = data.min()
            self.b = data.max()

    def forward(self, x):
        if self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            return self.std_scaler(self.boxcox(self.one_minus(x), self.lmbd), self.mu, self.std)
        
        elif self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            return self.ppf( self.uniform_scaler(self.boxcox(self.one_minus(x), self.lmbd), self.a, self.b) )
        
        elif not self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            return self.std_scaler(self.one_minus(x), self.mu, self.std)
        
        elif not self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            return self.uniform_scaler(self.one_minus(x), self.a, self.b)*2-1

    def backward(self, x):
        if self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            return self.inv_one_minus(self.inv_boxcox(self.inv_std_scaler(x, self.mu, self.std), self.lmbd))
        
        elif self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            return self.inv_one_minus(self.inv_boxcox(self.inv_uniform_scaler(self.cdf(x), self.a, self.b), self.lmbd))
        
        elif not self.boxcox_flag and self.gaussian_flag and not self.uniform_flag:
            return self.inv_one_minus(self.inv_std_scaler(x, self.mu, self.std))
        
        elif not self.boxcox_flag and not self.gaussian_flag and self.uniform_flag:
            return (self.inv_one_minus(self.inv_uniform_scaler((x+1)/2, self.a, self.b)) )


class YeoJohnson:
    def __init__(self, data=None) -> None:
        if data is None:
            self.lmbda = None
        else:
            self.fit(data)

    def fit(self, data):
        self.lmbda = stats.yeojohnson_normmax(data)

    def forward(self, x):
        x += 1e-9
        xp = torch if isinstance(x, torch.Tensor) else np
        pos = x >= 0
        neg = x < 0
        lmbda_zero = self.lmbda == 0
        lmbda_not_zero = self.lmbda != 0
        lmbda_two = self.lmbda == 2
        lmbda_not_two = self.lmbda != 2

        y = xp.where(pos * lmbda_not_zero, ((x + 1)**self.lmbda - 1) / self.lmbda, 0) + \
            xp.where(pos * lmbda_zero, xp.log(x + 1), 0) + \
            xp.where(neg * lmbda_not_two, -((-x + 1)**(2 - self.lmbda) - 1) / (2 - self.lmbda), 0) + \
            xp.where(neg * lmbda_two, -xp.log(-x + 1), 0)    
        return y

    def __call__(self, x):
        return self.forward(x)

    def backward(self, y):
        y += 1e-9
        xp = torch if isinstance(y, torch.Tensor) else np
        pos = y >= 0
        neg = y < 0
        lmbda_zero = self.lmbda == 0
        lmbda_not_zero = self.lmbda != 0
        lmbda_two = self.lmbda == 2
        lmbda_not_two = self.lmbda != 2

        x = xp.where(pos * lmbda_not_zero, (self.lmbda * y + 1)**(1 / self.lmbda) - 1, 0) + \
            xp.where(pos * lmbda_zero, xp.exp(y) - 1, 0) + \
            xp.where(neg * lmbda_not_two, -((2 - self.lmbda) * -y + 1)**(1 / (2 - self.lmbda)) + 1, 0) + \
            xp.where(neg * lmbda_two, -(xp.exp(-y) + 1), 0)
        return x


class BoxCox:
    def __init__(self, data=None, lmbda=None) -> None:
        self.λ = lmbda
        if data is not None and self.λ is None:
            self.fit(data)

    def fit(self, data):
        self.λ = stats.boxcox_normmax(data)

    def forward(self, x):
        xp = torch if isinstance(x, torch.Tensor) else np
        # x = xp.abs(x+1e-12)
        return xp.log(x) if abs(self.λ)<1e-6 else (xp.abs(x)**self.λ-1)/self.λ

    def backward(self, y):
        xp = torch if isinstance(y, torch.Tensor) else np
        # y = xp.abs(y+1e-12) 
        return xp.exp(y) if abs(self.λ)<1e-6 else (self.λ*xp.abs(y)+1)**(1/self.λ)

    def __call__(self, x):
        return self.forward(x)


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


class Uniform:
    def __init__(self, data=None, a=None, b=None):
        self.a = a
        self.b = b
        
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


class Gauss:
    def __init__(self, data=None, mu=None, std=None):
        self.mu  = mu
        self.std = std
        if data is not None and mu is None and std is None:
            self.fit(data)
            
    def fit(self, data):
        self.mu, self.std = stats.norm.fit(data)
        
    def forward(self, x):
        return (x-self.mu)/self.std
    
    def backward(self, y):
        return y*self.std+self.mu
    
    def __call__(self, x):
        return self.forward(x)


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