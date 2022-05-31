#%%
from inspect import Parameter
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from time import time

cuda  = torch.device('cuda') # Default CUDA device
#cuda  = torch.device('cpu') # Default CUDA device

sqr = lambda A: A[0]*x**2 + A[1]*x + A[2]
#sqr_jac = lambda A: 2*A[0]*x + A[1]

A_0 = torch.zeros([3,1], device=cuda)
A_1 = torch.tensor([[1.],[2.],[3.]], device=cuda)
x = torch.linspace(-5.0, 5.0, 100, device=cuda)


#%%
from torchimize.functions import lsq_gna, lsq_lma, lsq_gna_parallel

loss_fn = nn.L1Loss()
def func(p):
    return loss_fn(sqr(p), sqr(A_1)).unsqueeze(0)

# single gauss-newton
#coeffs_list = lsq_gna(A_0, func, max_iter=1000)
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()
coeffs_list = lsq_gna_parallel(p=A_1, function=func, max_iter=200)
end.record()
torch.cuda.synchronize()
print('GPU',int(start.elapsed_time(end)),'ms')

plt.plot(sqr(A_1).cpu())
plt.plot(sqr(coeffs_list[-1]).cpu())
plt.show()

#%%
from test_lsq_lma import test_lsq_lma

from torchmin import minimize

N = 100
xx, yy = torch.meshgrid(torch.linspace(-N, N-1, N*2), torch.linspace(-N, N-1, N*2))
def Gauss2D(X):
    return X[0]*torch.exp( -(xx-X[1])**2/(2*X[3]**2) - (yy-X[2])**2/(2*X[4]**2) )
X0 = torch.tensor([2., 10., -20., 20., 14.], requires_grad=False)
X1 = torch.tensor([1., 0.,    0., 5.,  5. ])

loss_fn = nn.L1Loss(reduction='sum')
def func(p):
    return loss_fn(Gauss2D(p), Gauss2D(X0)).unsqueeze(0)
coeffs_list = [X1]


#coeffs_list = test_lsq_lma(X1, func, max_iter=100)

result = minimize(func, X1, method='newton-exact')

print(result)

#plt.imshow(Gauss2D(X0))
#plt.show()
#plt.imshow(Gauss2D(coeffs_list[-1]))
#plt.show()
#
#ref = Gauss2D(X0)

#%%

from torchimize.optimizer.gna_opt import GNA
from torch.nn import Module, Parameter

class Gauss(Module):

    def __init__(self)-> None: #, device=None, dtype=None) -> None:
        #factory_kwargs = {'device': device, 'dtype': dtype}
        super(Gauss, self).__init__()
        #if in_params is None:
        self.A  = Parameter(torch.Tensor([1.0]))
        self.x  = Parameter(torch.Tensor([0.0]))
        self.y  = Parameter(torch.Tensor([0.0]))
        self.sx = Parameter(torch.Tensor([1.0]))
        self.sy = Parameter(torch.Tensor([1.0]))
        #else:
        #    self.params = Parameter(in_params)

    def forward(self, xx, yy):
        return self.A*torch.exp( -(xx-self.x)**2/(2*self.sx**2) - (yy-self.y)**2/(2*self.sy**2) )

model2 = Gauss()

N = 100
xx, yy = torch.meshgrid(torch.linspace(-N, N-1, N*2), torch.linspace(-N, N-1, N*2))

#plt.imshow(model2(xx,yy).detach().cpu())
#plt.show()

#%%

#%%
# parallel gauss-newton for several optimization problems at multiple costs
from torchimize.functions import lsq_gna_parallel
coeffs_list = lsq_gna_parallel(
                    p = initials_batch,
                    function = multi_cost_fun_batch,
                    jac_function = multi_jac_fun_batch,
                    args = (other_args,),
                    wvec = torch.ones(5, device='cuda', dtype=initials_batch.dtype),
                    ftol = 1e-8,
                    ptol = 1e-8,
                    gtol = 1e-8,
                    l = 1.,
                    max_iter = 80,
                )

# parallel levenberg-marquardt for several optimization problems at multiple costs
from torchimize.functions import lsq_lma_parallel
coeffs_list = lsq_lma_parallel(
                    p = initials_batch,
                    function = multi_cost_fun_batch,
                    jac_function = multi_jac_fun_batch,
                    args = (other_args,),
                    wvec = torch.ones(5, device='cuda', dtype=initials_batch.dtype),
                    ftol = 1e-8,
                    ptol = 1e-8,
                    gtol = 1e-8,
                    meth = 'marq',
                    max_iter = 40,
                )

# validate that your provided functions return correct tensor dimensionality
from torchimize.functions import test_fun_dims_parallel
ret = test_fun_dims_parallel(
    p = initials_batch,
    function = multi_cost_fun_batch,
    jac_function = multi_jac_fun_batch,
    args = (other_args,),
    wvec = torch.ones(5, device='cuda', dtype=initials_batch.dtype),
)
