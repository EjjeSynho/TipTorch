#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tools.plotting import plot_radial_profiles, draw_PSF_stack
from tools.utils import mask_circle
from data_processing.MUSE_STD_dataset_utils import LoadSTDStarData
import matplotlib as mpl

from project_settings import device
from torchmin import minimize
from data_processing.normalizers import Uniform, Atanh#, InputsCompressor, InputsManager
from managers.input_manager import InputsManager
from tqdm import tqdm
from tools.utils import PupilVLT, GradientLoss
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

derotate_PSF    = True
Moffat_absorber = True
fit_LO          = True
LO_pixels_modes = 75

#%%
# with open(MUSE_DATA_FOLDER+'/muse_df.pickle', 'rb') as handle:
#     muse_df = pickle.load(handle)

# wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 5), a_min=0, a_max=N_wvl_max-1)

wvl_ids = np.clip(np.arange(0, (N_wvl_max:=30)+1, 3), a_min=0, a_max=N_wvl_max-1)

PSF_0, norms, bgs, model_config = LoadSTDStarData(
    # ids = [394],
    # ids = [296],
    # ids = [324],
    # ids = [230],
    # ids = [230],
    # ids = [231],
    ids = [470],
    # ids = [462],
    # ids = [465],
    derotate_PSF = derotate_PSF,
    normalize = True,
    subtract_background = True,
    wvl_ids = wvl_ids,
    ensure_odd_pixels = True,
    device = device
)

if derotate_PSF:
    pupil_angle = 0.0
else:
    pupil_angle = model_config['telescope']['PupilAngle'].cpu().numpy().item()

N_wvl = PSF_0.shape[1]
wavelengths = model_config['sources_science']['Wavelength'].squeeze()
# GL_frac = np.minimum(config['atmosphere']['Cn2Weights'][0,0].item(), 0.98)

#%
cmap = mpl.colormaps.get_cmap('gray')  # viridis is the default colormap for imshow
cmap.set_bad(color='black')

for i in range(N_wvl):
    im = PSF_0[0,i,...].cpu().numpy()
    vmin = np.percentile(im[im > 0], 10) if np.any(im > 0) else 1
    vmax = np.percentile(im[im > 0], 99.975) if np.any(im > 0) else im.max()

    plt.imshow(im, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    # plt.show()


#%% Initialize the model
from PSF_models.TipTorch import TipTorch
from tools.static_phase import ArbitraryBasis, PixelmapBasis, ZernikeBasis, MUSEPhaseBump

pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=pupil_angle), device=device )
PSD_include = {
    'fitting':         True,
    'WFS noise':       True,
    'spatio-temporal': True,
    'aliasing':        False,
    'chromatism':      True,
    'diff. refract':   True,
    'Moffat':          Moffat_absorber
}
PSF_model = TipTorch(model_config, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1)
PSF_model.apodizer = PSF_model.make_tensor(1.0)

# LO_basis = PixelmapBasis(PSF_model, ignore_pupil=False)
Z_basis = ZernikeBasis(PSF_model, N_modes=LO_pixels_modes, ignore_pupil=False)
sausage_basis = MUSEPhaseBump(PSF_model, ignore_pupil=False)

#%%
# LO NCPAs + phase bump optimized jointly
composite_basis = torch.concat([
    sausage_basis.OPD_map.unsqueeze(0).flip(-2)*5e6,
    Z_basis.zernike_basis[2:10,...]
], dim=0)

LO_basis = ArbitraryBasis(PSF_model, composite_basis, ignore_pupil=False)
LO_pixels_modes = LO_basis.N_modes

# for i in range(LO_basis.N_modes):
#     plt.imshow(LO_basis.basis[i,...].cpu().numpy())
#     plt.title(f"LO Mode {i+1}")
#     plt.colorbar()
#     plt.show()

#%% Initialize inputs manager and set up parameters
inputs_manager = InputsManager()

# Initialize normalizers/transforms
norm_F           = Uniform(a=0.0,   b=1.0)
norm_bg          = Uniform(a=-5e-6, b=5e-6)
norm_r0          = Uniform(a=0,     b=1)
norm_dxy         = Uniform(a=-1,    b=1)
norm_J           = Uniform(a=0,     b=50)
norm_Jxy         = Uniform(a=-180,  b=180)
norm_dn          = Uniform(a=0,     b=5)
norm_amp         = Uniform(a=0,     b=10)
norm_b           = Uniform(a=0,     b=0.1)
norm_alpha       = Uniform(a=-1,    b=10)
norm_beta        = Uniform(a=0,     b=2)
norm_ratio       = Uniform(a=0,     b=2)
norm_theta       = Uniform(a=-np.pi/2, b=np.pi/2)
norm_wind_speed  = Uniform(a=0, b=10)
norm_sausage_pow = Uniform(a=0, b=1)
norm_LO          = Uniform(a=-100, b=100)
norm_wvl         = Uniform(a=wavelengths.min().item(), b=wavelengths.max().item())
# df_transforms_GL_frac = Atanh()

# Add base parameters
inputs_manager.add('r0',  torch.tensor([PSF_model.r0.item()]), norm_r0)
inputs_manager.add('F',   torch.tensor([[1.0,]*N_wvl]),  norm_F)
inputs_manager.add('dx',  torch.tensor([[0.0,]*N_wvl]),  norm_dxy)
inputs_manager.add('dy',  torch.tensor([[0.0,]*N_wvl]),  norm_dxy)
inputs_manager.add('bg',  torch.tensor([[0.0,]*N_wvl]),  norm_bg)
inputs_manager.add('dn',  torch.tensor([0.25]),          norm_dn)
inputs_manager.add('Jx',  torch.tensor([[2.5,]*N_wvl]),  norm_J)
inputs_manager.add('Jy',  torch.tensor([[2.5,]*N_wvl]),  norm_J)
inputs_manager.add('Jxy', torch.tensor([[0.0]]),         norm_Jxy)
# inputs_manager.add('wind_speed', PSF_model.wind_speed.detach().clone(),  norm_wind_speed)
# inputs_manager.add('GL_frac', torch.tensor([GL_frac]), df_transforms_GL_frac)

# Add Moffat parameters if needed
if Moffat_absorber:
    inputs_manager.add('amp',   torch.tensor([0.0]), norm_amp)
    inputs_manager.add('b',     torch.tensor([0.0]), norm_b)
    inputs_manager.add('alpha', torch.tensor([4.5]), norm_alpha)
    inputs_manager.add('beta',  torch.tensor([2.5]), norm_beta)
    inputs_manager.add('ratio', torch.tensor([1.0]), norm_ratio)
    inputs_manager.add('theta', torch.tensor([0.0]), norm_theta)

if fit_LO:
    if isinstance(LO_basis, PixelmapBasis):
        inputs_manager.add('LO_coefs', torch.zeros([1, LO_pixels_modes**2]), norm_LO)
    elif isinstance(LO_basis, ZernikeBasis) or isinstance(LO_basis, ArbitraryBasis):
        inputs_manager.add('LO_coefs', torch.zeros([1, LO_pixels_modes]), norm_LO)
    else:
        raise ValueError('Wrong LO type specified.')

inputs_manager.to(device)
inputs_manager.to_float()

_ = inputs_manager.stack()


        # # Ensure control points are sorted (soft constraint)
        # t_sorted = torch.sort(self.t_control)[0]
        
        # # Create spline with current control points
        # coeffs = natural_cubic_spline_coeffs(t_sorted, self.y_control)
        # spline = NaturalCubicSpline(coeffs)
        
        # # Evaluate at desired points
        # y_pred = spline.evaluate(x_eval).squeeze(-1)

#%%
def func(x_, include_list=None):
    x_torch = inputs_manager.unstack(x_)
    
    if fit_LO:
        if isinstance(LO_basis, PixelmapBasis):
            phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(1, LO_pixels_modes, LO_pixels_modes))
        elif isinstance(LO_basis, ZernikeBasis) or isinstance(LO_basis, ArbitraryBasis):
            phase_func = lambda: LO_basis(inputs_manager["LO_coefs"].view(1, LO_pixels_modes))
        else:
            raise ValueError('Wrong LO type specified.')
    else:
        phase_func = None

    if include_list is not None:
        return PSF_model({ key: x_torch[key] for key in include_list }, None, phase_generator=phase_func)
    else:
        return PSF_model(x_torch, None, phase_generator=phase_func)


wvl_weights = torch.linspace(1.0, 0.5, N_wvl).to(device).view(1, N_wvl, 1, 1)
wvl_weights = N_wvl / wvl_weights.sum() * wvl_weights # Normalize so that the total energy is preserved

mask = torch.tensor(mask_circle(PSF_0.shape[-1], 5)).view(1, 1, *PSF_0.shape[-2:]).to(device)
mask_inv = 1.0 - mask

grad_loss_fn = GradientLoss(p=1, reduction='mean')

def loss_fn1(x_):
    diff1 = (func(x_)-PSF_0) * wvl_weights
    mse_loss = (diff1 * 4000).pow(2).mean()
    mae_loss = (diff1 * 32000).abs().mean()
    if fit_LO:  
        if isinstance(LO_basis, PixelmapBasis):
            LO_loss = grad_loss_fn(inputs_manager['LO_coefs'].view(1, 1, LO_pixels_modes, LO_pixels_modes)) * 5e-5
        elif isinstance(LO_basis, ZernikeBasis) or isinstance(LO_basis, ArbitraryBasis):
            LO_loss = inputs_manager['LO_coefs'].abs().sum()**2 * 1e-7
            first_coef_penalty = torch.clamp(-inputs_manager['LO_coefs'][0, 0], min=0).pow(2) * 5e-5
            LO_loss += first_coef_penalty
            # phi = LO_basis.compute_OPD(inputs_manager["LO_coefs"].view(1, LO_pixels_modes)).squeeze()
            # phi_rot = torch.flip(phi, dims=(-2,-1)) # 180° rotation around array center
            # phi_even = 0.5 * (phi + phi_rot)
            # eta = 5e5
            # LO_loss += eta * (phi_even).pow(2).sum() / phi.pow(2).sum().clamp_min(1e-12)          
        else:
            raise ValueError('Wrong LO type specified.')
    else:
        LO_loss = 0.0
    return mse_loss + mae_loss + LO_loss


def loss_fn2(x_):
    diff1 = (func(x_)-PSF_0) * wvl_weights
    mse_loss = diff1.pow(2).sum() / PSF_0.shape[-2] / PSF_0.shape[-1] * 1250
    mae_loss = diff1.abs().sum()  / PSF_0.shape[-2] / PSF_0.shape[-1] * 2500
    if fit_LO:  
        if isinstance(LO_basis, PixelmapBasis):
            LO_loss = grad_loss_fn(inputs_manager['LO_coefs'].view(1, 1, LO_pixels_modes, LO_pixels_modes)) * 5e-5
        elif isinstance(LO_basis, ZernikeBasis) or isinstance(LO_basis, ArbitraryBasis):
            LO_loss = inputs_manager['LO_coefs'].abs().sum()**2 * 1e-7
            # Constraint to enforce first element of LO_coefs to be positive
            first_coef_penalty = torch.clamp(-inputs_manager['LO_coefs'][0, 0], min=0).pow(2) * 5e-5
            LO_loss += first_coef_penalty
            # phi = LO_basis.compute_OPD(inputs_manager["LO_coefs"].view(1, LO_pixels_modes)).squeeze()
            # phi_rot = torch.flip(phi, dims=(-2,-1)) # 180° rotation around array center
            # phi_even = 0.5 * (phi + phi_rot)
            # eta = 5e5
            # LO_loss += eta * (phi_even).pow(2).sum() / phi.pow(2).sum().clamp_min(1e-12)          
        else:
            raise ValueError('Wrong LO type specified.')
    else:
        LO_loss = 0.0
    
    return mse_loss + mae_loss + LO_loss


def loss_fn3(x_):
    diff1 = (func(x_ )-PSF_0) * wvl_weights
    mse_loss = (diff1 * 1500).pow(2).sum()
    mae_loss = (diff1 * 2500).abs().sum()
    return (mse_loss + mae_loss) / PSF_0.shape[-2] / PSF_0.shape[-1]


def loss_fn4(x_):
    diff1 = (func(x_ )-PSF_0) * wvl_weights
    sqrt_loss = (diff1 * 1500).abs().sqrt().sum()
    return sqrt_loss / PSF_0.shape[-2] / PSF_0.shape[-1]

#%%
def minimize_params(loss_fn, include_list, exclude_list, max_iter, verbose=True):
    if len(include_list) > 0:
        inputs_manager.set_optimizable(include_list, True)
    else:
        print('include_list is empty')
        
    inputs_manager.set_optimizable(exclude_list, False)

    print(inputs_manager)

    result = minimize(loss_fn, inputs_manager.stack(), max_iter=max_iter, tol=1e-4, method='l-bfgs', disp= 2 if verbose else 0)
    if fit_LO:  
        if isinstance(LO_basis, PixelmapBasis):
            OPD_map = inputs_manager['LO_coefs'].view(1, LO_pixels_modes, LO_pixels_modes).squeeze().detach().cpu().numpy()
        elif isinstance(LO_basis, ZernikeBasis) or isinstance(LO_basis, ArbitraryBasis):
            OPD_map = LO_basis.compute_OPD(inputs_manager["LO_coefs"].view(1, LO_pixels_modes)).squeeze().detach().cpu().numpy()
    else:
        OPD_map = None

    return result.x, func(result.x), OPD_map


                #   ['wind_speed'] + \
include_general = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy'] + \
                  (['amp', 'alpha', 'beta'] if Moffat_absorber else []) + \
                  (['LO_coefs'] if fit_LO else [])
                  
exclude_general = ['ratio', 'theta', 'b'] if Moffat_absorber else []

include_LO = ['LO_coefs']

exclude_LO = ['r0', 'F', 'dx', 'dy', 'bg', 'dn', 'Jx', 'Jy', 'Jxy'] + \
             ['ratio', 'theta', 'b', 'amp', 'alpha', 'beta'] if Moffat_absorber else []


#%%
x0, PSF_1, OPD_map = minimize_params(loss_fn1, include_general, exclude_general, 200)

if fit_LO:
    x1, PSF_1, OPD_map = minimize_params(loss_fn2, include_LO, exclude_LO, 100)
    x2, PSF_1, OPD_map = minimize_params(loss_fn1, include_general, exclude_general, 100)

if fit_LO:
    plt.imshow(OPD_map*1e9)
    plt.colorbar()
    plt.show()

#%%
from tools.plotting import plot_radial_profiles

wvl_select = np.s_[0, N_wvl//2, -1]

vmin = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 10)
vmax = np.percentile(PSF_0[PSF_0 > 0].cpu().numpy(), 99.995)

draw_PSF_stack(
    PSF_0.cpu().numpy()[0, wvl_select, ...],
    PSF_1.cpu().numpy()[0, wvl_select, ...],
    average=True,
    min_val=vmin,
    max_val=vmax,
    crop=100
)

PSF_disp = lambda x, w: (x[0,w,...]).cpu().numpy()

fig, ax = plt.subplots(1, len(wvl_select), figsize=(10, len(wvl_select)))
for i, lmbd in enumerate(wvl_select):
    plot_radial_profiles( PSF_disp(PSF_0, lmbd),  PSF_disp(PSF_1, lmbd),  'Data', 'TipTorch', cutoff=40,  y_min=3e-2, linthresh=1e-2, ax=ax[i] )
plt.show()

#%%


a = inputs_manager['dy'].squeeze().cpu().numpy()

plt.plot(wavelengths.squeeze(0).cpu().numpy()*1e9, a)


#%%
def bspline_basis(lams, knots, degree=3):
    """
    Return design matrix Phi of shape [B, N] for B-splines of given degree.
    knots: 1D tensor of non-decreasing knot positions with padding at ends.
    """
    # co-locate dtype/device
    lams  = torch.as_tensor(lams)
    knots = torch.as_tensor(knots, dtype=lams.dtype, device=lams.device)
    lams  = lams.to(knots.device).to(knots.dtype)[..., None]  # [B, 1]

    m = knots.numel()                  # number of knots
    # ---- degree 0: one basis per interval [t_i, t_{i+1}) => m-1 functions
    K0 = m - 1
    B0 = []
    for i in range(K0):
        left, right = knots[i], knots[i+1]
        B0.append(((lams >= left) & (lams < right)).to(lams.dtype))
    B = torch.cat(B0, dim=-1)          # [B, m-1]

    # optional: include right boundary so sum of bases = 1 at last knot
    at_right = (lams.squeeze(-1) == knots[-1])
    if at_right.any():
        B[at_right, :] = 0
        B[at_right, -1] = 1.0

    # ---- Cox–de Boor recursion
    for p in range(1, degree + 1):
        K_new = m - p - 1              # shrinks by one each step
        B_next = torch.zeros(B.shape[0], K_new, dtype=B.dtype, device=B.device)
        for i in range(K_new):
            den1 = (knots[i+p]   - knots[i]).item()
            den2 = (knots[i+p+1] - knots[i+1]).item()

            w1 = 0.0 if den1 == 0.0 else (lams - knots[i]) / den1      # [B,1]
            w2 = 0.0 if den2 == 0.0 else (knots[i+p+1] - lams) / den2  # [B,1]

            # keep column dims to avoid [B,B] broadcasting
            B_next[:, i:i+1] = w1 * B[:, i:i+1] + w2 * B[:, i+1:i+2]
        B = B_next

    return B  # [B, m-degree-1]

# --- Example usage ---
degree = 3
knots = torch.tensor([0, 0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1], dtype=torch.float32)
wavelengths = torch.linspace(0, 1, 100)

basis_matrix = bspline_basis(wavelengths, knots, degree)
coefficients = torch.randn(basis_matrix.shape[-1])

spline_values = basis_matrix @ coefficients

plt.figure(figsize=(10, 6))
plt.plot(wavelengths.numpy(), spline_values.numpy(), linewidth=2, label='B-spline')
plt.xlabel('Wavelength')
plt.ylabel('Spline Value')
plt.title('B-spline Interpolation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# %%
length, channels = 7, 3
t = torch.linspace(0, 1, length)
x = torch.rand(length, channels)
coeffs = natural_cubic_spline_coeffs(t, x)
spline = NaturalCubicSpline(coeffs)
point = torch.tensor(0.4)
out = spline.evaluate(point)

#%%
x_range = torch.linspace(0, 1, 100)
y_range = spline.evaluate(x_range)

plt.figure(figsize=(10, 6))
for i in range(channels):
    plt.plot(x_range.numpy(), y_range[:, i].numpy(), linewidth=2, label=f'Channel {i+1}')
    plt.scatter(t.numpy(), x[:, i].numpy(), s=50, alpha=0.7)
plt.xlabel('t')
plt.ylabel('Value')
plt.title('Natural Cubic Spline Interpolation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

#%%
class NonLinearSplineFitter(nn.Module):
    """
    Non-linear optimization based spline fitting using learnable control points.
    """
    def __init__(self, x_data, y_data, n_control_points=10):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.n_control_points = n_control_points
        
        # Initialize control point positions uniformly along x range
        x_min, x_max = x_data.min(), x_data.max()
        t_init = torch.linspace(x_min, x_max, n_control_points)
        
        # Initialize control point values by interpolating from data
        y_init = torch.zeros(n_control_points)
        for i, t_val in enumerate(t_init):
            # Find closest data point
            closest_idx = torch.argmin(torch.abs(x_data - t_val))
            y_init[i] = y_data[closest_idx]
        
        # Learnable parameters
        self.t_control = nn.Parameter(t_init)
        self.y_control = nn.Parameter(y_init.unsqueeze(-1))  # Add channel dimension
        
        # Keep endpoints fixed to avoid extrapolation issues
        self.register_buffer('t_fixed_mask', torch.tensor([True] + [False]*(n_control_points-2) + [True]))
    
    def forward(self, x_eval=None):
        """
        Evaluate spline at given points or at data points.
        """
        if x_eval is None:
            x_eval = self.x_data
            
        # Ensure control points are sorted (soft constraint)
        t_sorted = torch.sort(self.t_control)[0]
        
        # Create spline with current control points
        coeffs = natural_cubic_spline_coeffs(t_sorted, self.y_control)
        spline = NaturalCubicSpline(coeffs)
        
        # Evaluate at desired points
        y_pred = spline.evaluate(x_eval).squeeze(-1)
        return y_pred
    
    def get_spline_object(self):
        """Return the current spline object for evaluation."""
        t_sorted = torch.sort(self.t_control)[0]
        coeffs = natural_cubic_spline_coeffs(t_sorted, self.y_control)
        return NaturalCubicSpline(coeffs)

def fit_nonlinear_spline(x_data, y_data, n_control_points=10, lr=0.01, epochs=1000, 
                        regularization=1e-4, fix_endpoints=True, plot=True, verbose=True):
    """
    Fits a spline using non-linear optimization of control points.
    
    Args:
        x_data: torch.Tensor of x coordinates
        y_data: torch.Tensor of y coordinates
        n_control_points: number of control points to optimize
        lr: learning rate for optimizer
        epochs: number of optimization epochs
        regularization: L2 regularization strength for smoothness
        fix_endpoints: whether to fix first and last control points
        plot: whether to plot the results
        verbose: whether to print progress
    
    Returns:
        model: trained NonLinearSplineFitter
        spline: final spline object
        losses: list of loss values during training
    """
    # Ensure data is sorted by x
    sorted_indices = torch.argsort(x_data)
    x_sorted = x_data[sorted_indices]
    y_sorted = y_data[sorted_indices]
    
    # Create model
    model = NonLinearSplineFitter(x_sorted, y_sorted, n_control_points)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss function with regularization
    def loss_fn():
        y_pred = model()
        # Data fitting loss
        mse_loss = torch.mean((y_pred - y_sorted)**2)
        # Smoothness regularization (penalize large second differences)
        t_sorted = torch.sort(model.t_control)[0]
        y_ctrl = model.y_control.squeeze()
        if len(y_ctrl) > 2:
            second_diff = y_ctrl[2:] - 2*y_ctrl[1:-1] + y_ctrl[:-2]
            smooth_loss = regularization * torch.mean(second_diff**2)
        else:
            smooth_loss = 0.0
        
        # Monotonicity constraint for t_control (soft)
        t_diff = model.t_control[1:] - model.t_control[:-1]
        monotonic_loss = torch.sum(torch.clamp(-t_diff, min=0)**2) * 10.0
        
        return mse_loss + smooth_loss + monotonic_loss
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        
        # Optional: fix endpoints
        if fix_endpoints:
            with torch.no_grad():
                model.t_control.grad[0] = 0
                model.t_control.grad[-1] = 0
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and (epoch + 1) % (epochs // 10) == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # Get final spline
    with torch.no_grad():
        spline = model.get_spline_object()
        
        if plot:
            # Evaluation points for smooth curve
            t_eval = torch.linspace(x_sorted[0], x_sorted[-1], 200)
            y_eval = model(t_eval)
            
            plt.figure(figsize=(15, 10))
            
            # Main plot
            plt.subplot(2, 2, 1)
            plt.plot(x_data.numpy(), y_data.numpy(), 'o', alpha=0.5, label='Original data', markersize=3)
            t_ctrl_sorted = torch.sort(model.t_control)[0]
            y_ctrl_sorted = model.y_control.squeeze()[torch.argsort(model.t_control)]
            
            plt.plot(t_ctrl_sorted.detach().numpy(), y_ctrl_sorted.detach().numpy(), 's', 
                    markersize=8, label=f'Optimized control points ({n_control_points})')
            
            plt.plot(t_eval.numpy(), y_eval.detach().numpy(), '-', linewidth=2, label='Fitted spline')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Non-linear Spline Fitting')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Loss curve
            plt.subplot(2, 2, 2)
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Residuals
            plt.subplot(2, 2, 3)
            y_pred_data = model()
            residuals = (y_pred_data - y_sorted).detach().numpy()
            plt.plot(x_sorted.numpy(), residuals, 'o', alpha=0.6, markersize=2)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('x')
            plt.ylabel('Residual')
            plt.title('Fitting Residuals')
            plt.grid(True, alpha=0.3)
            
            # Control points evolution (if we tracked them)
            plt.subplot(2, 2, 4)
            plt.plot(t_ctrl_sorted.detach().numpy(), y_ctrl_sorted.detach().numpy(), 'o-')
            plt.xlabel('Control Point Position')
            plt.ylabel('Control Point Value')
            plt.title('Optimized Control Points')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    return model, spline, losses

# Example usage with the same data
torch.manual_seed(42)
n_points = 1000
x_random = torch.linspace(0, 2*np.pi, n_points)
y_random = torch.sin(x_random) + 0.3*torch.randn(n_points) + 0.1*torch.sin(5*x_random)

# Fit spline with different numbers of control points using non-linear optimization
model, spline, losses = fit_nonlinear_spline(
    x_random, y_random, 
    n_control_points=50,
    lr=0.01,
    epochs=500,
    regularization=1e-4,
    verbose=False
)

# Calculate final fitting error
with torch.no_grad():
    y_pred = model()
    mse = torch.mean((y_pred - y_random)**2)
    print(f"Final MSE: {mse:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")

# %%
