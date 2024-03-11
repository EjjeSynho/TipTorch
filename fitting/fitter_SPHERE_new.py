#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import pickle
import torch
import numpy as np
from torch import nn, optim
from tools.utils import SR, pdims
from tools.utils import cropper, FitGauss2D, LWE_basis, EarlyStopping
from PSF_models.TipToy_SPHERE_multisrc import TipTorch
from data_processing.SPHERE_preproc_utils import SPHERE_preprocess, SamplesByIds
from data_processing.normalizers import TransformSequence, Uniform, InputsTransformer
from tools.config_manager import GetSPHEREonsky
from project_globals import SPHERE_DATA_FOLDER, device
from torch.autograd.functional import hessian

from project_globals import SPHERE_DATA_FOLDER, SPHERE_FITTING_FOLDER, device

#%% Initialize data sample
with open(SPHERE_DATA_FOLDER+'sphere_df.pickle', 'rb') as handle:
    psf_df = pickle.load(handle)

psf_df = psf_df[psf_df['Corrupted'] == False]
# psf_df = psf_df[psf_df['Low quality'] == False]
# psf_df = psf_df[psf_df['Medium quality'] == False]
# psf_df = psf_df[psf_df['LWE'] == True]
# psf_df = psf_df[psf_df['mag R'] < 7]
# psf_df = psf_df[psf_df['Num. DITs'] < 50]
# psf_df = psf_df[psf_df['Class A'] == True]
# psf_df = psf_df[np.isfinite(psf_df['λ left (nm)']) < 1700]
# psf_df = psf_df[psf_df['Δλ left (nm)'] < 80]

good_ids = psf_df.index.values.tolist()

norm_F   = TransformSequence(transforms=[ Uniform(a=0.0,   b=1.0) ])
norm_bg  = TransformSequence(transforms=[ Uniform(a=-1e-5, b=1e-5)])
norm_r0  = TransformSequence(transforms=[ Uniform(a=0,     b=0.5) ])
norm_dxy = TransformSequence(transforms=[ Uniform(a=-1,    b=1)   ])
norm_J   = TransformSequence(transforms=[ Uniform(a=0,     b=30)  ])
norm_Jxy = TransformSequence(transforms=[ Uniform(a=0,     b=50)  ])
norm_LWE = TransformSequence(transforms=[ Uniform(a=-200,  b=200) ])


def to_store(x):
    if x is not None:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            return x
    else:
        return None


def gauss_fitter(PSF_stack):
    FWHMs = np.zeros([PSF_stack.shape[0], PSF_stack.shape[1], 2])
    for i in range(PSF_stack.shape[0]):
        for l in range(PSF_stack.shape[1]):
            f_x, f_y = FitGauss2D(PSF_stack[i,l,:,:].float())
            FWHMs[i,l,0] = f_x.item()
            FWHMs[i,l,1] = f_y.item()
    return FWHMs


#%%
def load_and_fit_sample(id):
    try:
        PSF_data, _, merged_config = SPHERE_preprocess(
            sample_ids    = [id],
            norm_regime   = 'sum',
            split_cube    = False,
            PSF_loader    = lambda x: SamplesByIds(x, synth=False),
            config_loader = GetSPHEREonsky,
            framework     = 'pytorch',
            device        = device)

        PSF_0   = PSF_data[0]['PSF (mean)'].unsqueeze(0)
        # PSF_var = PSF_data[0]['PSF (var)'].unsqueeze(0)
        bg      = PSF_data[0]['bg (mean)']
        norms   = PSF_data[0]['norm (mean)']
        del PSF_data
        
    except Exception as e:
        print(e)
        print('Failed to load sample', id)
        return None

    try:
        toy = TipTorch(merged_config, None, device)

        _ = toy()
        toy.optimizables = []
        _ = toy({ 'bg': bg.unsqueeze(0).to(device) })

        PSF_DL = toy.DLPSF()

        basis = LWE_basis(toy)

        transformer = InputsTransformer({
            'F':   norm_F,   'bg':  norm_bg,
            'r0':  norm_r0,  'dx':  norm_dxy,
            'dy':  norm_dxy, 'Jx':  norm_J,
            'Jy':  norm_J,   'Jxy': norm_Jxy,
            'basis_coefs': norm_LWE
        })

        inp_dict = {
            'r0':  toy.r0, 'F':   toy.F,
            'dx':  toy.dx, 'dy':  toy.dy,
            'bg':  toy.bg, 'Jx':  toy.Jx,
            'Jy':  toy.Jy, 'Jxy': toy.Jxy,
            'basis_coefs': basis.coefs
        }

        _ = transformer.stack(inp_dict) # to create index mapping

        x0 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.1] + [0,]*3 + [0,]*8

        x0 = torch.tensor(x0).float().to(device).unsqueeze(0)
        x0.requires_grad = True

        if basis.coefs.requires_grad:
            buf = basis.coefs.detach().clone()
            basis.coefs = buf

        optimizer = optim.LBFGS([x0], lr=10, history_size=20, max_iter=4, line_search_fn="strong_wolfe")
        crop_all = cropper(PSF_0, 100)

        def loss_fn(A, B):
            return nn.L1Loss(reduction='sum')(A[crop_all], B[crop_all])

        def func(x_):
            x_torch = transformer.destack(x_)
            return toy(x_torch, None, lambda: basis(x_torch['basis_coefs'].float()))

        early_stopping = EarlyStopping(patience=2, tolerance=1e-4, relative=False)

        for i in range(100):
            optimizer.zero_grad()

            loss = loss_fn(func(x0), PSF_0)

            if np.isnan(loss.item()):
                break
            
            early_stopping(loss)

            loss.backward(retain_graph=True)
            optimizer.step( lambda: loss_fn(func(x0), PSF_0) )

            print('Loss:', loss.item(), end="\r", flush=True)

            if early_stopping.stop:
                print('Stopped at it.', i, 'with loss:', loss.item())
                break

        # decomposed_variables = transformer.destack(x0)

        PSF_1 = func(x0)
    except Exception as e:
        print(e)
        print('Failed to fit sample', id)
        return None

    hessian_mat  = None
    hessian_mat_ = None
    hessian_inv  = None
    variance_estimates = None
    L = None
    V = None

    try:
        hessian_mat = hessian(lambda x_: loss_fn(func(x_), PSF_0).log(), x0).squeeze()
        hessian_mat_ = hessian_mat.clone()
        hessian_mat_ = hessian_mat_.nan_to_num()
        hessian_mat_[1:,0] = hessian_mat_[0,1:]
        hessian_mat_[hessian_mat.abs() < 1e-11] = 1e-11
    except Exception as e:
        pass
    else:
        try:
            hessian_inv = torch.inverse(hessian_mat_)
            variance_estimates = torch.diag(hessian_inv).abs()

            L_complex, V_complex = torch.linalg.eig(hessian_mat_)

            L = L_complex.real.cpu().numpy()
            V = V_complex.real.cpu().numpy()
            V = V / np.linalg.norm(V, axis=0)
        except Exception as e:
            pass

    save_data = {
        'comments':    'All-normalized, new approach',
        'config':      merged_config,
        'bg':          to_store(bg),
        'F':           to_store(toy.F),
        'dx':          to_store(toy.dx),
        'dy':          to_store(toy.dy),
        'r0':          to_store(toy.r0),
        'n':           to_store(toy.NoiseVariance(toy.r0.abs())),
        'dn':          to_store(toy.dn),
        'Jx':          to_store(toy.Jx),
        'Jy':          to_store(toy.Jy),
        'Jxy':         to_store(toy.Jxy),
        'Nph WFS':     to_store(toy.WFS_Nph),
        'LWE coefs':   to_store(basis.coefs),
        'Hessian':     to_store(hessian_mat_),
        'Variances':   to_store(variance_estimates),
        'Inv.H eigvs': to_store(L),
        'Inv.H eigvs': to_store(V),
        'SR data':     SR(PSF_0, PSF_DL).detach().cpu().numpy(),
        'SR fit':      SR(PSF_1, PSF_DL).detach().cpu().numpy(),
        'FWHM fit':    gauss_fitter(PSF_0), 
        'FWHM data':   gauss_fitter(PSF_1),
        'Img. data':   to_store(PSF_0*pdims(norms,2)),
        'Img. fit':    to_store(PSF_1*pdims(norms,2)),
        'PSD':         to_store(toy.PSD),
        'Data norms':  to_store(norms),
        'Model norms': to_store(toy.norm_scale),
        'loss':        loss_fn(PSF_1, PSF_0).item()
    }
    return save_data


# test = load_and_fit_sample(1660)

#%%
for id in good_ids:
    filename = SPHERE_FITTING_FOLDER + str(id) + '.pickle'
    
    save_data = load_and_fit_sample(id)
    if save_data is None:
        with open(filename, 'wb') as handle:
            pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    