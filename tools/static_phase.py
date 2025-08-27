from scipy.ndimage import label
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import torch
from scipy.io import loadmat
from utils import mask_circle, rad2mas, pdims
from project_settings import default_torch_type, DATA_FOLDER

decompose_WF = lambda WF, basis, pupil: WF[:, pupil > 0] @ basis[:, pupil > 0].T / pupil.sum()
project_WF   = lambda WF, basis, pupil: torch.einsum('mn,nwh->mwh', decompose_WF(WF, basis, pupil), basis)
calc_WFE     = lambda WF, pupil: WF[:, pupil > 0].std()


def separate_islands(binary_image):
    # Label each connected component (each "island") with a unique integer
    labeled_image, num_features = label(binary_image)
    separated_images = []

    # Iterate over each unique label (each island)
    for i in range(1, num_features + 1):
        # Create an image with only the current island
        island_image = (labeled_image == i).astype(int)
        separated_images.append(island_image)

    return separated_images


def BuildPTTBasis(pupil, pytorch=True):
    tip, tilt = np.meshgrid( np.linspace(-1, 1, pupil.shape[-2]), np.linspace(-1, 1, pupil.shape[-1]) )

    tip  = pupil * tip  / np.std( tip [np.where(pupil > 0)] )
    tilt = pupil * tilt / np.std( tilt[np.where(pupil > 0)] )

    PTT_basis = np.stack([pupil, tip, tilt], axis=0)
    
    if pytorch:
        PTT_basis = torch.tensor(PTT_basis)
                
    return PTT_basis


def BuildPetalBasis(segmented_pupil, pytorch=True):
    petals = np.stack( separate_islands(segmented_pupil) )
    x, y = np.meshgrid(np.arange(segmented_pupil.shape[-1]), np.arange(segmented_pupil.shape[-2]))

    tilt = (x[None, ...]*petals).astype(np.float64)
    tip  = (y[None, ...]*petals).astype(np.float64)

    tilt /= tilt.sum(axis=0)[np.where(segmented_pupil)].std()
    tip  /= tip.sum(axis=0) [np.where(segmented_pupil)].std()

    def normalize_TT(x):
        for i in range(petals.shape[0]):
            x[i,...] = (x[i,...] - x[i,...][np.where(petals[i,...])].mean()) * petals[i,...]
        return x

    tilt, tip = normalize_TT(tilt), normalize_TT(tip)
    coefs = [1.]*petals.shape[0] + [0.]*petals.shape[0] + [0.]*petals.shape[0]
    
    basis = np.vstack([petals, tilt, tip])
    
    basis_flatten = basis[:, segmented_pupil > 0]
    modes_STD = np.sqrt( np.diag(basis_flatten @ basis_flatten.T / segmented_pupil.sum()) )
    basis /= modes_STD[:, None, None]
    
    if not pytorch:
        return basis, np.array(coefs)
    else:
        return torch.from_numpy( basis ), torch.tensor(coefs)


def decouple_PTT_from_LWE(LWE_coefs, LWE_basis, pupil_border_offset=7):
    """
    Decouple LWE modes from the PTT modes (Piston, Tip, Tilt)
    While piston is removed, tip and tilt are translated into the pixel shifts
    """	
    pupil = LWE_basis.model.pupil
    PTT_basis = BuildPTTBasis(pupil.cpu().numpy(), True).to(LWE_basis.model.device).float()

    TT_max = PTT_basis.abs()[1,...].max().item()
    
    pixel_shift = lambda coef: 2 * TT_max * rad2mas * 1e-9 * coef / LWE_basis.model.psInMas / LWE_basis.model.D  / (1-pupil_border_offset/pupil.shape[-1])

    LWE_OPD   = torch.einsum('mn,nwh->mwh', LWE_coefs, LWE_basis.modal_basis)
    PPT_OPD   = project_WF  (LWE_OPD, PTT_basis, pupil)
    PTT_coefs = decompose_WF(LWE_OPD, PTT_basis, pupil)

    LWE_OPD_subtracted   = LWE_OPD - PPT_OPD
    LWE_coefs_subtracted = decompose_WF(LWE_OPD_subtracted, LWE_basis.modal_basis, pupil)
    
    shift_x = pixel_shift(PTT_coefs[:, 2]) # [pix], no, it's not a bug, tilt stays for the X-axis
    shift_y = pixel_shift(PTT_coefs[:, 1]) # [pix]
    
    return LWE_coefs_subtracted, LWE_OPD_subtracted, shift_x, shift_y


class ZernikeModes:
    def __init__(self, pupil, modes_num=1):
        self.nModes = modes_num
        self.modesFullRes = None
        self.pupil = pupil


    def zernikeRadialFunc(self, n, m, r):
        """
        Function to calculate the Zernike radial function

        Parameters:
            n (int): Zernike radial order
            m (int): Zernike azimuthal order
            r (ndarray): 2-d array of radii from the center of the array

        Returns:
            ndarray: The Zernike radial function
        """
        from math import factorial

        R = np.zeros(r.shape)
        # Can cast the below to "int", n,m are always *both* either even or odd
        for i in range(0, int((n-m)/2) + 1):
            R += np.array(r**(n - 2 * i) * (((-1)**(i)) *
                            factorial(n-i)) / (factorial(i) *
                            factorial(int(0.5 * (n+m) - i)) *
                            factorial(int(0.5 * (n-m) - i))),
                            dtype='float')
        return R
   

    def zernIndex(self, j):
        n = int((-1.0 + np.sqrt(8*(j-1)+1))/2.)
        p = (j-(n*(n+1))/2.)
        k = n % 2
        m = int((p+k)/2.)*2 - k

        if m != 0:
            if j % 2 == 0: s = 1
            else:  s = -1
            m *= s

        return [n, m]


    def __rotate_coordinates(self, angle, X, Y):
            angle_rad = np.radians(angle)

            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad),  np.cos(angle_rad)]
            ])

            coordinates = np.vstack((X, Y))
            rotated_coordinates = np.dot(rotation_matrix, coordinates)
            rotated_X, rotated_Y = rotated_coordinates[0, :], rotated_coordinates[1, :]

            return rotated_X, rotated_Y
        

    def computeZernike(self, angle=None, transposed=False):

        resolution = self.pupil.shape[-1]

        X, Y = np.where(self.pupil == 1)
        X = (X-resolution//2+0.5*(1-resolution%2)) / resolution
        Y = (Y-resolution//2+0.5*(1-resolution%2)) / resolution
        
        if transposed:
            X, Y = Y, X
        
        if angle is not None and angle != 0.0:
            X, Y = self.__rotate_coordinates(angle, X, Y)
        
        R = np.sqrt(X**2 + Y**2)
        R /= R.max()
        theta = np.arctan2(Y, X)

        self.modesFullRes = np.zeros([resolution**2, self.nModes])

        for i in range(1, self.nModes+1):
            n, m = self.zernIndex(i+1)
            if m == 0:
                Z = np.sqrt(n+1) * self.zernikeRadialFunc(n, 0, R)
            else:
                if m > 0: # j is even
                    Z = np.sqrt(2*(n+1)) * self.zernikeRadialFunc(n, m, R) * np.cos(m*theta)
                else:   #i is odd
                    m = abs(m)
                    Z = np.sqrt(2*(n+1)) * self.zernikeRadialFunc(n, m, R) * np.sin(m*theta)
            
            Z -= Z.mean()
            Z /= np.std(Z)

            self.modesFullRes[np.where(np.reshape(self.pupil, resolution*resolution)>0), i-1] = Z
            
        self.modesFullRes = np.reshape( self.modesFullRes, [resolution, resolution, self.nModes] )


class PhaseMap:
    """
    Base class for any OPD→complex-field mapping.
    Subclasses must implement compute_OPD(x) to return
    an OPD tensor in meters of shape (N_src, N_wvl, H, W).
    """

    def __init__(self, model, ignore_pupil=True):
        self.model = model
        self.ignore_pupil = ignore_pupil

    def compute_OPD(self, x):
        raise NotImplementedError("Subclass must implement compute_OPD")

    def forward(self, x):
        # 1) Build OPD map in [m]
        OPD = self.compute_OPD(x)

        # 2) Wave-number factor k = 2jπ / λ
        k = 2j * torch.pi / self.model.wvl.view(self.model.N_src, self.model.N_wvl, 1, 1)

        # 3) Phase term
        phase = torch.exp(k * OPD)

        # 4) Optionally multiply through by pupil × apodizer
        if self.ignore_pupil:
            return phase
        
        pupil_apod = pdims(self.model.pupil * self.model.apodizer, -2) if self.model.apodizer is not None else pdims(self.model.pupil, -2)
        return pupil_apod * phase

    __call__ = forward


class LWEBasis(PhaseMap):
    def __init__(self, model, ignore_pupil=False):
        super().__init__(model, ignore_pupil)
        # build petal (modal) basis once
        modal_basis, _ = BuildPetalBasis(model.pupil.cpu(), pytorch=True)
        self.modal_basis = modal_basis.float().to(model.device)

    def compute_OPD(self, coeffs):
        # coeffs shape: (N_src, N_wvl, n_modes)
        # modal_basis shape: (n_modes, H, W)
        # result: (N_src, N_wvl, H, W)
        return torch.einsum('om,mhw->ohw', coeffs, self.modal_basis) * 1e-9


class PixelmapBasis(PhaseMap):
    def __init__(self, model, ignore_pupil=True):
        super().__init__(model, ignore_pupil)

    def fft_upscale(self, x):
        h, w = x.shape[-2:]
        H, W = self.model.pupil.shape[-2:]
        X = torch.fft.fftshift(torch.fft.fft2(x))
        pad_top = (H - h) // 2
        pad_bot = H - h - pad_top
        pad_left = (W - w) // 2
        pad_right = W - w - pad_left
        Xp = torch.pad(X, (pad_left, pad_right, pad_top, pad_bot), "constant", 0)
        up = torch.fft.ifft2(torch.fft.ifftshift(Xp)) * (H * W / (h * w))
        return up.abs().unsqueeze(1)

    def interp_upscale(self, x, mode="bilinear"):
        return F.interpolate(
            x.unsqueeze(1),
            size=(self.model.pupil.shape[-2], self.model.pupil.shape[-1]),
            mode=mode,
            align_corners=True,
        )

    def compute_OPD(self, x):
        return self.interp_upscale(x) * 1e-9 # x: coeffs in the low-res space


class ZernikeBasis(PhaseMap):
    def __init__(self, model, N_modes, ignore_pupil=True):
        super().__init__(model, ignore_pupil)
       
        # build a Zernike basis
        if ignore_pupil:
            pupil_mask = mask_circle(N=model.pupil.shape[-1], r=model.pupil.shape[-1] // 2)
        else:
            pupil_mask = model.pupil.squeeze().cpu().numpy()

        Z = ZernikeModes(pupil_mask, N_modes)
        Z.computeZernike(angle=self.model.pupil_angle)
        # store as (N_modes, H, W)
        modes = torch.as_tensor(Z.modesFullRes, device=model.device, dtype=default_torch_type)
        self.zernike_basis = modes.permute(2, 0, 1)
        self.N_modes = N_modes

    def compute_OPD(self, coefs):
        # coefs: (N_src, N_wvl, N_modes)
        # zernike_basis: (N_modes, H, W)
        return torch.einsum('om,mhw->ohw', coefs, self.zernike_basis) * 1e-9


class MUSEPhaseBump(PhaseMap):
    def __init__(self, model, ignore_pupil=False):
        super().__init__(model, ignore_pupil=ignore_pupil)

        # load calibration map and rescale to meters
        mat = loadmat(DATA_FOLDER / "calibrations/VLT_CALIBRATION/MUSE_slopes/DelatRS2Phase.mat")["aa"]

        OPD = torch.tensor(mat).float().to(model.device)
        OPD *= 35 * 2 / 1.6 / 1e6  # [m]
        
        # up-sample to pupil resolution
        OPD_upsampled = F.interpolate(
            OPD[None, None, ...],
            size=model.pupil.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        
        # rotate during initialization
        angle = model.pupil_angle - 45.0
        self.OPD_map = TF.rotate(OPD_upsampled.unsqueeze(0), angle, interpolation=TF.InterpolationMode.BILINEAR).squeeze(0)

    def compute_OPD(self, coef):
        # coef: (N_src, N_wvl)
        # make full batch of maps
        batch_map = self.OPD_map.expand(self.model.N_src, self.model.N_wvl, -1, -1)
        # OPD = map * coef
        return batch_map * coef.view(self.model.N_src, self.model.N_wvl, 1, 1)
    

class ArbitraryBasis(PhaseMap):
    def __init__(self, model, basis, ignore_pupil=True):
        """
        Initialize with an arbitrary basis.
        
        Args:
            model: The model object
            basis: Tensor of shape (N_modes, H, W) containing the basis functions
            ignore_pupil: Whether to ignore pupil mask
        """
        super().__init__(model, ignore_pupil)
        
        # Store the basis, ensuring it's on the correct device and dtype
        self.basis = basis.to(model.device).float()
        self.N_modes = basis.shape[0]
        
    def orthogonalize_modes(self):
        """
        Orthogonalize the basis modes using Gram-Schmidt process.
        Only operates on pixels within the pupil if not ignoring pupil.
        """
        if not self.ignore_pupil:
            pupil_mask = self.model.pupil.squeeze() > 0
            # Extract pixels within pupil for each mode
            basis_masked = self.basis[:, pupil_mask]
            
            # Gram-Schmidt orthogonalization
            orthogonal_basis = torch.zeros_like(basis_masked)
            for i in range(self.N_modes):
                # Start with current mode
                v = basis_masked[i].clone()
                
                # Subtract projections onto previous orthogonal modes
                for j in range(i):
                    proj = torch.dot(v, orthogonal_basis[j]) / torch.dot(orthogonal_basis[j], orthogonal_basis[j])
                    v -= proj * orthogonal_basis[j]
                
                # Normalize
                orthogonal_basis[i] = v / torch.norm(v)
            
            # Reconstruct full basis
            self.basis[:, pupil_mask] = orthogonal_basis
            self.basis[:, ~pupil_mask] = 0
        else:
            # Flatten spatial dimensions for full array orthogonalization
            basis_flat = self.basis.view(self.N_modes, -1)
            
            # Gram-Schmidt orthogonalization
            for i in range(self.N_modes):
                v = basis_flat[i].clone()
                
                for j in range(i):
                    proj = torch.dot(v, basis_flat[j]) / torch.dot(basis_flat[j], basis_flat[j])
                    v -= proj * basis_flat[j]
                
                basis_flat[i] = v / torch.norm(v)
            
            # Reshape back
            self.basis = basis_flat.view(self.N_modes, *self.basis.shape[1:])

    def compute_OPD(self, coefs):
        """
        Compute OPD from coefficients and basis.
        
        Args:
            coefs: Tensor of shape (N_src, N_wvl, N_modes)
            
        Returns:
            OPD tensor of shape (N_src, N_wvl, H, W) in meters
        """
        # coefs: (N_src, N_wvl, N_modes)
        # basis: (N_modes, H, W)
        return torch.einsum('om,mhw->ohw', coefs, self.basis) * 1e-9
