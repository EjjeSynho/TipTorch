#%%
import numpy as np
import matplotlib.pyplot as plt


def covariance_to_rotated_gaussian(sx, sy, sxy, syx=None):
    """
    Convert a 2D covariance matrix into principal Gaussian axes.

    sx, sy:
        Standard deviations along original x/y axes.

    sxy, syx:
        Covariance terms. For a valid covariance matrix they should be equal.
    """

    if syx is None:
        syx = sxy

    cov_xy = 0.5 * (sxy + syx)

    Sigma = np.array([
        [sx**2,  cov_xy],
        [cov_xy, sy**2]
    ])

    eigvals, eigvecs = np.linalg.eigh(Sigma)

    # Sort from largest to smallest eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    sigma_major = np.sqrt(eigvals[0])
    sigma_minor = np.sqrt(eigvals[1])

    vx, vy = eigvecs[:, 0]
    theta = np.arctan2(vy, vx)

    return sigma_major, sigma_minor, theta, Sigma


def rotated_gaussian(x, y, mux, muy, sigma_major, sigma_minor, theta, amp=1.0):
    dx = x - mux
    dy = y - muy

    c = np.cos(theta)
    s = np.sin(theta)

    # Rotate coordinates into the Gaussian principal-axis frame
    xp =  c * dx + s * dy
    yp = -s * dx + c * dy

    return amp * np.exp(
        -0.5 * ((xp / sigma_major)**2 + (yp / sigma_minor)**2)
    )


def simulate_gaussian_from_covariance(sx, sy, sxy, syx=None, 
                                      mux=0.0, muy=0.0, amp=1.0,
                                      x_range=(-10, 10), y_range=(-10, 10), 
                                      N=200):
    """
    Simulate a 2D Gaussian from covariance parameters using the full covariance matrix.
    
    Evaluates: G(x,y) = amp * exp(-0.5 * (r^T @ Sigma^-1 @ r))
    where r = [x - mux, y - muy]^T
    
    Parameters
    ----------
    sx, sy : float
        Standard deviations along original x/y axes.
    sxy, syx : float
        Covariance terms. For a valid covariance matrix they should be equal.
        If syx is None, it defaults to sxy.
    mux, muy : float, optional
        Center coordinates of the Gaussian. Default is (0, 0).
    amp : float, optional
        Amplitude of the Gaussian. Default is 1.0.
    x_range, y_range : tuple, optional
        (min, max) range for the coordinate grid. Default is (-10, 10).
    N : int, optional
        Number of grid points along each axis. Default is 200.
    
    Returns
    -------
    xx, yy : ndarray
        Coordinate grids.
    G : ndarray
        Evaluated Gaussian values on the grid.
    Sigma : ndarray
        The 2x2 covariance matrix used.
    """
    if syx is None:
        syx = sxy
    
    # Average the off-diagonal terms for a symmetric covariance matrix
    cov_xy = 0.5 * (sxy + syx)
    
    # Build the covariance matrix
    Sigma = np.array([
        [sx**2,  cov_xy],
        [cov_xy, sy**2]
    ])
    
    # Compute inverse using the analytical formula for 2x2 matrix
    det = sx**2 * sy**2 - cov_xy**2
    
    if det <= 0:
        raise ValueError("Covariance matrix is not positive definite (determinant <= 0)")
    
    # Sigma^-1 = 1/det * [[sy^2, -cov_xy], [-cov_xy, sx^2]]
    Sigma_inv_11 = sy**2 / det
    Sigma_inv_12 = -cov_xy / det
    Sigma_inv_22 = sx**2 / det
    
    # Create coordinate grid
    x = np.linspace(x_range[0], x_range[1], N)
    y = np.linspace(y_range[0], y_range[1], N)
    xx, yy = np.meshgrid(x, y)
    
    # Compute deviations from center
    dx = xx - mux
    dy = yy - muy
    
    # Compute the quadratic form: (dx, dy) @ Sigma^-1 @ (dx, dy)^T
    # = Sigma_inv_11 * dx^2 + 2 * Sigma_inv_12 * dx*dy + Sigma_inv_22 * dy^2
    quadratic_form = (Sigma_inv_11 * dx**2 + 
                      2 * Sigma_inv_12 * dx * dy + 
                      Sigma_inv_22 * dy**2)
    
    # Evaluate Gaussian
    G = amp * np.exp(-0.5 * quadratic_form)
    
    return xx, yy, G, Sigma


#%%
sx  = 3.0
sy  = 1.5
sxy = 2.0
syx = 2.0

# Simple usage with your current parameters
xx, yy, G, Sigma = simulate_gaussian_from_covariance(
    sx=3.0, sy=1.5, sxy=2.0, syx=2.0
)

print("Covariance matrix used:")
print(Sigma)

# Then plot
plt.imshow(G, extent=[xx.min(), xx.max(), yy.min(), yy.max()], origin="lower", cmap="magma")
plt.colorbar(label="Gaussian value")
plt.contour(xx, yy, G, colors="white", linewidths=0.7, alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Gaussian from covariance matrix")
plt.axis("equal")
plt.tight_layout()
plt.show()

#%%
sigma_major, sigma_minor, theta, Sigma = covariance_to_rotated_gaussian(
    sx, sy, sxy, syx
)

print("Covariance matrix:")
print(Sigma)

print(f"sigma_major = {sigma_major:.3f}")
print(f"sigma_minor = {sigma_minor:.3f}")
print(f"theta       = {theta:.3f} [rad] = {np.rad2deg(theta):.2f} [deg]")


# Make a coordinate grid
N = 200
x = np.linspace(-10, 10, N)
y = np.linspace(-10, 10, N)
xx, yy = np.meshgrid(x, y)

# Evaluate Gaussian
G_rotated = rotated_gaussian(
    xx, yy,
    mux = 0.0,
    muy = 0.0,
    sigma_major = sigma_major,
    sigma_minor = sigma_minor,
    theta = theta,
    amp = 1.0
)

# Plot
plt.figure(figsize=(6, 5))
plt.imshow(
    G_rotated,
    extent = [x.min(), x.max(), y.min(), y.max()],
    origin = "lower",
    cmap = "magma"
)
plt.colorbar(label="Gaussian value")
plt.contour(xx, yy, G_rotated, colors="white", linewidths=0.7, alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Rotated 2D Gaussian from covariance matrix")
plt.axis("equal")
plt.tight_layout()
plt.show()


#%% Example: multiply two rotated Gaussians via covariance matrices

def covariance_from_rotated_gaussian(sigma_major, sigma_minor, theta):
    """Build a 2x2 covariance matrix from principal Gaussian widths and angle."""
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    D = np.diag([sigma_major**2, sigma_minor**2])
    return R @ D @ R.T


def covariance_matrix_to_rotated_gaussian(Sigma):
    """Convert a 2x2 covariance matrix into major/minor widths and rotation angle."""
    eigvals, eigvecs = np.linalg.eigh(Sigma)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    sigma_major = np.sqrt(eigvals[0])
    sigma_minor = np.sqrt(eigvals[1])

    vx, vy = eigvecs[:, 0]
    theta = np.arctan2(vy, vx)

    return sigma_major, sigma_minor, theta


def multiply_gaussian_parameters(mu1, Sigma1, mu2, Sigma2):
    """
    Multiply two non-normalized Gaussian exponentials:

        exp[-0.5 (x - mu1)^T Sigma1^-1 (x - mu1)]
      * exp[-0.5 (x - mu2)^T Sigma2^-1 (x - mu2)]

    The result is proportional to another Gaussian with covariance Sigma and mean mu.
    """
    I = np.eye(2)
    Lambda1 = np.linalg.solve(Sigma1, I)
    Lambda2 = np.linalg.solve(Sigma2, I)

    Lambda = Lambda1 + Lambda2
    Sigma = np.linalg.solve(Lambda, I)

    eta = Lambda1 @ mu1 + Lambda2 @ mu2
    mu = np.linalg.solve(Lambda, eta)

    return mu, Sigma


def gaussian_from_covariance_grid(xx, yy, mu, Sigma, amp=1.0):
    """Evaluate amp * exp[-0.5 (x - mu)^T Sigma^-1 (x - mu)] on an existing grid."""
    Sigma_inv = np.linalg.solve(Sigma, np.eye(2))

    dx = xx - mu[0]
    dy = yy - mu[1]

    q = (
        Sigma_inv[0, 0] * dx**2 + \
        Sigma_inv[1, 1] * dy**2 + \
        2 * Sigma_inv[0, 1] * dx * dy
    )

    return amp * np.exp(-0.5 * q)


# Two input Gaussians. Their covariance ellipses have different rotations.
mu1 = np.array([-0.7, 0.35])
mu2 = np.array([0.6, -0.25])

Sigma1 = covariance_from_rotated_gaussian(
    sigma_major=3.0,
    sigma_minor=1.0,
    theta=np.deg2rad(25.0),
)

Sigma2 = covariance_from_rotated_gaussian(
    sigma_major=2.0,
    sigma_minor=0.75,
    theta=np.deg2rad(-35.0),
)

mu12, Sigma12 = multiply_gaussian_parameters(mu1, Sigma1, mu2, Sigma2)
sigma_major12, sigma_minor12, theta12 = covariance_matrix_to_rotated_gaussian(Sigma12)

print("\n--- Product Gaussian from covariance algebra ---")
print("Sigma1:")
print(Sigma1)
print("Sigma2:")
print(Sigma2)
print("Combined mean:", mu12)
print("Combined covariance:")
print(Sigma12)
print(f"combined sigma_major = {sigma_major12:.4f}")
print(f"combined sigma_minor = {sigma_minor12:.4f}")
print(f"combined theta       = {theta12:.4f} rad = {np.rad2deg(theta12):.2f} deg")

# Build a grid and compare:
#   1. direct numerical multiplication G1 * G2
#   2. Gaussian reconstructed from the analytically combined covariance/mean
N = 300
x = np.linspace(-8, 8, N)
y = np.linspace(-8, 8, N)
xx, yy = np.meshgrid(x, y)

G1 = gaussian_from_covariance_grid(xx, yy, mu1, Sigma1)
G2 = gaussian_from_covariance_grid(xx, yy, mu2, Sigma2)
G_product_numeric = G1 * G2

G_product_from_cov = rotated_gaussian(
    xx, yy,
    mux=mu12[0],
    muy=mu12[1],
    sigma_major=sigma_major12,
    sigma_minor=sigma_minor12,
    theta=theta12,
    amp=1.0,
)

# The product of two unnormalized Gaussian exponentials is proportional to the
# combined Gaussian. Normalize both peaks to compare only the shape/covariance.
G_product_numeric_norm  = G_product_numeric / G_product_numeric.max()
G_product_from_cov_norm = G_product_from_cov / G_product_from_cov.max()

abs_diff = np.abs(G_product_numeric_norm - G_product_from_cov_norm)
print(f"max abs difference after peak normalization = {abs_diff.max():.3e}")
print(f"mean abs difference after peak normalization = {abs_diff.mean():.3e}")

extent = [x.min(), x.max(), y.min(), y.max()]

plt.figure(figsize=(6, 5))
plt.imshow(G1, extent=extent, origin="lower", cmap="magma")
plt.contour(xx, yy, G1, colors="white", linewidths=0.7, alpha=0.7)
plt.title("Input Gaussian 1")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.colorbar(label="Gaussian value")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(G2, extent=extent, origin="lower", cmap="magma")
plt.contour(xx, yy, G2, colors="white", linewidths=0.7, alpha=0.7)
plt.title("Input Gaussian 2")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.colorbar(label="Gaussian value")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(G_product_numeric_norm, extent=extent, origin="lower", cmap="magma")
plt.contour(xx, yy, G_product_numeric_norm, colors="white", linewidths=0.7, alpha=0.7)
plt.title("Numerical product: G1 * G2, peak-normalized")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.colorbar(label="Normalized value")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(G_product_from_cov_norm, extent=extent, origin="lower", cmap="magma")
plt.contour(xx, yy, G_product_from_cov_norm, colors="white", linewidths=0.7, alpha=0.7)
plt.title("Product from covariance algebra, peak-normalized")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.colorbar(label="Normalized value")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(abs_diff, extent=extent, origin="lower", cmap="magma")
plt.title("Absolute difference between both product methods")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.colorbar(label="Absolute difference")
plt.tight_layout()
plt.show()
# %%


import torch


def jitter_params_to_cov(Jx, Jy, Jxy_deg):
    """
    Convert jitter parameters to covariance matrix.
    
    Parameters
    ----------
    Jx : torch.Tensor
        Principal-axis jitter sigma (major axis) in mas. Shape (...,)
    Jy : torch.Tensor
        Principal-axis jitter sigma (minor axis) in mas. Shape (...,)
    Jxy_deg : torch.Tensor
        Rotation angle in degrees. Shape (...,)
    
    All inputs must have the same batch shape (...), which can be:
    - Scalar (single jitter)
    - (N,) for N jitters
    - (N, M) or any higher dimensional batch
    
    Returns
    -------
    Sigma : torch.Tensor
        Covariance matrix of shape (..., 2, 2)
    """
    theta = torch.deg2rad(Jxy_deg)

    c = torch.cos(theta)
    s = torch.sin(theta)

    # Build rotation matrix (..., 2, 2)
    R = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1),
    ], dim=-2)

    # Build diagonal matrix with squared sigmas (..., 2, 2)
    D = torch.zeros(*Jx.shape, 2, 2, device=Jx.device, dtype=Jx.dtype)
    D[..., 0, 0] = Jx**2
    D[..., 1, 1] = Jy**2

    # Sigma = R @ D @ R^T
    return R @ D @ R.transpose(-1, -2)


def cov_to_jitter_params(Sigma):
    """
    Convert covariance matrix back to Jx, Jy, Jxy_deg.

    Returns major-axis sigma first.
    
    Parameters
    ----------
    Sigma : torch.Tensor
        Covariance matrix of shape (..., 2, 2) where ... represents
        arbitrary batch dimensions. Can be a single matrix (2, 2),
        a batch (N, 2, 2), or higher dimensional batches.
    
    Returns
    -------
    Jx : torch.Tensor
        Major axis sigma, shape (...,)
    Jy : torch.Tensor
        Minor axis sigma, shape (...,)
    Jxy_deg : torch.Tensor
        Rotation angle in degrees, shape (...,)
    """
    # Ensure input is at least 2D (single covariance matrix case)
    input_shape = Sigma.shape
    if Sigma.ndim == 2:
        Sigma = Sigma.unsqueeze(0)
    
    # Compute eigendecomposition (batched operation)
    eigvals, eigvecs = torch.linalg.eigh(Sigma)

    # Sort eigenvalues and eigenvectors in descending order
    order = torch.argsort(eigvals, dim=-1, descending=True)

    # Reorder eigenvalues
    eigvals = torch.gather(eigvals, -1, order)

    # Reorder eigenvectors (columns)
    # order shape: (..., 2) -> expand to (..., 2, 2) for gathering along dim=-1
    batch_shape = Sigma.shape[:-2]
    index = order[..., None, :].expand(*batch_shape, 2, 2)
    eigvecs = torch.gather(eigvecs, -1, index)

    # Extract principal axis widths
    Jx = torch.sqrt(torch.clamp(eigvals[..., 0], min=0.0))
    Jy = torch.sqrt(torch.clamp(eigvals[..., 1], min=0.0))

    # Extract rotation angle from the major axis eigenvector
    vx = eigvecs[..., 0, 0]
    vy = eigvecs[..., 1, 0]
    Jxy_deg = torch.rad2deg(torch.atan2(vy, vx))

    # Squeeze back if input was a single matrix
    if len(input_shape) == 2:
        Jx = Jx.squeeze(0)
        Jy = Jy.squeeze(0)
        Jxy_deg = Jxy_deg.squeeze(0)

    return Jx, Jy, Jxy_deg


def combine_zero_centered_jitters(Jx1, Jy1, Jxy1, Jx2, Jy2, Jxy2):
    """
    Combine two zero-centered jitter distributions by adding their covariance matrices.
    
    Parameters
    ----------
    Jx1, Jy1, Jxy1 : torch.Tensor
        First jitter parameters (major sigma, minor sigma, rotation angle in degrees).
        Each has shape (...,) where ... represents arbitrary batch dimensions.
    Jx2, Jy2, Jxy2 : torch.Tensor
        Second jitter parameters with the same batch shape as the first.
    
    Returns
    -------
    Jx_total, Jy_total, Jxy_total : torch.Tensor
        Combined jitter parameters, each with shape (...,)
    
    Examples
    --------
    # Single jitter combination
    Jx, Jy, Jxy = combine_zero_centered_jitters(
        torch.tensor(4.0), torch.tensor(2.0), torch.tensor(20.0),
        torch.tensor(3.0), torch.tensor(1.0), torch.tensor(-35.0)
    )
    
    # Batch of N jitter combinations
    Jx, Jy, Jxy = combine_zero_centered_jitters(
        Jx1_batch,  # shape (N,)
        Jy1_batch,  # shape (N,)
        Jxy1_batch, # shape (N,)
        Jx2_batch,  # shape (N,)
        Jy2_batch,  # shape (N,)
        Jxy2_batch  # shape (N,)
    )  # Returns shapes (N,), (N,), (N,)
    """
    Sigma1 = jitter_params_to_cov(Jx1, Jy1, Jxy1)
    Sigma2 = jitter_params_to_cov(Jx2, Jy2, Jxy2)

    Sigma_total = Sigma1 + Sigma2

    return cov_to_jitter_params(Sigma_total)



Jx_total, Jy_total, Jxy_total = combine_zero_centered_jitters(
    Jx1=torch.tensor(4.0),
    Jy1=torch.tensor(2.0),
    Jxy1=torch.tensor(20.0),

    Jx2=torch.tensor(3.0),
    Jy2=torch.tensor(1.0),
    Jxy2=torch.tensor(-35.0),
)

print(Jx_total, Jy_total, Jxy_total)


#%%

import torch


def scale_jitter_to_mas(Jx, Jy, telescope_diameter):
    """
    Scale jitter ellipse axes like the original ellipsesFromCovMats function.

    The original code did:

        scale = mas_to_rad * D / (4e-9)
        Jx_scaled = Jx / scale
        Jy_scaled = Jy / scale

    which is equivalent to:

        Jx_scaled = Jx * (4e-9 / D) / mas_to_rad

    Parameters
    ----------
    Jx, Jy : torch.Tensor
        Major/minor jitter sigmas in the internal covariance units.
        If the old interpretation is correct, these are RMS TT coefficients in nm.

    telescope_diameter : float or torch.Tensor
        Telescope diameter in meters.

    Returns
    -------
    Jx_mas, Jy_mas : torch.Tensor
        Major/minor jitter sigmas in mas.
    """

    mas_to_rad = torch.pi / (180.0 * 3600.0 * 1000.0)

    scale = mas_to_rad * telescope_diameter / (4.0e-9)

    Jx_mas = Jx / scale
    Jy_mas = Jy / scale

    return Jx_mas, Jy_mas


#%%

def cov_to_scaled_jitter_params(Sigma, telescope_diameter):
    """
    Convert 2x2 TT covariance matrix to scaled jitter ellipse parameters.

    Parameters
    ----------
    Sigma : torch.Tensor
        Covariance matrix of shape (..., 2, 2).

    telescope_diameter : float or torch.Tensor
        Telescope diameter in meters.

    Returns
    -------
    Jx_mas : torch.Tensor
        Major-axis jitter sigma in mas.

    Jy_mas : torch.Tensor
        Minor-axis jitter sigma in mas.

    Jxy_deg : torch.Tensor
        Rotation angle in degrees.
    """

    Jx, Jy, Jxy_deg = cov_to_jitter_params(Sigma)

    Jx_mas, Jy_mas = scale_jitter_to_mas(
        Jx,
        Jy,
        telescope_diameter=telescope_diameter,
    )

    return Jx_mas, Jy_mas, Jxy_deg

sigmaToFWHM = 2 * np.sqrt(2.0 * np.log(2.0))
