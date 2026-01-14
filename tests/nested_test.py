#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import ultranest
from ultranest.plot import cornerplot, PredictionBand

# -------------------------
# Repro / device
# -------------------------
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float32  # GPU compute dtype

# -------------------------
# True noisy polynomial regression problem (consistent parameterization!)
# Model: y(x) = a0 + a1 x + a2 x^2 + ... + aD x^D + eps, eps~N(0, sigma)
# Parameters sampled: [a0..aD, log_sigma]
# -------------------------
degree = 3
ndim = degree + 2
param_names = [f"a{i}" for i in range(degree + 1)] + ["log_sigma"]

true_coeffs = np.array([0.7, -1.2, 0.9, 2.5], dtype=np.float64)  # [a0,a1,a2,a3]
true_sigma = 0.2
true_log_sigma = float(np.log(true_sigma))

# Data
N = 200
x = np.linspace(-1.0, 1.0, N).astype(np.float64)

# Generate truth USING THE SAME BASIS AS THE MODEL (no np.polyval ambiguity)
y_true = sum(true_coeffs[k] * x**k for k in range(degree + 1))
y_obs = y_true + np.random.normal(0.0, true_sigma, size=N)

# Sanity check: should be ~0
y_true_check = sum(true_coeffs[k] * x**k for k in range(degree + 1))
print("max|y_true - y_true_check| =", np.max(np.abs(y_true - y_true_check)))

# Move data to GPU once
x_t = torch.from_numpy(x.astype(np.float32)).to(device=device, dtype=torch_dtype)      # (N,)
y_t = torch.from_numpy(y_obs.astype(np.float32)).to(device=device, dtype=torch_dtype)  # (N,)

# Precompute powers on GPU once: (D+1, N)
powers_t = torch.stack([x_t**k for k in range(degree + 1)], dim=0)

# -------------------------
# Priors (uniform)
# -------------------------
coeff_low, coeff_high = -5.0, 5.0
logsig_low, logsig_high = np.log(1e-3), np.log(1.0)

prior_low  = np.array([coeff_low] * (degree + 1) + [logsig_low], dtype=np.float64)
prior_high = np.array([coeff_high] * (degree + 1) + [logsig_high], dtype=np.float64)

prior_low_t  = torch.from_numpy(prior_low.astype(np.float32)).to(device=device, dtype=torch_dtype)
prior_high_t = torch.from_numpy(prior_high.astype(np.float32)).to(device=device, dtype=torch_dtype)

# -------------------------
# UltraNest plumbing: float64 numpy in/out, torch float32 on GPU inside
# -------------------------
def transform(u):
    u = np.atleast_2d(u).astype(np.float64, copy=False)  # UltraNest-safe buffers
    u_t = torch.from_numpy(u).to(device=device, dtype=torch_dtype)
    theta = prior_low_t + (prior_high_t - prior_low_t) * u_t
    return theta.detach().cpu().numpy().astype(np.float64, copy=False)

@torch.no_grad()
def loglike(theta):
    theta = np.atleast_2d(theta).astype(np.float64, copy=False)
    th = torch.from_numpy(theta).to(device=device, dtype=torch_dtype)  # (B, ndim)

    coeffs = th[:, :degree + 1]     # (B, D+1) [a0..aD]
    log_sigma = th[:, degree + 1]   # (B,)
    sigma = torch.exp(log_sigma).clamp_min(1e-6)

    # y_model = coeffs @ powers
    y_model = coeffs @ powers_t     # (B, N)
    resid = y_t[None, :] - y_model  # (B, N)

    # Gaussian log-likelihood with sigma per sample
    Nf = float(N)
    chi2 = (resid**2).sum(dim=1) / (sigma**2)
    logl = -0.5 * (chi2 + Nf * np.log(2.0*np.pi) + 2.0 * Nf * torch.log(sigma))

    return logl.detach().cpu().numpy().astype(np.float64, copy=False)

# -------------------------
# Run UltraNest
# -------------------------
sampler = ultranest.ReactiveNestedSampler(
    param_names,
    loglike,
    transform=transform,
    vectorized=True,
    ndraw_min=32,
    ndraw_max=32,      # <-- same as min
    # draw_multiple=False # <-- don’t adaptively change batch size
)

result = sampler.run(
    max_ncalls=300_000,
    show_status=True,
    viz_callback="auto",
)

sampler.print_results()
print("logZ =", result["logz"], "+/-", result["logzerr"])

# -------------------------
# Built-in diagnostics + corner plot
# -------------------------
sampler.plot_run()
sampler.plot_trace()

truths = np.concatenate([true_coeffs, [true_log_sigma]]).astype(np.float64)
fig = cornerplot(
    result,
    show_titles=True,
    bins=30,
    smooth=True,
    truths=truths,
)
plt.show()

# -------------------------
# Posterior predictive band
# -------------------------
ws = result["weighted_samples"]
points = np.asarray(ws["points"], dtype=np.float64)
weights = np.asarray(ws["weights"], dtype=np.float64)
weights = weights / weights.sum()

Npred = 512
idx = np.random.choice(len(points), size=min(Npred, len(points)), replace=True, p=weights)

theta_sel = torch.from_numpy(points[idx]).to(device=device, dtype=torch_dtype)
coeffs_sel = theta_sel[:, :degree + 1]             # (B, D+1)
Y = (coeffs_sel @ powers_t).detach().cpu().numpy() # (B, N)

band = PredictionBand(x)
for y in Y:
    band.add(y)

plt.figure(figsize=(8, 4))
band.shade(q=0.341)  # ~1σ
band.shade(q=0.477)  # ~2σ-ish
band.line()

plt.plot(x, y_true, "-", lw=2, label="true polynomial")
plt.plot(x, y_obs, ".", ms=3, alpha=0.6, label="data")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Posterior predictive band (polynomial regression)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# ---- sanity: compare loglike(true) vs loglike(posterior median) ----
ws = result["weighted_samples"]
points = np.asarray(ws["points"], dtype=np.float64)
weights = np.asarray(ws["weights"], dtype=np.float64)
weights = weights / weights.sum()

def wquantile(x, w, qs):
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cw = np.cumsum(w)
    return np.interp(qs, cw, x)

post_med = np.array([wquantile(points[:, j], weights, [0.5])[0] for j in range(points.shape[1])])
truth = np.concatenate([true_coeffs, [true_log_sigma]]).astype(np.float64)

logL_truth = loglike(truth)[0]
logL_med   = loglike(post_med)[0]
print("logL(truth)  =", logL_truth)
print("logL(median) =", logL_med)
print("ΔlogL (median - truth) =", logL_med - logL_truth)

# %%
