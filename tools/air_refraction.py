#%%
import torch
import torch.nn as nn


class AirRefractiveIndexCalculator(nn.Module):
    """
    Fast PyTorch implementation of the humid-air refractive index model.

    Based on the logic in the uploaded NumPy version:
    - Ciddor 1996 for wavelengths < 1.3 um
    - Mathar 2006 for wavelengths >= 1.3 um

    Returns n - 1.
    """

    def __init__(self, dtype=torch.float64, device=None):
        super().__init__()

        self.default_dtype = dtype
        if device is None:
            device = torch.device("cpu")
        self.default_device = torch.device(device)

        # Reference environmental parameters from Mathar (2006), Eq. 7
        self.register_buffer("T_ref", torch.tensor(273.15 + 17.5, dtype=dtype, device=device))
        self.register_buffer("p_ref", torch.tensor(75000.0, dtype=dtype, device=device))
        self.register_buffer("H_ref", torch.tensor(10.0, dtype=dtype, device=device))

        # --- Band metadata ---
        band_names = ["K", "L", "M", "N", "Q"]
        band_ranges = torch.tensor([
            [1.3,  2.5],
            [2.8,  4.2],
            [4.35, 5.3],
            [7.5, 14.1],
            [16.0, 28.0],
        ], dtype=dtype, device=device)
        
        band_nu_ref = torch.tensor([
            10000.0 / 2.25,
            10000.0 / 3.4,
            10000.0 / 4.8,
            10000.0 / 10.1,
            10000.0 / 20.0,
        ], dtype=dtype, device=device)

        self.band_names = band_names
        self.register_buffer("band_ranges", band_ranges)
        self.register_buffer("band_nu_ref", band_nu_ref)

        # coeffs shape: (num_bands=5, poly_order=6, state_terms=10)
        coeffs = torch.empty((5, 6, 10), dtype=dtype, device=device)

        coeffs[0] = torch.tensor([  # K
            [ 0.200192e-3,   0.588625e-1,  -3.01513,      -0.103945e-7,   0.573256e-12,  0.267085e-8,   0.609186e-17,  0.497859e-4,   0.779176e-6,  -0.206567e-15],
            [ 0.113474e-9,  -0.385766e-7,   0.406167e-3,   0.136858e-11,  0.186367e-16,  0.135941e-14,  0.519024e-23, -0.661752e-8,   0.396499e-12,  0.106141e-20],
            [-0.424595e-14,  0.888019e-10, -0.514544e-6,  -0.171039e-14, -0.228150e-19,  0.135295e-18, -0.419477e-27,  0.832034e-11,  0.395114e-16, -0.149982e-23],
            [ 0.100957e-16, -0.567650e-13,  0.343161e-9,   0.112908e-17,  0.150947e-22,  0.818218e-23,  0.434120e-30, -0.551793e-14,  0.233587e-20,  0.984046e-27],
            [-0.293315e-20,  0.166615e-16, -0.101189e-12, -0.329925e-21, -0.441214e-26, -0.222957e-26, -0.122445e-33,  0.161899e-17, -0.636441e-24, -0.288266e-30],
            [ 0.307228e-24, -0.174845e-20,  0.106749e-16,  0.344747e-25,  0.461209e-30,  0.249964e-30,  0.134816e-37, -0.169901e-21,  0.716868e-28,  0.299105e-34]
        ], dtype=dtype, device=device)

        coeffs[1] = torch.tensor([  # L
            [ 0.200049e-3,   0.588431e-1,  -3.13579,     -0.108142e-7,   0.586812e-12,  0.266900e-8,   0.608860e-17,  0.517962e-4,   0.778638e-6,  -0.217243e-15],
            [ 0.145221e-9,  -0.825182e-7,   0.694124e-3,  0.230102e-11,  0.312198e-16,  0.168162e-14, -0.112149e-21,  0.461560e-22,  0.446396e-12,  0.104747e-20],
            [ 0.250951e-12,  0.137982e-9,  -0.500604e-6, -0.154652e-14, -0.197792e-19,  0.353075e-17,  0.184282e-24,  0.776507e-11,  0.784600e-15, -0.523689e-23],
            [-0.745834e-15,  0.352420e-13, -0.116668e-8, -0.323014e-17, -0.461945e-22, -0.963455e-20, -0.524471e-27,  0.172569e-13, -0.195151e-17,  0.817386e-26],
            [-0.161432e-17, -0.730651e-15,  0.209644e-11, 0.630616e-20,  0.788398e-25, -0.223079e-22, -0.121299e-29, -0.320582e-16, -0.542083e-20,  0.309913e-28],
            [ 0.352780e-20, -0.167911e-18,  0.591037e-14, 0.173880e-22,  0.245580e-27,  0.453166e-25,  0.246512e-32, -0.899435e-19,  0.103530e-22, -0.363491e-31]
        ], dtype=dtype, device=device)

        coeffs[2] = torch.tensor([  # M
            [ 0.200020e-3,   0.590035e-1,  -4.09830,     -0.140463e-7,   0.543605e-12,  0.266898e-8,   0.610706e-17,  0.674488e-4,  0.778627e-6,  -0.211676e-15],
            [ 0.275346e-9,  -0.375764e-6,   0.250037e-2,  0.839350e-11,  0.112802e-15,  0.273629e-14,  0.116620e-21, -0.406775e-7,  0.593296e-12,  0.487921e-20],
            [ 0.325702e-12,  0.134585e-9,   0.275187e-6, -0.190929e-14, -0.229979e-19,  0.463466e-17,  0.244736e-24,  0.289063e-11, 0.145042e-14, -0.682545e-23],
            [-0.693603e-14,  0.124316e-11, -0.653398e-8, -0.121399e-16, -0.191450e-21, -0.916894e-19, -0.497682e-26,  0.819898e-13, 0.489815e-17,  0.942802e-25],
            [ 0.285610e-17,  0.508510e-13, -0.310589e-9, -0.898863e-18, -0.120352e-22,  0.136685e-21,  0.742024e-29,  0.468386e-14, 0.327941e-19, -0.946422e-27],
            [ 0.338758e-18, -0.189245e-15,  0.127747e-11, 0.364662e-20,  0.500955e-25,  0.413687e-23,  0.224625e-30, -0.191182e-16, 0.128020e-21, -0.153682e-29]
        ], dtype=dtype, device=device)

        coeffs[3] = torch.tensor([  # N
            [ 0.199885e-3,   0.593900e-1,  -6.50355,      -0.221938e-7,   0.393524e-12,  0.266809e-8,   0.610508e-17,  0.106776e-3,   0.778368e-6,  -0.206365e-15],
            [ 0.344739e-9,  -0.172226e-5,   0.103830e-1,   0.347377e-10,  0.464083e-15,  0.3695247e-14, 0.227694e-22, -0.168516e-6,   0.216404e-12,  0.300234e-19],
            [-0.273714e-12,  0.237654e-8,  -0.139464e-4,  -0.465991e-13, -0.621764e-18,  0.159070e-17,  0.786323e-25,  0.226201e-9,   0.581805e-15, -0.426519e-22],
            [ 0.393383e-15, -0.381812e-11,  0.220077e-7,   0.735848e-16,  0.981126e-21, -0.303451e-20, -0.174448e-27, -0.356457e-12, -0.189618e-17,  0.684306e-25],
            [-0.569488e-17,  0.305050e-14, -0.272412e-10, -0.897119e-19, -0.121384e-23, -0.661489e-22, -0.359791e-29,  0.437980e-15, -0.198869e-19, -0.467320e-29],
            [ 0.164556e-19, -0.157464e-16,  0.126364e-12,  0.380817e-21,  0.515111e-26,  0.178226e-24,  0.978307e-32, -0.194545e-17,  0.589381e-22,  0.126117e-30]
        ], dtype=dtype, device=device)

        coeffs[4] = torch.tensor([  # Q
            [ 0.199436e-3,   0.621723e-1, -23.2409,      -0.772707e-7,  -0.326604e-12,  0.266827e-8,   0.613675e-17,  0.375974e-3,   0.778436e-6,  -0.272614e-15],
            [ 0.299123e-8,  -0.177074e-4,   0.108557,     0.347237e-9,   0.463606e-14,  0.120788e-14,  0.585494e-22, -0.171849e-5,   0.461840e-12,  0.304662e-18],
            [-0.214862e-10,  0.152213e-6,  -0.102439e-2, -0.272675e-11, -0.364272e-16,  0.522646e-17,  0.286055e-24,  0.146704e-7,   0.306229e-14, -0.239590e-20],
            [ 0.143338e-12, -0.954584e-9,   0.634072e-5,  0.170858e-13,  0.228756e-18,  0.783027e-19,  0.425193e-26, -0.917231e-10, -0.623183e-16,  0.149285e-22],
            [ 0.122398e-14,  0.921476e-13, -0.675587e-9, -0.150004e-17, -0.200547e-22,  0.753235e-21,  0.413455e-28, -0.955922e-12, -0.161119e-18,  0.136086e-24],
            [-0.114628e-16, -0.996706e-11,  0.762517e-7,  0.156889e-15,  0.209502e-20, -0.228819e-24, -0.812941e-32,  0.880502e-14,  0.800756e-20, -0.130999e-26]
        ], dtype=dtype, device=device)

        self.register_buffer("coeffs", coeffs)
        self.register_buffer("poly_powers", torch.arange(6, dtype=dtype, device=device)) # powers 0...5 for polynomial evaluation


    def _as_tensor(self, x, *, dtype=None, device=None):
        if dtype is None:
            dtype = self.default_dtype
        if device is None:
            device = self.default_device
        return torch.as_tensor(x, dtype=dtype, device=device)


    def get_refractivity(self, wvl_m, T_celsius=15.0, P_pa=101325.0, H_pct=20.0):
        """
        Compute n - 1 for wavelengths in meters.

        Inputs may be Python scalars, NumPy arrays, or torch tensors.
        All inputs are broadcastable.
        """
        wvl_m = self._as_tensor(wvl_m)
        T_celsius = self._as_tensor(T_celsius, dtype=wvl_m.dtype, device=wvl_m.device)
        P_pa  = self._as_tensor(P_pa, dtype=wvl_m.dtype, device=wvl_m.device)
        H_pct = self._as_tensor(H_pct, dtype=wvl_m.dtype, device=wvl_m.device)

        scalar_input = (wvl_m.ndim == 0)
        if scalar_input:
            wvl_m = wvl_m.unsqueeze(0)

        # Broadcast all environmental parameters to wavelength shape
        T_celsius = torch.broadcast_to(T_celsius, wvl_m.shape)
        P_pa      = torch.broadcast_to(P_pa, wvl_m.shape)
        H_pct     = torch.broadcast_to(H_pct, wvl_m.shape)

        wvl_um = wvl_m * 1e6
        nu = 10000.0 / wvl_um
        T_kelvin = T_celsius + 273.15

        n_minus_1 = torch.zeros_like(wvl_um)

        # ------------------------------------------------------------------
        # CIDDOR branch: w < 1.3 um
        # ------------------------------------------------------------------
        mask_c = wvl_um < 1.3
        if mask_c.any():
            w   = wvl_um[mask_c]
            T_k = T_kelvin[mask_c]
            T_c = T_celsius[mask_c]
            P   = P_pa[mask_c]
            H   = H_pct[mask_c]

            sigma = 1.0 / w  # um^-1

            k0 = 238.0185
            k1 = 5792105.0
            k2 = 57.362
            k3 = 167917.0
            n_as_minus_1 = (k1 / (k0 - sigma**2) + k3 / (k2 - sigma**2)) * 1e-8
            n_axs_minus_1 = n_as_minus_1

            cf = 1.022
            w0 = 295.235
            w1 = 2.6422
            w2 = -0.032380
            w3 = 0.004028
            n_ws_minus_1 = cf * (w0 + w1 * sigma**2 + w2 * sigma**4 + w3 * sigma**6) * 1e-8

            A_svp = 1.2378847e-5
            B_svp = -1.9121316e-2
            C_svp = 33.93711047
            D_svp = -6.3431645e3
            svp = torch.exp(A_svp * T_k**2 + B_svp * T_k + C_svp + D_svp / T_k)

            alpha = 1.00062
            beta  = 3.14e-8
            gamma = 5.6e-7
            f_enh = alpha + beta * P + gamma * (T_c**2)

            P_w = (H / 100.0) * f_enh * svp
            P_a = P - P_w

            rho_a_ratio = (P_a / 101325.0) * (288.15 / T_k)
            rho_w_ratio = (P_w / 1333.0)   * (293.15 / T_k)

            n_minus_1[mask_c] = rho_a_ratio * n_axs_minus_1 + rho_w_ratio * n_ws_minus_1

        # ------------------------------------------------------------------
        # MATHAR branch: w >= 1.3 um
        # ------------------------------------------------------------------
        mask_m = ~mask_c
        
        if mask_m.any():
            w = wvl_um[mask_m]
            nu_m = nu[mask_m]
            T_k = T_kelvin[mask_m]
            P = P_pa[mask_m]
            H = H_pct[mask_m]

            dT_inv = (1.0 / T_k) - (1.0 / self.T_ref)
            dP = P - self.p_ref
            dH = H - self.H_ref

            # shape: (N, 10)
            state = torch.stack([
                torch.ones_like(dT_inv),
                dT_inv,
                dT_inv**2,
                dH,
                dH**2,
                dP,
                dP**2,
                dT_inv * dH,
                dT_inv * dP,
                dH * dP,
            ], dim=-1)

            out = torch.zeros_like(w)

            # default fallback band = K, as in original code
            assigned = torch.zeros_like(w, dtype=torch.bool)

            for band_idx in range(self.band_ranges.shape[0]):
                lo = self.band_ranges[band_idx, 0]
                hi = self.band_ranges[band_idx, 1]
                m = (w >= lo) & (w <= hi)
                if not m.any():
                    continue

                assigned |= m

                state_b = state[m]                # (Nb, 10)
                coeffs_b = self.coeffs[band_idx]  # (6, 10)
                c_i = state_b @ coeffs_b.T        # (Nb, 6)

                d_nu = nu_m[m] - self.band_nu_ref[band_idx]     # (Nb,)
                powers = d_nu.unsqueeze(-1) ** self.poly_powers # (Nb, 6)

                out[m] = torch.sum(c_i * powers, dim=-1)

            # fallback to K for wavelengths not in any explicit Mathar interval
            if (~assigned).any():
                m = ~assigned
                state_b = state[m]
                coeffs_b = self.coeffs[0]  # K fallback
                c_i = state_b @ coeffs_b.T
                d_nu = nu_m[m] - self.band_nu_ref[0]
                powers = d_nu.unsqueeze(-1) ** self.poly_powers
                out[m] = torch.sum(c_i * powers, dim=-1)

            n_minus_1[mask_m] = out

        return n_minus_1[0] if scalar_input else n_minus_1


    def n_air(self, *args, **kwargs):
        return self.get_refractivity(*args, **kwargs) + 1.0


    def __call__(self, *args, **kwargs):
        return self.n_air(*args, **kwargs)


#%
# import matplotlib.pyplot as plt

# wvl = torch.linspace(0.5e-6, 1.2e-6, 10000, device="cuda", dtype=torch.float64)
# model = AirRefractiveIndexCalculator(dtype=torch.float64, device="cuda")

# n_minus_1 = model.get_refractivity(wvl, T_celsius=20.0, P_pa=101325.0, H_pct=20.0).cpu().numpy()
# plt.plot(wvl.cpu().numpy() * 1e6, n_minus_1 * 1e8)

# plt.xlabel("Wavelength (μm)")
# plt.ylabel("(n - 1) x 10⁸")
# plt.title("Refractive Index of Air vs Wavelength")
# plt.grid()
# plt.show()

