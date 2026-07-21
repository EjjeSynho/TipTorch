import gc

import torch

from tiptorch._config import default_device
from tiptorch.managers.input_manager import InputsManager, InputsManagersUnion
from tiptorch.tools.cubic_splines import natural_cubic_spline_coeffs, NaturalCubicSpline
from tiptorch.tools.normalizers import DataTransform, Identity, SafeLog10, Uniform, Uniform0_1


class MoffatPSFModelNFM:
    """
    Compact NFM-wrapper-like model that renders elliptical Moffat PSFs in pixel space.

    Parameters may be passed explicitly, or inferred from a processed TipTorch-style
    config for convenience. The config is not stored or used by the renderer.
    """

    def __init__(
        self,
        config,
        N_spline_nodes=5,
        device=default_device,
        dtype=torch.float32,
        *,
        λ_min=475.0e-9,
        λ_max=935.0e-9,
        num_λ_slices=3681,
        **_,
    ):
        N_src = config["NumberSources"]
        wavelengths = config["sources_science"]["Wavelength"].squeeze()
        image_size = config["sensor_science"]["FieldOfView"]

        self.device = torch.device(device)
        self.dtype = dtype
        self.use_splines = N_spline_nodes is not None
        self.N_src = int(N_src)
        self.N_pix = int(image_size)
        self.model = self  # Backward-compatible with observations.py expecting PSF_model.model.N_pix.

        self.λ_sim = torch.as_tensor(wavelengths, device=self.device, dtype=self.dtype).flatten()
        self.λ_min = λ_min
        self.λ_max = λ_max
        self.num_λ_slices = num_λ_slices
        self.Δλ = (λ_max - λ_min) / (num_λ_slices - 1)
        self.λ_full = torch.linspace(λ_min, λ_max, num_λ_slices, device=self.device, dtype=self.dtype)

        self.polychromatic_params = ["F", "bg", "dx", "dy", "alpha_x", "alpha_y", "beta", "F_norm_λ"]

        if self.use_splines:
            if N_spline_nodes < 2 or N_spline_nodes > num_λ_slices:
                raise ValueError(f"N_spline_nodes must be between 2 and {num_λ_slices}. Got {N_spline_nodes}.")
            
            self.N_wvl_ctrl = N_spline_nodes
            self.norm_wvl = Uniform0_1(a=λ_min, b=λ_max)
            self.λ_ctrl_norm = torch.linspace(0, 1, N_spline_nodes, device=self.device, dtype=self.dtype)
            self.λ_ctrl = self.norm_wvl.inverse(self.λ_ctrl_norm)
            self.λ_sim_normed = self.norm_wvl(self.λ_sim)
            self.λ_full_normed = self.norm_wvl(self.λ_full)
        else:
            self.N_wvl_ctrl = None

        self._init_model_inputs()


    @property
    def N_wvl(self):
        return len(self.λ_sim)


    @staticmethod
    def _tree_to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu().clone()
        if isinstance(obj, dict):
            return {key: MoffatPSFModelNFM._tree_to_cpu(value) for key, value in obj.items()}
        return obj


    def save(self, *, cpu: bool = True):
        data = {
            "N_src": self.N_src,
            "wavelengths": self.λ_sim,
            "image_size": self.N_pix,
            "N_spline_nodes": self.N_wvl_ctrl,
            "λ_min": self.λ_min,
            "λ_max": self.λ_max,
            "num_λ_slices": self.num_λ_slices,
            "inputs": self.inputs_manager.save(),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
        }
        return self._tree_to_cpu(data) if cpu else data


    @classmethod
    def load(cls, data: dict, *, device=None):
        dtype = data.get("dtype", torch.float32)
        dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, str(dtype).replace("torch.", ""))
        instance = cls(
            N_src=data["N_src"],
            wavelengths=data["wavelengths"],
            image_size=data["image_size"],
            N_spline_nodes=data.get("N_spline_nodes"),
            device=device if device is not None else data.get("device", default_device),
            dtype=dtype,
            λ_min=data.get("λ_min", 475.0e-9),
            λ_max=data.get("λ_max", 935.0e-9),
            num_λ_slices=data.get("num_λ_slices", 3681),
        )
        instance.inputs_manager = instance.inputs_manager.load(data["inputs"])
        instance.inputs_manager.to(instance.device)
        return instance


    def copy(self):
        instance = MoffatPSFModelNFM(
            N_src=self.N_src,
            wavelengths=self.λ_sim.clone(),
            image_size=self.N_pix,
            N_spline_nodes=self.N_wvl_ctrl,
            device=self.device,
            dtype=self.dtype,
            λ_min=self.λ_min,
            λ_max=self.λ_max,
            num_λ_slices=self.num_λ_slices,
        )
        instance.inputs_manager = self.inputs_manager.copy()
        instance.backup_manager = self.backup_manager.copy()
        return instance


    def cleanup(self):
        gc.collect()


    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


    def _init_model_inputs(self):
        self.inputs_manager = InputsManagersUnion({"shared": InputsManager(), "per_src": InputsManager()})

        def add_input(name: str, values: torch.Tensor, norm: DataTransform = Identity(), optimizable: bool = True, is_shared: bool = False):
            manager = self.inputs_manager.input_managers["shared" if is_shared else "per_src"]
            manager.add(name, torch.as_tensor(values, device=self.device, dtype=self.dtype), norm, optimizable=optimizable)

        N_ctrl = self.N_wvl_ctrl if self.use_splines else self.N_wvl
        suffix = "_ctrl" if self.use_splines else ""

        add_input(f"F{suffix}",  torch.ones(1, N_ctrl), Uniform(a=0.0, b=1.0), is_shared=True)
        add_input(f"bg{suffix}", torch.zeros(1, N_ctrl), Uniform(a=-5e-6, b=5e-6), is_shared=True)
        add_input(f"dx{suffix}", torch.zeros(self.N_src, N_ctrl), Uniform(a=-1.0, b=1.0))
        add_input(f"dy{suffix}", torch.zeros(self.N_src, N_ctrl), Uniform(a=-1.0, b=1.0))
        add_input(f"F_norm_λ{suffix}", torch.ones(1, N_ctrl), Uniform(a=0.5, b=1.0), optimizable=False, is_shared=True)

        add_input(f"alpha_x{suffix}", torch.full((1, N_ctrl), 2.0), Uniform(a=0.2,  b=30.0), is_shared=True)
        add_input(f"alpha_y{suffix}", torch.full((1, N_ctrl), 2.0), Uniform(a=0.2,  b=30.0), is_shared=True)
        add_input(f"beta{suffix}",    torch.full((1, N_ctrl), 2.5), Uniform(a=1.01, b=10.0), is_shared=True)
        add_input("theta",  torch.zeros(self.N_src, 1), Uniform(a=-torch.pi / 2, b=torch.pi / 2))
        add_input("F_norm", torch.ones(self.N_src, 1), SafeLog10(), optimizable=True)

        self.inputs_manager.to(self.device)
        self.inputs_manager.to_float()
        self.inputs_manager.stack()
        self.backup_manager = self.inputs_manager.copy()
        self.per_src_inputs_list = list(self.inputs_manager.input_managers["per_src"].parameters)


    def reset_parameters(self):
        self.inputs_manager = self.backup_manager.copy()


    def get_optimizable_param_names(self):
        return self.inputs_manager.get_names(optimizable_only=True, flattened=False)


    def get_param_names(self):
        return self.inputs_manager.get_names(optimizable_only=False, flattened=False)


    def get_fixed_param_names(self):
        return list(set(self.get_param_names()) - set(self.get_optimizable_param_names()))


    def evaluate_splines(self, y_points, λ_grid):
        if not self.use_splines:
            return y_points
        if y_points.shape[-1] != self.N_wvl_ctrl:
            raise ValueError(f"Expected {self.N_wvl_ctrl} control wavelengths, got {y_points.shape[-1]}.")
        if y_points.ndim == 1:
            y_points = y_points.unsqueeze(0)
        spline = NaturalCubicSpline(natural_cubic_spline_coeffs(t=self.λ_ctrl_norm, x=y_points.T))
        return spline.evaluate(λ_grid).T


    def _evaluate_chromatic_inputs(self, x_dict):
        if not self.use_splines:
            return x_dict
        for name in self.polychromatic_params:
            ctrl_name = f"{name}_ctrl"
            if ctrl_name in x_dict:
                x_dict[name] = self.evaluate_splines(x_dict[ctrl_name], self.λ_sim_normed)
        return x_dict


    def update_manager_params(self, x_dict: dict, src_ids=None):
        self.inputs_manager.input_managers["per_src"].update(x_dict, selected_ids=src_ids)
        self.inputs_manager.input_managers["shared"].update(x_dict)


    def forward(self, x_dict=None, src_ids=None, include_list=None, update_params=True):
        need_update = update_params
        if x_dict is None:
            x_dict = self.inputs_manager.to_dict()
            need_update = False
        else:
            x_dict = dict(x_dict)

        if src_ids is not None:
            for key in self.per_src_inputs_list:
                if key in x_dict:
                    x_dict[key] = x_dict[key][src_ids].clone()
                    if x_dict[key].ndim <= 1:
                        x_dict[key] = x_dict[key].unsqueeze(0)

        x_dict = self._evaluate_chromatic_inputs(x_dict)
        if need_update:
            self.update_manager_params(x_dict, src_ids=src_ids)

        x = {key: x_dict[key] for key in include_list} if include_list is not None else x_dict
        return self._render(x)

    __call__ = forward


    def _expand(self, x, N_src, N_wvl, min_value=None):
        if x.ndim == 1:
            x = x.unsqueeze(1)
        if x.shape[0] == 1 and N_src > 1:
            x = x.repeat(N_src, 1)
        if x.shape[1] == 1 and N_wvl > 1:
            x = x.repeat(1, N_wvl)
        x = x[:N_src, :N_wvl]
        return x.clamp_min(min_value) if min_value is not None else x


    def _render(self, x):
        N_src = x["dx"].shape[0]
        N_wvl = self.N_wvl
        N_pix = self.N_pix

        coords = torch.arange(N_pix, device=self.device, dtype=self.dtype) - (N_pix - 1) / 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        dx = self._expand(x["dx"], N_src, N_wvl).view(N_src, N_wvl, 1, 1)
        dy = self._expand(x["dy"], N_src, N_wvl).view(N_src, N_wvl, 1, 1)
        
        alpha_x = self._expand(x["alpha_x"], N_src, N_wvl, min_value=1e-3).view(N_src, N_wvl, 1, 1)
        alpha_y = self._expand(x["alpha_y"], N_src, N_wvl, min_value=1e-3).view(N_src, N_wvl, 1, 1)
        beta  = self._expand(x["beta"], N_src, N_wvl, min_value=1.01).view(N_src, N_wvl, 1, 1)
        theta = self._expand(x["theta"], N_src, N_wvl).view(N_src, N_wvl, 1, 1)

        # dx/dy shifts are applied in pixel-space by evaluating the Moffat kernel at (x-dx, y-dy)

        x0 = xx.view(1, 1, N_pix, N_pix) - dx
        y0 = yy.view(1, 1, N_pix, N_pix) - dy
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x_rot =  x0 * cos_t + y0 * sin_t
        y_rot = -x0 * sin_t + y0 * cos_t

        psf = (1.0 + (x_rot / alpha_x) ** 2 + (y_rot / alpha_y) ** 2).pow(-beta)
        psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

        F  = self._expand(x["F"],  N_src, N_wvl).view(N_src, N_wvl, 1, 1)
        bg = self._expand(x["bg"], N_src, N_wvl).view(N_src, N_wvl, 1, 1)
        
        return psf * F + bg

    def __getitem__(self, item):
        return self.inputs_manager[item]


    def __setitem__(self, key, value):
        self.inputs_manager[key] = value


    def SetWavelengths(self, wavelengths: torch.Tensor):
        wavelengths = torch.as_tensor(wavelengths, device=self.device, dtype=self.dtype).flatten()
        if self.λ_sim.shape == wavelengths.shape and torch.allclose(wavelengths, self.λ_sim, atol=1e-12):
            return
        
        self.λ_sim = wavelengths
        if self.use_splines:
            self.λ_sim_normed = self.norm_wvl(self.λ_sim)


    def SetImageSize(self, img_size: int):
        self.N_pix = int(img_size)


    @torch.no_grad()
    def SimulateSpectralRange(self, λ_min=None, λ_max=None, src_ids=None, λ_batch_size=100, sequential=True, verbose=False, force_cpu=True):
        λ_lo = self.λ_full.min().item() if λ_min is None else λ_min
        λ_hi = self.λ_full.max().item() if λ_max is None else λ_max
        mask = (self.λ_full >= λ_lo) & (self.λ_full <= λ_hi)
        if mask.sum() == 0:
            raise ValueError(f"No wavelength slices found in [{λ_lo}, {λ_hi}]. Check units.")

        λ_range        = self.λ_full[mask]
        global_indices = mask.nonzero(as_tuple=True)[0]
        src_ids = list(range(self.N_src)) if src_ids is None else ([src_ids] if isinstance(src_ids, int) else list(src_ids))
        N_src_sim  = len(src_ids)
        out_device = torch.device("cpu") if force_cpu else self.device
        out = torch.empty(N_src_sim, len(λ_range), self.N_pix, self.N_pix, device=out_device, dtype=self.dtype)
        initial_wavelengths = self.λ_sim.clone()

        # Pre-evaluate all chromatic splines over the full λ_full range once, then slice per batch.
        # This mirrors NFM_wrapper.SimulateSpectralRange and avoids re-running the spline solver
        # inside every forward() call.
        x_dict = self.inputs_manager.to_dict()
        x_dict_λ_full = {}
        if self.use_splines:
            for name in self.polychromatic_params:
                ctrl_name = f"{name}_ctrl"
                if ctrl_name in x_dict:
                    x_dict_λ_full[name] = self.evaluate_splines(x_dict[ctrl_name], self.λ_full_normed)
                    x_dict.pop(ctrl_name)
        else:
            for name in self.polychromatic_params:
                if name in x_dict:
                    x_dict_λ_full[name] = x_dict.pop(name)

        # Slice per-source rows in advance
        per_src_keys = list(self.inputs_manager.input_managers["per_src"].parameters.keys())
        non_ctrl_per_src = [k.replace("_ctrl", "") for k in per_src_keys]
        for key in per_src_keys:
            if key in x_dict:
                x_dict[key] = x_dict[key][src_ids]
                if x_dict[key].ndim <= 1:
                    x_dict[key] = x_dict[key].unsqueeze(0)
        for name in non_ctrl_per_src:
            if name in x_dict_λ_full:
                x_dict_λ_full[name] = x_dict_λ_full[name][src_ids]

        out_batches = range(0, len(λ_range), λ_batch_size)
        if verbose:
            from tqdm import tqdm
            out_batches = tqdm(out_batches, total=len(range(0, len(λ_range), λ_batch_size)), desc="Simulating spectral range")

        for start in out_batches:
            stop      = min(start + λ_batch_size, len(λ_range))
            out_sl    = slice(start, stop)
            global_sl = global_indices[out_sl]

            self.SetWavelengths(λ_range[out_sl])

            batch_dict = dict(x_dict)
            for name, vals in x_dict_λ_full.items():
                batch_dict[name] = vals[..., global_sl]

            out[:, out_sl] = self._render(batch_dict).to(out_device)

        self.SetWavelengths(initial_wavelengths)
        
        return out, λ_range


    def SimulateFullSpectrum(self, src_ids=None, λ_batch_size=100, verbose=False, force_cpu=True):
        return self.SimulateSpectralRange(src_ids=src_ids, λ_batch_size=λ_batch_size, verbose=verbose, force_cpu=force_cpu)[0]