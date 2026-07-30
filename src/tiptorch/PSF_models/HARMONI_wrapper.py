from copy import deepcopy
from pathlib import Path

import torch
from astropy.io import fits

from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM
from tiptorch.PSF_models.TipTorch import TipTorch
from tiptorch.managers.config_manager import ConfigManager
from tiptorch.tools.static_phase import ArbitraryBasis, ZernikeBasis
from tiptorch.tools.utils import to_little_endian
from tiptorch._config import default_device


class PSFModelHARMONI(PSFModelNFM):
    """Fitting-ready TipTorch wrapper for ELT/HARMONI."""

    def __init__(
        self,
        config,
        multiple_obs    = False,
        LO_NCPAs        = True,
        use_Zernike     = True,
        use_static_WFE  = True,
        static_WFE_path = None,
        static_WFE_scale = 1.0,
        use_Moffat      = False,
        retain_PSDs     = False,
        Z_mode_max      = 9,
        N_spline_nodes  = 5,
        device          = default_device,
        dtype           = torch.float32,
        model_type      = None,
        *,
        λ_min           = 470.e-9,
        λ_max           = 2450.e-9,
        num_λ_slices    = 3961,
    ):
        self._config_raw = config
        self.use_splines = False
        self.use_Zernike = bool(LO_NCPAs and use_Zernike)
        self.static_WFE_path = self._resolve_static_WFE_path(config, static_WFE_path)
        self.use_static_WFE = bool(
            LO_NCPAs and use_static_WFE and self.static_WFE_path is not None
        )
        self.static_WFE_scale = float(static_WFE_scale)
        self.static_WFE_mode = None

        phase_aberrations = self.use_Zernike or self.use_static_WFE

        super().__init__(
            config=config,
            multiple_obs=multiple_obs,
            LO_NCPAs=phase_aberrations,
            chrom_defocus=False,
            use_Moffat=use_Moffat,
            retain_PSDs=retain_PSDs,
            Z_mode_max=Z_mode_max,
            N_spline_nodes=N_spline_nodes,
            device=device,
            dtype=dtype,
            model_type=model_type,
            λ_min=λ_min,
            λ_max=λ_max,
            num_λ_slices=num_λ_slices,
        )

    @staticmethod
    def _resolve_static_WFE_path(config, static_WFE_path):
        if static_WFE_path is not None:
            return str(Path(static_WFE_path))

        configs = [config] if isinstance(config, dict) else list(config)
        paths = {
            entry.get('telescope', {}).get('PathStaticOn')
            or entry.get('PathStaticOn')
            for entry in configs
            if isinstance(entry, dict)
        }
        paths.discard(None)

        if len(paths) > 1:
            raise ValueError(
                "Different static WFE maps for different observations are not supported."
            )

        return str(Path(paths.pop())) if paths else None

    def _init_model(self, config):
        physics_on = self.model_type != 'psfao'
        PSD_include = {
            'fitting':         True,
            'WFS noise':       physics_on,
            'spatio-temporal': physics_on,
            'aliasing':        False,
            'chromatism':      physics_on,
            'diff. refract':   physics_on,
            'Moffat':          self.model_type in ('hybrid', 'psfao'),
        }

        self.model = TipTorch(
            AO_config=config,
            AO_type='MCAO' if physics_on else 'PSFAO',
            pupil=None,
            PSD_include=PSD_include,
            norm_regime='sum',
            device=self.device,
            oversampling=1,
            retain_PSDs=self.retain_PSDs,
            dtype=self.dtype,
        )

    def _load_static_WFE(self):
        """Load a pupil-sampled static WFE map whose values are in nanometres."""
        path = Path(self.static_WFE_path)
        static_WFE = fits.getdata(path)

        if static_WFE.ndim != 2:
            raise ValueError(
                f"Static WFE map must be two-dimensional, got shape {static_WFE.shape}."
            )

        static_WFE = torch.as_tensor(
            to_little_endian(static_WFE),
            device=self.device,
            dtype=self.dtype,
        )

        pupil = self.model.pupil.squeeze()
        if static_WFE.shape != pupil.shape:
            raise ValueError(
                f"Static WFE map shape {tuple(static_WFE.shape)} does not match "
                f"the pupil shape {tuple(pupil.shape)}."
            )
        if not torch.isfinite(static_WFE).all():
            raise ValueError("Static WFE map contains non-finite values.")

        return static_WFE * pupil

    def _init_NCPAs(self):
        basis = []
        self.static_WFE_mode = None

        if self.use_static_WFE:
            self.static_WFE_mode = 0
            basis.append(self._load_static_WFE().unsqueeze(0))

        if self.use_Zernike:
            if self.Z_mode_max < 3:
                raise ValueError("Z_mode_max must be at least 3 when Zernikes are enabled.")
            zernike_basis = ZernikeBasis(
                self.model,
                N_modes=self.Z_mode_max,
                ignore_pupil=False,
            )
            basis.append(zernike_basis.basis[2:self.Z_mode_max])

        self.LO_basis = ArbitraryBasis(
            self.model,
            torch.cat(basis, dim=0),
            ignore_pupil=False,
        )
        self.LO_N_params = self.LO_basis.N_modes

    def _init_model_inputs(self):
        self.λ_sim = self.λ_sim.flatten()
        if self.use_splines:
            self.λ_sim_normed = self.norm_wvl(self.λ_sim)

        super()._init_model_inputs()

        if self.static_WFE_mode is not None:
            self.inputs_manager['LO_coefs'][:, self.static_WFE_mode] = (
                self.static_WFE_scale
            )
            self.inputs_manager.stack()
            self.backup_manager = self.inputs_manager.copy()

    def save(self, *, cpu=True):
        store_data = super().save(cpu=False)
        store_data.pop('chrom_defocus', None)
        store_data.update({
            'instrument':        'HARMONI',
            'use_Zernike':       self.use_Zernike,
            'use_static_WFE':    self.use_static_WFE,
            'static_WFE_path':   self.static_WFE_path,
            'static_WFE_scale':  self.static_WFE_scale,
        })
        return self._tree_to_cpu(store_data) if cpu else store_data

    @classmethod
    def load(cls, store_data, *, device=None, config=None):
        def dtype_from_name(dtype):
            if isinstance(dtype, torch.dtype):
                return dtype
            name = str(dtype or 'float32').replace('torch.', '')
            return getattr(torch, name)

        device = torch.device(
            device if device is not None else store_data.get('device', default_device)
        )
        dtype = dtype_from_name(store_data.get('dtype'))
        raw_config = deepcopy(store_data.get('config', config))

        if raw_config is None:
            raise ValueError(
                "The saved wrapper does not contain a config. Pass one via "
                "PSFModelHARMONI.load(..., config=...)."
            )

        if (
            isinstance(raw_config, dict)
            and 'NumberSources' in raw_config
            and torch.is_tensor(raw_config['atmosphere']['Seeing'])
        ):
            ConfigManager().Convert(
                raw_config,
                framework='pytorch',
                device=device,
                dtype=dtype,
            )

        instance = cls(
            config=raw_config,
            multiple_obs=store_data.get(
                'multiple_obs', store_data.get('multiple_OBs', False)
            ),
            LO_NCPAs=store_data.get('LO_NCPAs', True),
            use_Zernike=store_data.get('use_Zernike', True),
            use_static_WFE=store_data.get('use_static_WFE', True),
            static_WFE_path=store_data.get('static_WFE_path'),
            static_WFE_scale=store_data.get('static_WFE_scale', 1.0),
            model_type=store_data.get(
                'model_type',
                'hybrid' if store_data.get('use_Moffat', False)
                else 'physics-based',
            ),
            retain_PSDs=store_data.get('retain_PSDs', False),
            Z_mode_max=store_data.get('Z_mode_max', 9),
            N_spline_nodes=store_data.get('N_spline_nodes'),
            device=device,
            dtype=dtype,
            λ_min=store_data['λ_min'],
            λ_max=store_data['λ_max'],
            num_λ_slices=store_data['num_λ_slices'],
        )

        instance.inputs_manager = instance.inputs_manager.load(store_data['inputs'])
        instance.inputs_manager.to(device)
        if dtype == torch.float64:
            instance.inputs_manager.to_double()
        else:
            instance.inputs_manager.to_float()
        instance.backup_manager = instance.inputs_manager.copy()
        return instance

    def copy(self):
        return type(self).load(
            self.save(cpu=False),
            device=self.device,
            config=deepcopy(self._config_raw),
        )
