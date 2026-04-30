import sys
sys.path.insert(0, '../..')

import warnings
import torch
import torch.nn as nn
import pandas as pd
from typing import Optional
from data_processing.MUSE.data_utils import filter_dataframe, reduce_dataframe, TELEMETRY_CACHE
from managers.input_manager import InputsTransformer
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import pickle


class SmallCalibratorNet(nn.Module):
    """
    Compact neural network for calibration with strong regularization.
    Designed for small datasets (~500 samples).
    
    Args:
        n_features: Number of input features
        n_outputs: Number of output values
        hidden_dim: Size of hidden layers (default: 32)
        dropout_rate: Dropout probability (default: 0.3)
    """
    def __init__(self, n_features, n_outputs, hidden_dim=32, dropout_rate=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Better than BatchNorm for small batches
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(hidden_dim, n_outputs),
        )
        
        # Initialize weights with smaller values for better regularization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Smaller gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)



class NFMCalibrator():
    def __init__(
        self,
        checkpoint_path,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.device = device
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Initialize NN architecture from saved state
        self.net = SmallCalibratorNet(
            n_features   = state['net_arch']['n_features'],
            n_outputs    = state['net_arch']['n_outputs'],
            hidden_dim   = state['net_arch']['hidden_dim'],
            dropout_rate = state['net_arch']['dropout_rate'],
        ).to(device)
        
        self.net.load_state_dict(state['net_state_dict'])
        self.net.eval()
        self.net.to(device)    

        # Outputs transformer — stored as a plain dict from InputsTransformer.save()
        raw = state['outputs_transformer']
        self.outputs_transformer = InputsTransformer.load(data=raw)
        
        # Inputs transformers
        with open(TELEMETRY_CACHE / 'MUSE/muse_telemetry_imputer.pickle', 'rb') as handle:
            self.telemetry_imputer = pickle.load(handle)

        with open(TELEMETRY_CACHE / 'MUSE/muse_telemetry_scaler.pickle', 'rb') as handle:
            self.telemetry_scaler = pickle.load(handle)
            self.features = self.telemetry_scaler.feature_names_in_

        # Aux params
        self.LO_modes_max   = state.get('LO_modes_max', None)
        self.N_spline_nodes = state.get('N_spline_nodes', None)


    def prepare_telemetry(self, reduced_telemetry: pd.DataFrame) -> torch.Tensor:
        telemetry_pruned = reduce_dataframe(filter_dataframe(reduced_telemetry))
        # Standartize/normalize features
        telemetry_ = self.telemetry_scaler.transform(telemetry_pruned[self.features])
        # Restore feature names back to avoid warnings
        telemetry_ = pd.DataFrame(telemetry_, columns=self.features, index=telemetry_pruned.index)
        # Impute missing values
        telemetry_ = self.telemetry_imputer.transform(telemetry_)
        return torch.tensor(telemetry_, dtype=torch.float32, device=self.device)


    def forward(self, reduced_telemetry: pd.DataFrame):
        # Prepare the telemetry data: filter, reduce, scale, and impute
        telemetry_vector = self.prepare_telemetry(reduced_telemetry)
    
        # Pass through the network to get predictions for the PSF model inputs
        x_pred = self.net(telemetry_vector)
        x_dict_pred = self.outputs_transformer.unstack(x_pred)

        # Prepend a zeros column as the phase-bump placeholder (not predicted by the net)
        if 'LO_coefs' in x_dict_pred:
            coefs = x_dict_pred['LO_coefs']
            phase_bump = torch.zeros(coefs.shape[0], 1, device=coefs.device, dtype=coefs.dtype)
            x_dict_pred['LO_coefs'] = torch.cat([phase_bump, coefs], dim=-1)

        return x_dict_pred


    def __call__(self, reduced_telemetry: pd.DataFrame):
        return self.forward(reduced_telemetry)


    @torch.no_grad()
    def calibrate(self, reduced_telemetry: pd.DataFrame, PSF_model) -> None:
        """
        Method to get a reconciled parameter dict ready for PSFModelNFM.
        Checks compatibility and adapts the predicted parameters to match the PSF model's expected input structure.
        """
        # Do parameters dict prediction
        x_dict_pred = self.forward(reduced_telemetry)
        # If PSF modle has slightly different settings, adapt the predicted parameters accordingly (e.g. by cropping or padding with zeros)
        x_dict_adapted = self._adapt_to_PSF_model(x_dict_pred, PSF_model)
        # Update PSF model's internal PSF model inputs with the predicted parameters
        PSF_model.update_manager_params(x_dict_adapted)
        # Trigger PSF update with new parameters
        _ = PSF_model()


    def check_compatibility(self, PSF_model) -> dict:
        """
        Check structural compatibility between the calibrator and a PSFModelNFM instance.
        Issues warnings for each mismatch.
        """
        issues = {}

        if self.LO_modes_max is not None and hasattr(PSF_model, 'Z_mode_max'):
            n_calib = self.LO_modes_max - 2  # modes predicted (phase bump excluded)
            n_model = PSF_model.Z_mode_max - 2
            if n_calib != n_model:
                issues['LO_modes'] = (n_calib, n_model)
                action = 'cropping' if n_calib > n_model else 'padding with zeros'
                warnings.warn(
                    f"NFMCalibrator: LO modes mismatch — calibrator predicts {n_calib} "
                    f"Zernike modes, PSF model expects {n_model}. Will adapt by {action}."
                )

        if self.N_spline_nodes is not None and hasattr(PSF_model, 'N_wvl_ctrl'):
            n_calib = self.N_spline_nodes
            n_model = PSF_model.N_wvl_ctrl
            if n_calib != n_model:
                issues['N_ctrl'] = (n_calib, n_model)
                warnings.warn(
                    f"NFMCalibrator: N_spline_nodes mismatch — calibrator has {n_calib} "
                    f"control points, PSF model has {n_model}. Will re-sample via cubic spline."
                )
        
        return issues


    @staticmethod
    def _resample_spline_ctrls(values: torch.Tensor, n_to: int) -> torch.Tensor:
        """
        Re-sample spline control-point values to a different number of uniformly-spaced
        nodes in [0, 1] using natural cubic splines.
        """
        n_from = values.shape[-1]
        t_from = torch.linspace(0, 1, n_from, device=values.device, dtype=values.dtype)
        t_to   = torch.linspace(0, 1, n_to,   device=values.device, dtype=values.dtype)
        # torchcubicspline expects [n_nodes, batch] for x
        coeffs = natural_cubic_spline_coeffs(t_from, values.T)
        spline = NaturalCubicSpline(coeffs)
        return spline.evaluate(t_to).T  # [batch, n_to]


    def _adapt_to_PSF_model(self, x_dict: dict, PSF_model) -> dict:

        x_dict = dict(x_dict)  # shallow copy, tensors are not cloned

        # Spline control points
        if self.N_spline_nodes is not None and hasattr(PSF_model, 'N_wvl_ctrl'):
            n_ctrl_calib = self.N_spline_nodes
            n_ctrl_model = PSF_model.N_wvl_ctrl
            if n_ctrl_calib != n_ctrl_model:
                for key, val in x_dict.items():
                    if (
                        key.endswith('_ctrl')
                        and isinstance(val, torch.Tensor)
                        and val.dim() == 2
                        and val.shape[-1] == n_ctrl_calib
                    ):
                        x_dict[key] = self._resample_spline_ctrls(val, n_ctrl_model)

        # LO modes (first column = phase bump, already included)
        if 'LO_coefs' in x_dict and hasattr(PSF_model, 'Z_mode_max'):
            n_model = PSF_model.Z_mode_max - 1  # total modes the model needs (incl. phase bump)
            coefs   = x_dict['LO_coefs']
            n_have  = coefs.shape[-1]
            
            if n_have > n_model:
                x_dict['LO_coefs'] = coefs[:, :n_model]
                
            elif n_have < n_model:
                pad = torch.zeros(coefs.shape[0], n_model - n_have, device=coefs.device, dtype=coefs.dtype)
                x_dict['LO_coefs'] = torch.hstack((coefs, pad))

        return x_dict
