import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from managers.input_manager import InputsTransformer

class Gnosis(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=100, dropout_p=0.25):
        super(Gnosis, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(dropout_p)
        self.fc4 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x))
        return x


class Calibrator(nn.Module):

    def __init__(self, inputs_manager, calibrator_network, predicted_values, device=torch.device('cpu'), dtype=torch.float32):
        super().__init__()
        self.device = device
        self.predicted_values = predicted_values
        self.dtype = dtype
        
        # Initialiize inputs normalizer and staker/unstacker
        self.normalizer = InputsTransformer({ inp: inputs_manager.get_transform(inp) for inp in predicted_values })
        _ = self.normalizer.stack({ inp: inputs_manager[inp] for inp in predicted_values }, no_transform=True)

        # Initialize the calibrator network
        net_class      = calibrator_network['artichitecture']
        inputs_size    = calibrator_network['inputs_size']
        outputs_size   = calibrator_network['outputs_size'] if 'outputs_size' in calibrator_network else self.normalizer.get_stacked_size()
        weights_folder = calibrator_network['weights_folder']
        NN_kwargs      = calibrator_network['NN_kwargs']

        self.net = net_class(inputs_size, outputs_size, **NN_kwargs)
        self.net.to(dtype=self.dtype)
        self.net.to(device)
        self.net.load_state_dict(torch.load(weights_folder, map_location=torch.device('cpu')))

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()
    
    def forward(self, x):
        # Handle conversion to tensor based on input type
        if isinstance(x, (pd.DataFrame, pd.Series)):
            NN_inp = torch.as_tensor(x.to_numpy(), device=self.device, dtype=self.dtype)
        elif isinstance(x, (list, np.ndarray)):
            NN_inp = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        elif isinstance(x, torch.Tensor):
            # Only convert if needed
            if x.device != self.device or x.dtype != self.dtype:
                NN_inp = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
            else:
                NN_inp = x  # Use as is, no conversion needed
        else:
            raise ValueError('NN_inputs must be a pandas DataFrame, numpy array, list, or torch tensor')
        
        # Handle single floats or 1D arrays
        if NN_inp.ndim == 1:
            NN_inp = NN_inp.view(1, -1)
        
        # Process and return
        # Scale the inputs back to the original range and pack them into the dictionary format
        return self.normalizer.unstack(self.net(NN_inp))
    
    '''
    def forward(self, x):
        if type(x) is pd.DataFrame or type(x) is pd.Series:
            NN_inp = torch.as_tensor(x.to_numpy(), device=self.device, dtype=self.dtype)
        elif type(x) is list or type(x) is np.ndarray:
            NN_inp = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        elif type(x) is torch.Tensor:
            NN_inp = x.to(self.device, dtype=self.dtype)
        else:
            raise ValueError('NN_inputs must be a pandas DataFrame, numpy array, list, or torch tensor')

        if NN_inp.ndim == 1: NN_inp = NN_inp.unsqueeze(0)

        # Scale the inputs back to the original range and pack them into the dictionary format
        return self.normalizer.unstack(self.net(NN_inp))
    '''