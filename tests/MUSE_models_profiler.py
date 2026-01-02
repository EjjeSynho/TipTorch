#%%
%reload_ext autoreload
%autoreload 2

import sys
sys.path.insert(0, '..')

import torch
from project_settings import device
from PSF_models.TipTorch import TipTorch


with torch.no_grad():
    pupil = torch.tensor( PupilVLT(samples=320, rotation_angle=0), device=device )

    PSD_include = {
        'fitting':         True,
        'WFS noise':       True,
        'spatio-temporal': True,
        'aliasing':        True,
        'chromatism':      True,
        'diff. refract':   False,
        'Moffat':          True
    }
    
    tiptorch_half = TipTorch(config_file, 'LTAO', pupil, PSD_include, 'sum', device, oversampling=1)
    tiptorch_half.to_float()
    
#%%
from torch.utils.tensorboard import SummaryWriter
import torch.profiler as profiler
import torch.nn as nn

writer = SummaryWriter(log_dir='./runs/tiptorch_half_inference_profiling')

# model = nn.Linear(10, 5)
# model.eval()  # Set the model to evaluation mode
# input_tensor = torch.randn(1, 10)

for _ in range(10):
    # tiptorch_half.Update(regrids=True, repupils=True)
    tiptorch_half(x=inputs)


# Define the profiler
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA],  # Use CUDA if available
    on_trace_ready=profiler.tensorboard_trace_handler('./runs/tiptorch_half_inference_profiling'),
    record_shapes=True,
    with_stack=True  # Optional: Add stack tracing
    
) as prof:
    with torch.no_grad():
        # tiptorch_half.Update(regrids=True, repupils=True)
        tiptorch_half(x=inputs)
        

# Print profiler summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# prof.export_chrome_trace("trace.json")


#%%
"""
from torch.utils.tensorboard import SummaryWriter
import torch.profiler as profiler
import torch.nn as nn

n_air = AirRefractiveIndexCalculator(device=torch.device("cuda:0"), dtype=torch.float32)
wvl = torch.tensor(1.6e-6, device=torch.device("cuda:0"), dtype=torch.float32)

writer = SummaryWriter(log_dir='./runs/tiptorch_half_inference_profiling')

# model = nn.Linear(10, 5)
# model.eval()  # Set the model to evaluation mode
# input_tensor = torch.randn(1, 10)

for _ in range(10):
    # tiptorch_half.Update(regrids=True, repupils=True)
    n_air(wvl)


# Define the profiler
with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA],  # Use CUDA if available
    on_trace_ready=profiler.tensorboard_trace_handler('./runs/tiptorch_half_inference_profiling'),
    record_shapes=True,
    with_stack=True
    
) as prof:
    with torch.no_grad():
        # tiptorch_half.Update(regrids=True, repupils=True)
        n_air(wvl)


# Print profiler summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# prof.export_chrome_trace("trace.json")
"""