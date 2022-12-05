#%%
import sys
sys.path.insert(0, '..')

import torch
import numpy as np
from torch import nn
from data_processing.SPHERE_data import SPHERE_dataset
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
from torchsummary import summary

#device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
device = torch.device('cpu')

#%%
#la chignon et tarte
dataset = SPHERE_dataset()
database_wvl = dataset.FilterWavelength()
p_in, p_out, i0, i1 = dataset.GenerateDataset(database_wvl)

i01 = i0 / torch.amax(i0, dim=(1,2), keepdim=True)
i11 = i1 / torch.amax(i1, dim=(1,2), keepdim=True)

#%%
crop_win = 24
ROI = slice(i0.shape[1]//2-crop_win//2, i0.shape[2]//2+crop_win//2)
dI = torch.nan_to_num((i11-i01)[:,ROI,ROI])
dI /= dI.amax(dim=(1,2)).median()

#%%
rand_id = np.random.randint(dI.shape[0])
plt.imshow(dI[rand_id,:,:].pow(2).cpu())
plt.show()

#dI_1 = interpolate(dI.unsqueeze(1), size=(dI.shape[1]//2,dI.shape[2]//2), mode='bicubic').squeeze(1)
dI_1 = dI
N_pix = dI_1.shape[1]

plt.imshow(dI_1[rand_id,:,:].pow(2).cpu())
plt.show()

#%%
ids = torch.randperm(dI_1.shape[0])

dI_1 = dI_1.flatten(start_dim=1)

p_in_train = torch.nan_to_num(p_in[ids[20:],:]).float().to(device)
p_in_val   = torch.nan_to_num(p_in[ids[:20],:]).float().to(device)
dI_1_train = dI_1[ids[20:],:].float().to(device)
dI_1_val   = dI_1[ids[:20],:].float().to(device)

#%%
net_sizes = [p_in.shape[1], 128, 256, dI_1.shape[1]]
layers = []
for i in range(len(net_sizes)-1):
    layers.append(nn.Linear(net_sizes[i], net_sizes[i+1]))
    layers.append(nn.GELU())
layers = layers[:-1]

model = nn.Sequential(*layers).to(device)

summary(model)

loss_fn = nn.MSELoss(reduction='mean')
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 1000
#%%
%matplotlib qt
loss_vals = []
loss_trains = []

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(0,0, label='Train')
line2, = ax.plot(0,0, label='Validation')
plt.legend(loc='lower left')
id_show = 1

ax2 = fig.add_axes([.5, .5, .6, .2])
ax3 = fig.add_axes([.5, .75, .6, .2])
im = ax2.imshow(model(p_in_val).reshape([dI_1_val.shape[0], N_pix, N_pix]).detach().cpu()[id_show,:,:])
im2 = ax3.imshow(dI_1_val.reshape([dI_1_val.shape[0], N_pix, N_pix]).detach().cpu()[id_show,:,:])

for epoch in range(2000):
    #a = time()
    dI_hat_train = model(p_in_train)
    dI_hat_val = model(p_in_val)
    loss = loss_fn(dI_hat_train, dI_1_train)
    loss.backward()

    loss_trains.append(loss.item())
    loss_vals.append(loss_fn(dI_hat_val, dI_1_val).item())

    sys.stdout.write("%.4f" % loss.item())
    sys.stdout.write("\b")
    sys.stdout.write("\r")
    sys.stdout.flush()

    img = dI_hat_val.reshape([dI_1_val.shape[0], N_pix, N_pix]).detach().cpu()[0,:,:]
    im.set_data(img)
    im.set_clim(vmin=img.min(), vmax=img.max())

    ax.set_ylim([0.0, max([max(loss_trains), max(loss_vals)])])
    ax.set_xlim([0.0, epoch])

    line1.set_xdata(np.arange(0, epoch+1))
    line2.set_xdata(np.arange(0, epoch+1))
    line1.set_ydata(np.array(loss_trains))
    line2.set_ydata(np.array(loss_vals))
    
    fig.canvas.draw()
    fig.canvas.flush_events()
 
    plt.savefig('C:/Users/akuznets/Data/SPHERE/DATA/captures/MLP/'+str(epoch)+'.png')

    opt.step()
    #b = time()
    #print(b-a)

# %%
