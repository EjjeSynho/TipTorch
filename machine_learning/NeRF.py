#%%
import sys
sys.path.insert(0, '..')

import torch
import numpy as np
from torch import nn
from data_processing.SPHERE_data import SPHERE_dataset
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
#device = torch.device('cpu')

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, D, W, in_channels_xy, in_channels_p, skips=[3]):#D=8 and W=256 respectively
        """
        D: number of layers for the encoder
        W: number of hidden units in each layer
        in_channels_xy: number of input channels for x and y
        in_channels_p: number of input channels for telemetry parameters
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xy = in_channels_xy
        self.in_channels_p  = in_channels_p

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xy, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xy, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xy_encoding_final = nn.Linear(W, W)

        # parameters encoding layer
        self.p_encoding = nn.Sequential( nn.Linear(W+in_channels_p, W//2), nn.ReLU(True) )

        # output layer
        self.intensity = nn.Linear(W//2, 1)
        #self.intensity = nn.Sequential(  #TODO: for normalized intensity, should it use a sigmoid?
        #                nn.Linear(W, 1),
        #                nn.Sigmoid())


    def forward(self, x):
        """
        Encodes input (xy+p) to intensity I(x,y), where p are the telemetry parameters

        Inputs:
            x: (B, xy_channels+num_params)
               the embedded vector of pixel coordinates and telemetry parameters
        Outputs:
            I: (B, 1) intensity as a function (x,y)
        """
        
        input_xy, input_p = torch.split(x, [self.in_channels_xy, self.in_channels_p], dim=-1)

        xy_ = input_xy
        for i in range(self.D):
            if i in self.skips:
                xy_ = torch.cat([input_xy, xy_], -1)
            xy_ = getattr(self, f"xyz_encoding_{i+1}")(xy_)

        xy_encoding_final = self.xy_encoding_final(xy_)
        p_encoding = self.p_encoding(torch.cat([xy_encoding_final, input_p], -1))
        intensity  = self.intensity(p_encoding)

        return intensity

#%%
#la chignon et tarte
dataset = SPHERE_dataset()
database_wvl = dataset.FilterWavelength()
p_in, p_out, i0, i1 = dataset.GenerateDataset(database_wvl)

i01 = i0 / torch.amax(i0, dim=(1,2), keepdim=True)
i11 = i1 / torch.amax(i1, dim=(1,2), keepdim=True)

crop_win = 32
ROI = slice(i0.shape[1]//2-crop_win//2, i0.shape[2]//2+crop_win//2)

dI = torch.nan_to_num((i1-i0)[:,ROI,ROI])
dI /= dI.amax(dim=(1,2)).median()

#%%
rand_id = np.random.randint(dI.shape[0])
plt.imshow(dI[rand_id,:,:].pow(2).cpu())
plt.show()

#%%
N_samp = dI.shape[0]
N_val = 20

x = ( torch.range(-dI.shape[1]//2, dI.shape[1]//2-1) + 0.5 ) / dI.shape[1]*2
y = ( torch.range(-dI.shape[2]//2, dI.shape[2]//2-1) + 0.5 ) / dI.shape[2]*2
xx, yy = torch.meshgrid(x, y, indexing='ij')
xx = xx.unsqueeze(0).unsqueeze(0).repeat([N_samp,1,1,1])
yy = yy.unsqueeze(0).unsqueeze(0).repeat([N_samp,1,1,1])
pp = torch.nan_to_num( p_in.unsqueeze(-1).unsqueeze(-1).repeat([1,1,dI.shape[1],dI.shape[2]]) )
data = torch.cat((xx,yy,dI.unsqueeze(1),pp),1)

pixel_ids = torch.range(0,len(data.flatten())-1).int().reshape(data.shape)

embedding_xy = Embedding(2, 10)

flatten_data = lambda x: x.permute([1,0,2,3]).flatten(start_dim=1).T
embed_data   = lambda x: ( x[:,2].unsqueeze(1).float(), torch.hstack([embedding_xy(x[:,:2]), x[:,3:]]).float() )
#embed_data   = lambda x: ( x[:,2].unsqueeze(1).float(), torch.hstack([x[:,:2], x[:,3:]]).float() )

ids = torch.randperm(N_samp)
dI_train, pix_train = embed_data(flatten_data(data[ids[N_val:],:,:,:]))
dI_val,   pix_val   = embed_data(flatten_data(data[ids[:N_val],:,:,:]))

dI_train  = dI_train.to(device)
pix_train = pix_train.to(device)
pix_val   = pix_val.to(device)
dI_val    = dI_val.to(device)

#%%
model = NeRF(D=8, W=256,
             in_channels_xy = pix_train.shape[1]-p_in.shape[1],
             in_channels_p = p_in.shape[1],
             skips = [3]).to(device)

loss_fn = nn.L1Loss(reduction='mean')
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

#%%
%matplotlib qt
loss_vals = []
loss_trains = []

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(0,0)
line2, = ax.plot(0,0)

id_show = 0

ax2 = fig.add_axes([.5, .5, .6, .2])
ax3 = fig.add_axes([.5, .75, .6, .2])
im = ax2.imshow(model(pix_val).reshape([N_val, dI.shape[1], dI.shape[2]]).detach().cpu()[id_show,:,:])
im2 = ax3.imshow(dI.detach().cpu()[id_show,:,:])


for epoch in range(2000):
    dI_hat_train = model(pix_train)
    dI_hat_val = model(pix_val)
    loss = loss_fn(dI_hat_train, dI_train)
    loss.backward()

    loss_trains.append(loss.item())
    loss_vals.append(loss_fn(dI_hat_val, dI_val).item())

    sys.stdout.write("%.4f" % loss.item())
    sys.stdout.write("\b")
    sys.stdout.write("\r")
    sys.stdout.flush()

    img = dI_hat_val.reshape([N_val, dI.shape[1], dI.shape[2]]).detach().cpu()[id_show,:,:]
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
    plt.savefig('C:/Users/akuznets/Data/SPHERE/DATA/captures/NeRF/'+str(epoch)+'.png')

    opt.step()

# %%
print(loss_fn(model(pix_train), dI_train).item())
print(loss_fn(model(pix_val), dI_val).item())

#%%
%matplotlib inline

test_0 = dI_val.reshape([N_val, dI.shape[1], dI.shape[2]])
test_1 = model(pix_val).reshape([N_val, dI.shape[1], dI.shape[2]])
print(loss_fn(test_0, test_1).item())

rect = [0.2,0.2,0.7,0.7]
ax1 = add_subplot_axes(ax,rect)

plt.imshow(test_0[0,:,:].detach().cpu())
plt.show()
plt.imshow(test_1[0,:,:].detach().cpu())
plt.show()

#%%


