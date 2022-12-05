#%%
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm


def remove_transparency(im, bg_colour=(255, 255, 255)):
    alpha = im.convert('RGBA').split()[-1]
    bg = Image.new("RGBA", im.size, bg_colour + (255,))
    bg.paste(im, mask=alpha)
    return bg

#folder = 'C:\\Users\\akuznets\\Data\\SPHERE\\DATA\\captures\\MLP\\'
folder = 'C:\\Users\\akuznets\\Data\\SPHERE\\DATA\\captures\\NeRF\\'

files = os.listdir(folder)
files1 = [str(i)+'.png' for i in range(len(files))]

#%%
gif_anim = []

path = 'C:\\Users\\akuznets\\Data\\SPHERE\\DATA\\captures\\NeRF.gif'

for file in tqdm(files1):
    with Image.open(folder+file) as im:
        gif_anim.append( remove_transparency(im) )
gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=True, duration=0.001, loop=0)
print('Writing')

# %%
