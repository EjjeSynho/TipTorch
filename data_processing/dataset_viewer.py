#%%
%reload_ext autoreload
%autoreload 2
%matplotlib qt


import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import os
from copy import deepcopy

view_folder = 'C:/Users/akuznets/Data/SPHERE/temp/'

files = os.listdir(view_folder)
imgs = [plt.imread(view_folder+file) for file in files]

#%%
axis_color = 'lightgoldenrodyellow'
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=-0.25, bottom=0.25)

img = ax.imshow( imgs[0] )

options = (
    '<None>',
    'Blurry',
    'LWE',
    'Empty',
    'Double',
    'Clipped',
    'Corrupted',
    'Streched',
    'Crossed',
    'Noisy',
    'Wingsgone',
    'Good')

im_id = 0

# Add a button for resetting the parameters
color_radios_ax = fig.add_axes([0.75, 0.5,  0.15, 0.35], facecolor=axis_color)
next_button_ax  = fig.add_axes([0.80, 0.90, 0.10, 0.04])
clear_button_ax = fig.add_axes([0.70, 0.90, 0.10, 0.04])

text = fig.text(0.8, 0.05, '', fontsize=12, verticalalignment='bottom')
color_radios = RadioButtons(color_radios_ax, options, active=0)
next_button  = Button(next_button_ax, 'Next', color=axis_color, hovercolor='0.975')
clear_button = Button(clear_button_ax, 'Clear', color=axis_color, hovercolor='0.975')

properties_current = []

labeled_data = []

def color_radios_on_clicked(label):
    global properties_current
    properties_current.append(label)
    test = ''
    for i in properties_current:
        test += i + '\n'
    text.set_text(test)
    fig.canvas.draw_idle()

ax.set_title(files[0].split('.')[0][4:])

def next_button_on_clicked(mouse_event):
    global im_id, properties_current
    ax.set_title(files[im_id].split('.')[0][4:])
    if im_id == len(files): plt.close('all')
    with open("C:/Users/akuznets/Data/SPHERE/labels.txt", "a") as myfile:
        buf = files[im_id].split('.')[0][4:] + ' '
        for i in properties_current: buf += i + ' '
        buf += '\n'
        myfile.write(buf)
    im_id += 1
    text.set_text('')
    properties_current = []
    img.set_data(imgs[im_id])
    fig.canvas.draw_idle()

def clear_button_on_clicked(mouse_event):
    global properties_current
    properties_current = []
    text.set_text('')
    fig.canvas.draw_idle()

color_radios.on_clicked(color_radios_on_clicked)
next_button.on_clicked(next_button_on_clicked)
clear_button.on_clicked(clear_button_on_clicked)


plt.show()

# %%
