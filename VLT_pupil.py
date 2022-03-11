# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:07:00 2021

@author: akuznets
"""

import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt


def CircPupil(samples, D=8.0, centralObstruction=1.12):
    x      = np.linspace(-1/2, 1/2, samples)*D
    xx,yy  = np.meshgrid(x,x)
    circle = np.sqrt(xx**2 + yy**2)
    obs    = circle >= centralObstruction/2
    pupil  = circle < D/2 
    return pupil * obs


def PupilVLT(samples, vangle=[0,0], petal_modes=False):
    pupil_diameter = 8.0	  # pupil diameter [m]
    secondary_diameter = 1.12 # diameter of central obstruction [m] 1.12
    alpha = 101				  # spider angle [degrees]
    spider_width = 0.039	  # spider width [m] 0.039;  spider is 39 mm
    # wide excepted on small areas where 50 mm width are reached over a length 
    # of 80 mm,near the centre of the spider (before GRAVITY modification?), 
    # see VLT-DWG-AES-11310-101010

    shx = np.cos(np.deg2rad(vangle[1]))* 101.4*np.tan(np.deg2rad(vangle[0]/60))  # shift of the obscuration on the entrance pupil [m]
    shx = np.cos(np.deg2rad(vangle[1]))* 101.4*np.tan(np.deg2rad(vangle[0]/60))  # shift of the obscuration on the entrance pupil [m]
    shy = np.sin(np.deg2rad(vangle[1]))* 101.4*np.tan(np.deg2rad(vangle[0]/60))  # shift of the obscuration on the entrance pupil [m]
    delta = pupil_diameter/samples # distance between samples [m]
    ext = 2*np.max(np.fix(np.abs(np.array([shx, shy]))))+1

    # create coordinate matrices
    x1_min = -(pupil_diameter+ext - 2*delta)/2
    x1_max = (pupil_diameter + ext)/2
    num_grid = int((x1_max-x1_min)/delta)+1

    x1 = np.linspace(x1_min, x1_max, num_grid) #int(1/delta))
    x, y = np.meshgrid(x1, x1)

    #  Member data
    mask = np.ones([num_grid, num_grid], dtype='bool')
    mask[ np.where( np.sqrt( (x-shx)**2 + (y-shy)**2 ) > pupil_diameter/2 ) ] = False
    mask[ np.where( np.sqrt( x**2 + y**2 ) < secondary_diameter/2 ) ] = False

    # Spiders
    alpha_rad = alpha * np.pi / 180
    slope     = np.tan( alpha_rad/2 )

    petal_1 = np.zeros([num_grid, num_grid], dtype='bool')
    petal_2 = np.zeros([num_grid, num_grid], dtype='bool')
    petal_3 = np.zeros([num_grid, num_grid], dtype='bool')
    petal_4 = np.zeros([num_grid, num_grid], dtype='bool')

    #North
    petal_1[ np.where(   
        (( -y > 0.039/2 + slope*(-x - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x<0)  & (y<=0)) | \
        (( -y > 0.039/2 + slope*( x - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x>=0) & (y<=0)) )] = True
    petal_1 *= mask

    #East 
    petal_2[ np.where(   
        (( -y < 0.039/2 + slope*( x - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x>0) & (y<=0)) | \
        ((  y < 0.039/2 + slope*( x - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x>0) & (y>0)) )] = True
    petal_2 *= mask
        
    #South
    petal_3[ np.where(   
        ((  y > 0.039/2 + slope*(-x - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x<=0) & (y>0)) | \
        ((  y > 0.039/2 + slope*( x - secondary_diameter/2 ) + spider_width/np.sin( alpha_rad/2 )/2) & (x>0)  & (y>0)) )] = True
    petal_3 *= mask
        
    #West
    petal_4[ np.where(   
        (( -y < 0.039/2 + slope*(-x - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x<0) & (y<0)) |\
        ((  y < 0.039/2 + slope*(-x - secondary_diameter/2 ) - spider_width/np.sin( alpha_rad/2 )/2) & (x<0) & (y>=0)) )] = True
    petal_4 *= mask
        
    lim_x = [ ( np.fix((shy+ext/2)/delta) ).astype('int'), ( -np.fix((-shy+ext/2)/delta) ).astype('int') ]
    lim_y = [ ( np.fix((shx+ext/2)/delta) ).astype('int'), ( -np.fix((-shx+ext/2)/delta) ).astype('int') ]

    petal_1 = resize(petal_1[ lim_x[0]:-1+lim_x[1], lim_y[0]:-1+lim_y[1] ], (samples, samples), anti_aliasing=False)
    petal_2 = resize(petal_2[ lim_x[0]:-1+lim_x[1], lim_y[0]:-1+lim_y[1] ], (samples, samples), anti_aliasing=False)
    petal_3 = resize(petal_3[ lim_x[0]:-1+lim_x[1], lim_y[0]:-1+lim_y[1] ], (samples, samples), anti_aliasing=False)
    petal_4 = resize(petal_4[ lim_x[0]:-1+lim_x[1], lim_y[0]:-1+lim_y[1] ], (samples, samples), anti_aliasing=False)

    if petal_modes:
        xx1, yy1 = np.meshgrid(np.linspace( -0.5, 0.5,  samples), np.linspace(-0.25, 0.75, samples))
        xx2, yy2 = np.meshgrid(np.linspace(-0.75, 0.25, samples), np.linspace( -0.5, 0.5,  samples))
        xx3, yy3 = np.meshgrid(np.linspace( -0.5, 0.5,  samples), np.linspace(-0.75, 0.25, samples))
        xx4, yy4 = np.meshgrid(np.linspace(-0.25, 0.75, samples), np.linspace( -0.5, 0.5,  samples))

        def normalize_petal_mode(petal, coord):
            mode = petal.astype('double') * coord
            mode -= mode.min()
            mode /= (mode.max()+mode.min())
            mode -= 0.5
            mode[np.where(petal==False)] = 0.0
            mode[np.where(petal==True)] -= mode[np.where(petal==True)].mean()
            mode /= mode[np.where(petal==True)].std()
            return mode

        tip_1 = normalize_petal_mode(petal_1, yy1)
        tip_2 = normalize_petal_mode(petal_2, yy2)
        tip_3 = normalize_petal_mode(petal_3, yy3)
        tip_4 = normalize_petal_mode(petal_4, yy4)

        tilt_1 = normalize_petal_mode(petal_1, xx1)
        tilt_2 = normalize_petal_mode(petal_2, xx2)
        tilt_3 = normalize_petal_mode(petal_3, xx3)
        tilt_4 = normalize_petal_mode(petal_4, xx4)

        return np.dstack( [petal_1, petal_2, petal_3, petal_4, tip_1, tip_2, tip_3, tip_4, tilt_1, tilt_2, tilt_3, tilt_4] )

    else:
        return petal_1 + petal_2 + petal_3 + petal_4
