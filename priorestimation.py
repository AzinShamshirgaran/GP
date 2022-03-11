#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:13:09 2022
@author: azin
"""

import math, random
import os, sys


import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from matplotlib import cm, animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from scipy.stats import multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern

Goals_locations=[[0,0],[32, 13], [23, 39], [43, 18], [16, 38], [41, 21], [26, 26], [ 7, 42], [ 2, 46], [32, 43], [18, 18], [48,  8], [17, 23], [41, 31], [39, 33], [27, 42], [13, 44], [43,3], [46, 42], [12, 16], [23,1], [9,7], [17,  1], [31, 17], [36,38],  [13,48], [23, 20], [12, 14], [34, 40], [7, 10], [28, 10], [37,  8], [22, 17], [37, 21], [38, 26], [12, 33], [ 1, 27], [30, 44], [20, 28], [ 9, 28], [18,20], [39,14], [ 3, 14], [10, 22], [43 ,20], [47, 43], [12, 1], [10, 49], [15, 35], [8,4], [ 3, 40], [31, 8], [17, 45], [45, 50], [33, 38], [37, 36], [34, 15], [11, 44], [43, 41], [ 5 ,41], [38, 19], [18, 39],  [28, 49], [36, 18], [32, 21] ,[36, 15], [48, 40], [45, 31], [15, 23], [14, 18], [35,  2], [37, 11], [1 , 3], [35, 19], [50, 26], [43, 25], [31 ,46], [39, 38], [11, 38], [46, 50], [35, 45], [35, 48], [44,  9], [30, 3], [ 5, 45],[ 2,12],[12,16], [ 3,10],[50,29],[47,16], [46,28], [17, 7], [25,47], [46,46], [40,27], [22,11], [ 8,45], [20,44], [46,24],[50,50]]  


xmin,ymin,xmax,ymax = (-1,-1,51,51)
xw, yw = 500, 500
xstep, ystep = ((xmax-xmin)/xw, (ymax-ymin)/yw)

x, y = np.mgrid[xmin:xmax:xstep, ymin:ymax:ystep]
pos = np.dstack((x, y))

#model1
rv = multivariate_normal([17,22 ], [[20, -10], [-10, 20]])
rv2=  multivariate_normal([13, 44], [[50, 30], [10, 30]])
rv3=  multivariate_normal([36, 18], [[20, 30], [20, 40]])
rv4=  multivariate_normal([37, 36], [[100, 50], [-70, 80]])
rv5=  multivariate_normal([10, 10], [[50, 30], [10, 30]])
z = rv.pdf(pos)+rv2.pdf(pos)+rv3.pdf(pos)+rv4.pdf(pos)++rv5.pdf(pos)


samploc_x=[]
samploc_y=[]
for jj in Goals_locations:
    samploc_x.append(jj[0])
    samploc_y.append(jj[1])
samploc_x=[int(a) for a in samploc_x]
samploc_y=[int(a) for a in samploc_y]
test_pts = np.dstack((samploc_x,samploc_y))[0]

def coord_to_index(values, seq):
    return [(np.abs(seq-n)).argmin() for n in values]
xseq, yseq = pos[:,0][:,0], pos[0,:][:,1]

sampx, sampy = coord_to_index(samploc_x, xseq), coord_to_index(samploc_y, yseq)

sampz = z[sampx, sampy]


coord_pts = np.dstack((sampx,sampy))[0]


kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

gp.fit(coord_pts, sampz)
z_pred, z_cov = gp.predict(test_pts, return_cov=True)

print(z_pred.tolist())
