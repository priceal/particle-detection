#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 13:14:25 2021

@author: allen
"""

from skimage.feature import peak_local_max

pkxy = peak_local_max(mapFinal,min_distance=3,threshold_rel=0.5)


plt.imshow(mapFinal)

plt.plot(pkxy[:,1],pkxy[:,0],'r.')