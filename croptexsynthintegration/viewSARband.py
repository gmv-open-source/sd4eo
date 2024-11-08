# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     viewSARband.py
# Author:        David Miraut
# License:       MIT License
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
		
# Description: 
# Small script to visizulize VH SAR band

import copy
from dataclasses import dataclass
from enum import Enum
from scipy.signal import fftconvolve
import geopandas as gpd
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle
import random
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import re
from scipy.ndimage import distance_transform_edt
import time
from typing import Any, List
import xarray as xr

from multichannelFileFormat import load_from_netCDF, save_as_netCDF, show_mcArray, MultiChannelFileMetadata, year_month_list, Romania_Crop_Codes, S2BAND10m_CODE, S2BAND20m_CODE,ALL_BAND_CODE, S_BAND_INDX



if __name__ == "__main__":  
    for i in range(5):
        # full_filename = f'C:/DMTA/projects/SD4EO/chaotic-aggregation/CyL_and_Cat/20180{i}01_Barley_2048.nc'
        full_filename = f'C:/DATASETs/OUT_HighOrder/HO_Barley_20180801_1024_{i}.nc'
        numpy_array, max_value, max_visible_value, metadata = load_from_netCDF(full_filename, flag_show=True)

if __name__ == "__main__2":  

    i = 5
    # full_filename = f'C:/DMTA/projects/SD4EO/chaotic-aggregation/CyL_and_Cat/20180{i}01_Barley_2048.nc'
    full_filename = f'HO_Sunflower_20171001_256_{i}.nc'
    numpy_array, max_value, max_visible_value, metadata = load_from_netCDF(full_filename, flag_show=True)
    vh_band = np.squeeze(numpy_array[:,:,S_BAND_INDX['VH']])
    for b_name in ALL_BAND_CODE:
        v_band = np.squeeze(numpy_array[:,:,S_BAND_INDX[b_name]])
        plt.figure()
        plt.imshow(v_band)
        plt.colorbar()
        plt.title(b_name)
    plt.figure()
    plt.imshow(vh_band)
    plt.title('VH')
    plt.figure()
    plt.imshow(vh_band < -9000)
    plt.title('VH bool')
    plt.show()
    # rgb_array = np.zeros((numpy_array.shape[0], numpy_array.shape[1], 3))
    # rgb_array[:,:,0] = numpy_array[:,:,2] # R
    # rgb_array[:,:,1] = numpy_array[:,:,1] # G
    # rgb_array[:,:,2] = numpy_array[:,:,0] # B
    # if max_visible_value == 0:
    #     max_visible_value = np.max(np.max(rgb_array))
    # rgb_array = rgb_array * 1.0/max_visible_value
    # plt.imshow(rgb_array)
    # if title_str != '':
    #     plt.title(title_str)
    # plt.show()

    pass

