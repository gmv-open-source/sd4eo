# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     checkBandConsistency.py
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
# Small script to check band consistency in a pair of pickle 

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
import re
from scipy.ndimage import distance_transform_edt
import time
from typing import Any, List
import xarray as xr

NON_VALID_VALUE = -9999
NAN_THRESHOLD = -9990
MIN_CROP_SIZE_PIXELS = 3
MAX_SIDE_MARGIN = 20
S2BAND10m_CODE = ['B02', 'B03', 'B04', 'B08']
S2BAND20m_CODE = ['B05', 'B06', 'B07', 'B11', 'B12', 'B8A']
S1BAND_CODE = ['VH'] # ['VH','VV']  # CyL : only one | Catalonia : all
ALL_BAND_CODE = S2BAND10m_CODE + S2BAND20m_CODE + S1BAND_CODE
S_BAND_INDX = {
  'B02': 0,
  'B03': 1,
  'B04': 2, 
  'B08': 3,  # This is the origin of the bug !!! 
  'B05': 4, 
  'B06': 5,
  'B07': 6, 
  'B11': 7, 
  'B12': 8, 
  'B8A': 9,
  'VH': 10,
  'VV': 11
}

year_month_list = ['20171001', '20171101', '20171201', '20180101', '20180201', '20180301', '20180401', '20180501', '20180601', '20180701', '20180801', '20180901']

Romania_Crop_Codes = [('Barley', 5), ('Wheat', 1), ('OtherGrainLeguminous', 39), ('Peas' , 40), ('FallowAndBareSoil', 21), ('Vetch', 52), ('Alfalfa', 60), ('Sunflower', 33), ('Oats' , 8)]


class b20mFIT(Enum):
    PERFECT_MATCH = 1
    BIGGER_THAN_10m = 2
    PAD_NEEDED = 3
    ALLTOUCHED_NEEDED = 4
    NOT_VALID = 5

@dataclass
class CropRegisterClass:
    internal_index : int = 0
    uuid : int = 0
    crop_type_ID : int = 1
    area_ha : float = 0.0
    area_px : int = 0
    b10m_rows : int = 0
    b10m_cols : int = 0
    mask2D : Any = None
    distance_map_2D : Any = None
    array3D : Any = None
    b20m_margin_ID : Any = b20mFIT.NOT_VALID
    S1_marginID : Any = b20mFIT.NOT_VALID
    # coords_in_texture_domain_row : int = 0
    # coords_in_texture_domain_col : int = 0

@dataclass
class CropDeck:
    crop_in_region: List[CropRegisterClass] = None
    AOI_ID : str = ''
    date_str : str = ''
    reference_band : str = 'B02'

@dataclass
class CropCoords:
    row : int = 0
    col : int = 0
    internal_index : int = 0
    uuid : int = 0

def remap_bands(array3D_old):
    ''' It provides a new array with the bands in the expected order'''
    array3D_new = copy.deepcopy(array3D_old)
    # Let's rearrange the wrong bands
    array3D_new[:,:,3] = array3D_old[:,:,6] # B08 (new) <- B07 (old)
    array3D_new[:,:,4] = array3D_old[:,:,3] # B05 (new) <- B08 (old)
    array3D_new[:,:,5] = array3D_old[:,:,4] # B06 (new) <- B05 (old)
    array3D_new[:,:,6] = array3D_old[:,:,5] # B07 (new) <- B06 (old)
    return array3D_new



if __name__ == "__main__": 
    path_1 = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/CyL/'
    path_2 = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/CyL.old/'
    filename = '20180929_Oats.pickle'
    with open(path_1 + filename, 'rb') as pickle_file:
        sorted_crop_deck_1 = pickle.load(pickle_file)
        first_crop_1 = sorted_crop_deck_1.crop_in_region[0]
    with open(path_2 + filename, 'rb') as pickle_file:
        sorted_crop_deck_2 = pickle.load(pickle_file)
        first_crop_2 = sorted_crop_deck_2.crop_in_region[0]        
    # Check it is the same crop
    if first_crop_1.internal_index != first_crop_2.internal_index:
        print(f'Not the same internal index : {first_crop_1.internal_index} vs {first_crop_2.internal_index}')
    if first_crop_1.uuid != first_crop_2.uuid:
        print(f'Not the same uuid : {first_crop_1.uuid} vs {first_crop_2.uuid}')   
    num_bands = first_crop_1.array3D.shape[2]
    # Rearrange old bands
    first_crop_2.array3D = remap_bands(first_crop_2.array3D)

    # Check each rearranged band
    for i in range(num_bands):
        if np.any(first_crop_1.array3D[:,:,i] != first_crop_2.array3D[:,:,i]):
            print(f'Mistmatch band: {ALL_BAND_CODE[i]}')
            plt.figure()
            plt.imshow(first_crop_1.array3D[:,:,i])
            plt.colorbar()
            plt.title(f'New : band {ALL_BAND_CODE[i]}')
            plt.figure()
            plt.imshow(first_crop_2.array3D[:,:,i])
            plt.colorbar()
            plt.title(f'Old : band {ALL_BAND_CODE[i]}')
            plt.show()
