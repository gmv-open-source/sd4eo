# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     forceBandConsistency.py
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
# Small script to force new check band consistency in Catalonia pickles
# This script solves/patches a problem caused by a data-bug detected in May 2024
# It should only be executed ONCE

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


def list_files_with_extension(directory, ext_str='.pickle'):
    """
    List all files with a specific extension within a specified directory.
    
    Args:
    directory (str): The directory to search within.
    
    Returns:
    list: A list of filenames ending with the extension found within the specified directory.
    """
    # List comprehension to find all files with the indicated extension
    return [file for file in os.listdir(directory) if file.endswith(ext_str)]

def masked_count_below_threshold(matrix, mask, threshold=-9990):
    """
    Counts the number of elements in 'matrix' that are below a given 'threshold'
    at positions where 'mask' is True.
    
    Args:
    matrix (np.ndarray): A 2D numpy array of numerical values.
    mask (np.ndarray): A 2D boolean array where True indicates that the corresponding
                       element in 'matrix' should be considered for threshold comparison.
    threshold (int, optional): The threshold value to compare against. Defaults to -9990.
    
    Returns:
    int: The number of elements in 'matrix' that are below 'threshold' at positions where 'mask' is True.
    """
    if matrix.shape != mask.shape:
        raise ValueError("The shape of the matrix and mask must be the same.")
    
    # Apply mask and find elements below the threshold
    filtered_values = matrix[mask]  # Elements of matrix where mask is True
    below_threshold = filtered_values < threshold  # Boolean array where filtered values are below threshold
    return np.sum(below_threshold)  # Count of True values

if __name__ == "__main__": 
    old_base_path = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/Cat.old/'
    new_base_path = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/Cat/'
    Tile_ID_list = ['T31TBF', 'T31TBG', 'T31TCF', 'T31TCG', 'T31TDG'] 

    for tile_id in Tile_ID_list:
        old_full_path = old_base_path + tile_id + '/'
        new_full_path = new_base_path + tile_id + '/'
        file_list = list_files_with_extension(old_full_path, '.pickle')
        for filename_i in file_list:
            # Load CropDeck data structure
            with open(old_full_path + filename_i, 'rb') as old_pickle_file:
                # We load the serialized data structures from pickle file
                # We expect data to be a CropDeck sorted by pixel size of plots
                old_sorted_crop_deck = pickle.load(old_pickle_file)
            elements_to_remove = []
            for crop_i in old_sorted_crop_deck.crop_in_region:
                # First, we remap the bands
                crop_i.array3D = remap_bands(crop_i.array3D)
                # Second we mark the bad crops if detected 
                num_bad_pixels = masked_count_below_threshold(np.squeeze(crop_i.array3D[:,:,S_BAND_INDX['VH']]), crop_i.mask2D)
                if num_bad_pixels > 1:
                    print(f'{tile_id} -> {filename_i}')
                    print(f'Crop # {crop_i.internal_index}  band VH   num.pix. = {num_bad_pixels}')
                    elements_to_remove.append(crop_i)
            # We remove all bad crops
            for item in elements_to_remove:
                old_sorted_crop_deck.crop_in_region.remove(item)
            with open(new_full_path+filename_i, 'wb') as new_pickle_file:
                pickle.dump(old_sorted_crop_deck, new_pickle_file)
                                 



