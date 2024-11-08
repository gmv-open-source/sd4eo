# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     checkPuzzlePiecies.py
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
# This is just a small script to study what's happeining inside the puzzle pieces
# As we have detected that non-valid-values have sneaked into our assemblies and 
# we are not using non valid values in our pice creation (beyond what is included 
# by rasterio itself)

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

class b20mFIT(Enum):
    PERFECT_MATCH = 1
    BIGGER_THAN_10m = 2
    PAD_NEEDED = 3
    ALLTOUCHED_NEEDED = 4
    NOT_VALID = 5

class PickleCombination(Enum):
    ALL_MONTH = 1           # Takes all dates from the same year-month
    MULTITILE = 2           # only for Catalonia dataset, only if same date is available
    ALL_MONTH_MT = 3        # only for Catalonia dataset
    ALL_MONTH_BOTH_DS = 4   # Same year-month both datatasets & multitile
    NOT_VALID = 5

CatTiles = ['T31TBF', 'T31TBG', 'T31TCF', 'T31TCG', 'T31TDG']

CyL_date_S2_list = ['20171009', '20171019', '20171029', '20171108', '20171118', '20171128', '20171208', '20171218', '20171223', '20180102', '20180107', '20180112', '20180117', '20180122', '20180127', '20180201', '20180206', '20180211', '20180216', '20180221', '20180226', '20180303', '20180308', '20180313', '20180318', '20180323', '20180328', '20180402', '20180407', '20180412', '20180417', '20180422', '20180427', '20180502', '20180507', '20180512', '20180517', '20180522', '20180527', '20180601', '20180606', '20180611', '20180616', '20180621', '20180626', '20180701', '20180706', '20180711', '20180716', '20180721', '20180726', '20180731', '20180810', '20180815', '20180820', '20180825', '20180830', '20180904', '20180909', '20180914', '20180919', '20180924', '20180929']

Cat_date_S2_dic = {
    'T31TBF': ['20171001', '20171016', '20171101', '20171116', '20171201', '20171216', '20180101', '20180116', '20180201', '20180301', '20180316', '20180401', '20180501', '20180516', '20180601', '20180701', '20180801', '20180816', '20180901'],
    'T31TBG': ['20171001', '20171016', '20171101', '20171116', '20171201', '20171216', '20180101', '20180116', '20180201', '20180216', '20180301', '20180316', '20180401', '20180501', '20180516', '20180601', '20180701', '20180716', '20180801', '20180816', '20180901'],
    'T31TCF': ['20171001', '20171016', '20171101', '20171116', '20171201', '20171216', '20180101', '20180116', '20180201', '20180301', '20180316', '20180401', '20180501', '20180516', '20180601', '20180701', '20180716', '20180801', '20180816', '20180901'],
    'T31TCG': ['20171001', '20171016', '20171101', '20171116', '20171201', '20171216', '20180101', '20180116', '20180201', '20180301', '20180316', '20180401', '20180501', '20180516', '20180601', '20180701', '20180716', '20180801', '20180816', '20180901'],
    'T31TDG': ['20171001', '20171016', '20171101', '20171116', '20171201', '20171216', '20180101', '20180116', '20180201', '20180216', '20180301', '20180316', '20180401', '20180501', '20180516', '20180601', '20180701', '20180801', '20180816', '20180901']
}

year_month_list = ['20171001', '20171101', '20171201', '20180101', '20180201', '20180301', '20180401', '20180501', '20180601', '20180701', '20180801', '20180901']

Romania_Crop_Codes = [('Barley', 5), ('Wheat', 1), ('OtherGrainLeguminous', 39), ('Peas' , 40), ('FallowAndBareSoil', 21), ('Vetch', 52), ('Alfalfa', 60), ('Sunflower', 33), ('Oats' , 8)]

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

def simple_count_below_threshold(array, threshold=-9990):
    """
    Counts the number of elements in a 2D numpy array that are below a specified threshold.
    
    Args:
    array (np.ndarray): A 2D numpy array.
    threshold (int, optional): The threshold value. Defaults to -9990.
    
    Returns:
    int: The count of elements below the threshold.
    """
    # Validate that the input is a 2D numpy array
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("The input must be a 2D numpy array.")
    
    # Count elements less than the threshold
    count = np.sum(array < threshold)
    return count


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
    crop_index = 1  # Change to select the desired crop type
    crop_name = Romania_Crop_Codes[crop_index][0]
    crop_type_ID = Romania_Crop_Codes[crop_index][1]
    # 1) Load all CyL pieces
    folder_to_examine = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/CyL/'
    # folder_to_examine = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/Cat/T31TBF/'
    # folder_to_examine = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/Cat/T31TBG/'
    # folder_to_examine = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/Cat/T31TCF/'
    # folder_to_examine = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/Cat/T31TCG/'
    # folder_to_examine = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/Cat/T31TDG/'
    for filename_i in list_files_with_extension(folder_to_examine, '.pickle'):
        full_filename = folder_to_examine + filename_i
        with open(full_filename, 'rb') as pickle_file:
            print(filename_i)
            # We load the serialized data structures from pickle file
            # We expect data to be a CropDeck sorted by pixel size of plots
            sorted_crop_deck = pickle.load(pickle_file)
            # print(full_filename)
            num_total_pieces = len(sorted_crop_deck.crop_in_region)
            num_fault_pieces = 0
            for crop_i in sorted_crop_deck.crop_in_region:
                # Let's examine each crop inside the deck
                num_bands = crop_i.array3D.shape[2]
                flag_view = False
                for band_i in range(num_bands):
                    num_bad_pixels = masked_count_below_threshold(np.squeeze(crop_i.array3D[:,:,band_i]), crop_i.mask2D)
                    if num_bad_pixels > 1:
                        if flag_view == False:
                            print(f'{folder_to_examine[-7:-1]} -> {filename_i}')
                        print(f'Crop # {crop_i.internal_index}  band {ALL_BAND_CODE[band_i]} num.pix. = {num_bad_pixels}')
                        # plt.figure()
                        # plt.imshow(np.squeeze(crop_i.array3D[:,:,band_i]))
                        # plt.colorbar()
                        # plt.title(f'Crop # {crop_i.internal_index}  band {ALL_BAND_CODE[band_i]} num.pix. = {num_bad_pixels}')
                        flag_view = True
                # if flag_view:
                #     for band_i in range(num_bands):
                #         plt.figure()
                #         plt.imshow(np.squeeze(crop_i.array3D[:,:,band_i]))
                #         plt.colorbar()
                #         plt.title(f'Crop # {crop_i.internal_index}  band {ALL_BAND_CODE[band_i]} num.pix. = {num_bad_pixels}')  
                #     plt.figure()
                #     plt.imshow(crop_i.mask2D)
                #     plt.show()





