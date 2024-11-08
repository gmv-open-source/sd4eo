# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     fixWrongPixelsInPuzzle.py
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


# I don't understand how, but some erroneous pixels have crept into some of the puzzles. 
# Despite all the modifications I made, there are some pixels with NaN values (-9999) in the VH band 
# of the largest puzzles. I think the only explanation is that these are pixels that haven't been properly 
# filled in during the puzzle filling process due to tiny numerical errors, but it's almost impossible.
# This script locates and fills in those values for ALL layers with the value of another (near) pixel.

import argparse, copy
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from PIL import Image
import sys, os
from scipy.stats import skew, kurtosis
import time
import xarray as xr

import sutils
import steerable_pyramid as steerable
import texture_analysis as ta
# from multichannelFileFormatMS1 import show_mcArray, load_NC_patch, save_new_NC_patch, bands_IDs
from multichannelFileFormat import load_from_netCDF, save_as_netCDF, show_mcArray,MultiChannelFileMetadata, year_month_list, Romania_Crop_Codes, ALL_BAND_CODE, S_BAND_INDX




def count_below_threshold(array, threshold=-9990):
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



def list_nc_files(directory):
    """
    List all files with a '.nc' extension within a specified directory.
    
    Args:
    directory (str): The directory to search within.
    
    Returns:
    list: A list of filenames ending with '.nc' found within the specified directory.
    """
    # List comprehension to find all '.nc' files
    return [file for file in os.listdir(directory) if file.endswith('.nc')]


def find_elements_below_threshold(array2d, threshold=-9990):
    """Finds coordinates of elements below a given threshold in a 2D numpy array.

    Args:
        array2d (np.array): Input 2D array.
        threshold (float): Threshold value to find elements below.

    Returns:
        list: List of tuples (row, col) indicating the positions of elements below the threshold.
    """
    rows, cols = np.where(array2d < threshold)
    return list(zip(rows, cols))


def replace_with_neighborhood_average(array2d, coords):
    """Replaces elements in the given coordinates with the average of their neighborhood.

    Args:
        array2d (np.array): Input 2D array.
        coords (list): List of tuples (row, col) for elements to replace.

    Returns:
        np.array: Updated array with replaced values.
    """
    max_rows, max_cols = array2d.shape
    for row, col in coords:
        neighbors = []

        # Check all 8 possible neighbors
        for r in range(row - 1, row + 2):
            for c in range(col - 1, col + 2):
                if (r, c) != (row, col) and 0 <= r < max_rows and 0 <= c < max_cols:
                    if (r, c) not in coords:
                        neighbors.append(array2d[r, c])

        # Replace with the average of neighbors if there are any valid neighbors
        if neighbors:
            array2d[row, col] = np.mean(neighbors)

    return array2d

if __name__ == "__main__":  
    input_folder = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/CyL_and_Cat/'
    list_wrong_files = ['20171001_Alfalfa_1536.nc', '20171001_Wheat_2048.nc', '20171101_Barley_2048.nc', '20180101_Barley_2048.nc', '20180101_Wheat_2048.nc', '20180201_Barley_2048.nc', '20180301_Barley_2048.nc', '20180401_Barley_2048.nc, 20180401_Wheat_1536.nc', '20180501_Barley_2048.nc', '20180501_Wheat_2048.nc', '20180601_Alfalfa_1024.nc', '20180601_Wheat_1536.nc', '20180701_Barley_2048.nc', '20180701_Wheat_2048.nc', '20180801_Barley_2048.nc', '20180901_Barley_2048.nc']
    norm_flag = True
    view_flag = False

    # for input_filename in list_wrong_files:
    for input_filename in list_nc_files(input_folder):
        flag_preview = False # True
        full_nc_path_filename = input_folder + input_filename
        full_band_numpy_array, max_value, max_visible_value, metadata = load_from_netCDF(full_nc_path_filename, flag_show=flag_preview)
        num_bands = full_band_numpy_array.shape[2]
        # print(num_bands)
        for i in range(num_bands):
            num_wrong_pixels = count_below_threshold(full_band_numpy_array[:,:,i], threshold=-9990)
            if num_wrong_pixels > 0:
                print(f'file {input_filename} ->  band {ALL_BAND_CODE[i]} : {num_wrong_pixels}')
                # plt.figure()
                # # plt.imshow(np.squeeze(full_band_numpy_array[:,:,i])<-9990)
                # plt.imshow(np.squeeze(full_band_numpy_array[:,:,i]))
                # plt.colorbar()
                # plt.title(ALL_BAND_CODE[i])
                # plt.show()
                coords = find_elements_below_threshold(np.squeeze(full_band_numpy_array[:,:,i]), threshold=-9990)
                for j in range(num_bands):
                    full_band_numpy_array[:,:,j] = replace_with_neighborhood_average(np.squeeze(full_band_numpy_array[:,:,j]), coords)
                save_as_netCDF(full_band_numpy_array, full_nc_path_filename + '.extra.nc', metadata.crop_type_name, metadata.date, metadata.dataset, metadata.synthetic_method, metadata.max_visible_value, metadata)
    
        # new_composed_puzzle, new_composed_metadata = extract_and_transform_bands(input_folder + input_filename, flag_normalization=norm_flag, flag_preview=view_flag)
        # full_output_filename = output_folder + input_filename[:-3] + '_slim.nc'
        # save_as_slim_netCDF(new_composed_puzzle, full_output_filename, new_composed_metadata, flag_revert_normalization=norm_flag, flag_show_3_bands=view_flag)
    

