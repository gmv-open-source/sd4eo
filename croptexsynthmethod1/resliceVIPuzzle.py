# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     multichannelFileFormat.py
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
# This is a very simple script that take the full puzzle and extracts only the key bands
# to generate a 3-channel new puzzle with only the needed information for the crop use case:
#  S2:  NDVI = (B08 - B04) / (B08 + B04) = (nir - red) / (nir + red)
#  S2:  MSI = B11/B08 = swir/nir
#  S1:  VH
#
# It takes all the full band puzzles from a folder and transform them into a slimmer 
# version with only these Vegetation Indexes and VH band

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
from multichannelFileFormat import load_from_netCDF, save_as_netCDF, show_mcArray, normalize_array, reverse_normalize_array, MultiChannelFileMetadata, year_month_list, Romania_Crop_Codes, ALL_BAND_CODE, S_BAND_INDX



def extract_and_transform_bands(full_nc_path_filename, flag_normalization=False, flag_preview=False):
    """
    This function only takes bands B04, B08, B11, and VH and cobines them into NDVI + MSI + VH
    """
    # 1) Extract data
    full_band_numpy_array, max_value, max_visible_value, metadata = load_from_netCDF(full_nc_path_filename, flag_show=flag_preview)
    S2_band04 = full_band_numpy_array[:,:,S_BAND_INDX['B04']] # red
    S2_band08 = full_band_numpy_array[:,:,S_BAND_INDX['B08']] # nir
    S2_band11 = full_band_numpy_array[:,:,S_BAND_INDX['B11']] # swir
    S1_bandVH = full_band_numpy_array[:,:,S_BAND_INDX['VH']]  
    # 2) Estimate vegetation indexes
    # NDVI = (B08 - B04) / (B08 + B04) = (nir - red) / (nir + red)
    S2_NDVI = (S2_band08 - S2_band04) / (S2_band08 + S2_band04)
    #  S2:  MSI = B11/B08 = swir/nir
    S2_MSI = S2_band11 / S2_band08
    # 3) Rebuild key metadata
    new_bands_IDs = ['NDVI', 'MSI', 'VH']
    max_NDVI = np.max(np.max(S2_NDVI))
    min_NDVI = np.min(np.min(S2_NDVI))
    max_MSI = np.max(np.max(S2_MSI))
    min_MSI = np.min(np.min(S2_MSI))
    max_VH = metadata.band_max[S_BAND_INDX['VH']]
    min_VH = metadata.band_min[S_BAND_INDX['VH']]
    new_c_band_max = [max_NDVI, max_MSI, max_VH]
    new_c_band_min = [min_NDVI, min_MSI, min_VH]
    max_all_bands = np.max(new_c_band_max)
    result_VH = count_below_threshold(S1_bandVH)
    if result_VH > 0:
        print('Dynamic ranges:')
        print(f'  - NDVI: [{min_NDVI},{max_NDVI}]')
        print(f'  - MSI: [{min_MSI},{max_MSI}]')
        print(f'  - VH: [{min_VH},{max_VH}]')
        print("Number of elements below threshold in VH:", result_VH)
        print('')

    # 4) Compose new data structure
    new_composed_puzzle = np.zeros((full_band_numpy_array.shape[0], full_band_numpy_array.shape[1], 3))
    if flag_normalization:
        new_composed_puzzle[:,:,0] = normalize_array(S2_NDVI[:,:], max_NDVI, min_NDVI)
        new_composed_puzzle[:,:,1] = normalize_array(S2_MSI[:,:], max_MSI, min_MSI)
        new_composed_puzzle[:,:,2] = normalize_array(S1_bandVH[:,:], max_VH, min_VH)
    else:
        new_composed_puzzle[:,:,0] = S2_NDVI[:,:]
        new_composed_puzzle[:,:,1] = S2_MSI[:,:]
        new_composed_puzzle[:,:,2] = S1_bandVH[:,:]
    new_composed_metadata = MultiChannelFileMetadata(metadata.crop_type_name, max_all_bands, metadata.max_visible_value, metadata.date, metadata.dataset, metadata.synthetic_method, metadata.coverture, 'B04', new_bands_IDs, new_c_band_min, new_c_band_max, flag_normalization)
    return new_composed_puzzle, new_composed_metadata



def save_as_slim_netCDF(base_3D_array, full_output_path, metadata, flag_revert_normalization=False, flag_show_3_bands=False):
    """
    Save the synthetized multiespectral image as a texture in a xarray labelled data
    structure, which is stored in a standarized netCDF file, so it will be used easily
    by the next algorithms in family 2.x
    Additional useful information is also recorded as metadata.

    Args:
        base_3D_array (np.array): multidimensional (multi-espectral) 3D array
        full_output_path (str): full path to save the file
        metadata (MultiChannelFileMetadata): It must be completely filled in
        flag_revert_normalization (bool): It forces the original dynamic range restoration

    Returns:
        The output is a file stored in the hard disk according to prior naming rules

    Warning:
        This function may potentially denormalize base_3D_array, so it is designed to be used 
        at the end of the execution
    """
    if flag_show_3_bands:
        plt.figure()
        plt.imshow(base_3D_array)
        plt.title('before de-normalization')
        plt.show()

    num_bands_in_patch = base_3D_array.shape[2]
    if (metadata is not None and metadata.flag_normalization) or flag_revert_normalization:
            # Let's recover the dynamic range
            for band_i in range(num_bands_in_patch):
                max_val = metadata.band_max[band_i]
                min_val = metadata.band_min[band_i]
                base_3D_array[:,:,band_i] = reverse_normalize_array(base_3D_array[:,:,band_i], max_val, min_val)
            metadata.flag_normalization = False

    composed_crop = xr.DataArray(base_3D_array, dims=("x", "y", "band"), coords={"band": metadata.bands_IDs})
    composed_crop.attrs["long_name"] = metadata.crop_type_name
    composed_crop.attrs["date"] = metadata.date
    composed_crop.attrs["dataset"] = metadata.dataset
    composed_crop.attrs["synthetic_method"] = metadata.synthetic_method
    composed_crop.attrs['max_visible_value'] = metadata.max_visible_value
    composed_crop.attrs["coverture"] = metadata.coverture

    if flag_show_3_bands:
        plt.figure()
        plt.imshow(composed_crop)
        plt.title('after de-normalization')
        plt.show()

    filename_nc = full_output_path  # f'.\{subset_str}\{date_str}_{crop_name}_{effective_res}.nc'
    composed_crop.to_netcdf(filename_nc)
    print(f'Saved :   {filename_nc}')



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

# Example usage (commented out to prevent execution):
# arr = np.array([[1, -10000], [5000, -9991]])





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


if __name__ == "__main__":  
    input_folder = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/CyL_and_Cat/'
    output_folder = 'C:/DMTA/projects/SD4EO/chaotic-aggregation/slim_puzzles/'
    # input_filename = '20180901_Peas_768.nc'
    norm_flag = True
    view_flag = False

    for input_filename in list_nc_files(input_folder):
        new_composed_puzzle, new_composed_metadata = extract_and_transform_bands(input_folder + input_filename, flag_normalization=norm_flag, flag_preview=view_flag)
        full_output_filename = output_folder + input_filename[:-3] + '_slim.nc'
        save_as_slim_netCDF(new_composed_puzzle, full_output_filename, new_composed_metadata, flag_revert_normalization=norm_flag, flag_show_3_bands=view_flag)
    

