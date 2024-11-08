# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     plotReplacement.py
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
# This is a small script to replace the pixel content inside the plot crops 
# and an additional margin surrounding them. These crops extensions will be
# replaced by the multiespectral synthetized textures, which has been obtained 
# in SD4EO
# We can choose the types of crop to be swapped and the target date 

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
from scipy.ndimage import binary_dilation
import time
from typing import Any, List
import xarray as xr

#from legacycode import *
from multichannelFileFormat import load_from_netCDF, save_as_netCDF, show_mcArray, MultiChannelFileMetadata, year_month_list, Romania_Crop_Codes, S2BAND10m_CODE, S2BAND20m_CODE,ALL_BAND_CODE, S_BAND_INDX

# Shared data structures

NON_VALID_VALUE = -9999
NAN_THRESHOLD = -9990

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

CyL_date_S2_list = ['20171009', '20171019', '20171029', '20171108', '20171118', '20171128', '20171208', '20171218', '20171223', '20180102', '20180107', '20180112', '20180117', '20180122', '20180127', '20180201', '20180206', '20180211', '20180216', '20180221', '20180226', '20180303', '20180308', '20180313', '20180318', '20180323', '20180328', '20180402', '20180407', '20180412', '20180417', '20180422', '20180427', '20180502', '20180507', '20180512', '20180517', '20180522', '20180527', '20180601', '20180606', '20180611', '20180616', '20180621', '20180626', '20180701', '20180706', '20180711', '20180716', '20180721', '20180726', '20180731', '20180810', '20180815', '20180820', '20180825', '20180830', '20180904', '20180909', '20180914', '20180919', '20180924', '20180929']


# Functions

def load_shapefile(filepath):
    """
    Load a shapefile into a GeoDataFrame.

    Args:
        filepath (str): The path to the shapefile.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the loaded shapefile data.
    """
    return gpd.read_file(filepath)


def extract_polygons_by_CyL_crop_type_ID(shapefile, crop_type_ID):
    """
    Extract polygons from a shapefile based on the crop type.

    Args:
        shapefile (GeoDataFrame): A GeoDataFrame containing the shapefile data.
        crop_type (int): The crop type ID to filter by.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the polygons for the specified crop type.
    """
    # Filter the GeoDataFrame based on the crop type
    filtered_data = shapefile[shapefile['CODE'] == crop_type_ID]  
    return filtered_data

def exhaustive_crop_search(list_crop_registers, key_code):
    crop_register = None
    for crop_reg_i in list_crop_registers:
        if crop_reg_i.uuid == key_code:
            crop_register = crop_reg_i
    return crop_register


def find_tif_file(directory_path):
    """
    Finds the first .tif file within the specified directory.
    
    Args:
        directory_path (str): The path to the directory where to search for the .tif file.
        
    Returns:
        str: The full path of the first .tif file found in the directory. If no .tif file is found,
             returns a message indicating that no .tif file was found.
    """
    # List all files and directories in the specified path
    for item in os.listdir(directory_path):
        # Check if the item is a file and ends with the .tif extension
        if os.path.isfile(os.path.join(directory_path, item)) and item.endswith('.tif'):
            # Return the full path of the .tif file
            return os.path.join(directory_path, item)
    
    # Return a message if no .tif file is found
    return "No .tif file found in the specified directory."

def extend_mc_texture(multispectral_texture, src_height, src_width, bandname, np_type):
    """
    This function takes the original texture and replicates it in both dimensions
    to cover the full geotiff domain. 
    It is conceptually equivalent to texture wrap with GL_REPEAT in OpenGL.
    """
    # It requires more RAM, but it is faster to code and less prone to errors
    mc_height = multispectral_texture.shape[0]
    mc_width = multispectral_texture.shape[1]
    band_indx = S_BAND_INDX[bandname]
    num_mc_multiple_height = int(np.ceil(src_height/mc_height))
    num_mc_multiple_width = int(np.ceil(src_width/mc_width))
    synth_extended_values = np.zeros((num_mc_multiple_height*mc_height, num_mc_multiple_width*mc_width), dtype=np_type)
    for h_i in range(num_mc_multiple_height):
        for w_i in range(num_mc_multiple_width):
            init_row = h_i*mc_height
            init_col = w_i*mc_width
            synth_extended_values[init_row:init_row+mc_height,init_col:init_col+mc_width] = multispectral_texture[:,:,band_indx]
    cut_synth_values = synth_extended_values[:src_height, :src_width]
    return cut_synth_values


def swap_rastered_polygons(unsorted_polygons, crs_shapefile, base_full_S1_GeoTIFF_filename, base_full_S2_GeoTIFF_filename, multispectral_texture, date_str, output_path, crop_name_orig, crop_name_dest, flag_display=False, flag_replace_clouds=False):
    """
    This function takes the multispectral texture and replaces the selected projected crops 
    in the geotiff file (the content in pixels) by the mapped flat texture on all the surface
    comprised by all the polygon crop plots of the selected type.
    We also increase the crop plot border by 1 surrounding pixel margin in order to avoid 
    later problems because slight misalignment or small numerical errors in geometrical
    transformations / sampling grids...
    """
    subset_str = 'CyLsynth'
    max_S2_num_bands = len(ALL_BAND_CODE)
    bandname_reference = S2BAND10m_CODE[0]
    
    rastered_polygons = CropDeck([], subset_str, date_str, bandname_reference)
    band_crs_raster = None
    band_projection = None

    key_for_crop_code = 'CODE'
    key_for_crop_area = 'Area_ha'
    key_for_crop_ID = 'uuid'

    # TODO: LOOP for all bands
    for bandname in ALL_BAND_CODE:
        if bandname in S2BAND10m_CODE+S2BAND20m_CODE:
            band_geotiff_full_path = base_full_S2_GeoTIFF_filename.format(bandname)
        else: # S1
            band_geotiff_full_path = base_full_S1_GeoTIFF_filename

        with rasterio.open(band_geotiff_full_path) as src:
            original_data = replace_crop_texture_in_band(unsorted_polygons, src, crs_shapefile, multispectral_texture,bandname, flag_replace_clouds=False)
            meta = src.meta

            # Write the modified raster data to the output GeoTIFF
            output_filename = f'SD4EO_Spain_Crops_CLM_{date_str}_from_{crop_name_orig}_to_{crop_name_dest}_{bandname}_10m.tif'
            with rasterio.open(output_path + output_filename, "w", **meta) as dest:
                dest.write(original_data)




def replace_crop_texture_in_band(unsorted_polygons, src, crs_shapefile, multispectral_texture,bandname, flag_replace_clouds=False):
    """
    This function takes the multispectral texture and replaces the selected projected crops 
    in the geotiff file (the content in pixels) by the mapped flat texture on all the surface
    comprised by all the polygon crop plots of the selected type.
    We also increase the crop plot border by 1 surrounding pixel margin in order to avoid 
    later problems because slight misalignment or small numerical errors in geometrical
    transformations / sampling grids...
    """
    # Let's check the reference system
    band_crs_raster = src.crs
    band_projection = src.transform
    if bandname in S2BAND10m_CODE:
        print(f"CRS GeoTIFF for 10m {bandname} band: {band_crs_raster}")
    elif bandname in S2BAND20m_CODE:
        print(f"CRS GeoTIFF for 20m {bandname} band: {band_crs_raster}")
    else: 
        print(f"CRS GeoTIFF for S1 VH polarization band: {band_crs_raster}")
    # If it's different, we transform it just in the polygons
    if crs_shapefile != band_crs_raster:
        polygons = unsorted_polygons.to_crs(band_crs_raster)
    else:
        polygons = unsorted_polygons

    # Read the original image data
    original_data = src.read()
    meta = src.meta

    # Create a single rasterized mask for all polygons
    shape_mask = rasterize(polygons.geometry, out_shape=(src.height, src.width),
                        fill=0, transform=band_projection, all_touched=True, dtype='uint8')

    # We enlarge the mask 1 pixel around in order to avoid border effects (besides, 
    # the crop plots were originally reduced 1 pixel to get pure -non contaminated- crop 
    # values of the selected type)
    dilated_shape_mask = dilate_binary_mask(shape_mask)

    # Compose the corresponding layer to fit in the original resolution of geotiff
    replacement_values = extend_mc_texture(multispectral_texture, src.height, src.width, bandname, meta['dtype'])

    # Apply the rasterized mask to modify the original data
    # We expect the number of layers to be the first (0) dimension
    # and replacement_values to contain only one band
    original_data[0][dilated_shape_mask == 1] = replacement_values[dilated_shape_mask == 1]
    if flag_replace_clouds:  # Crazy tick for simpler visualization
        cloud_mask = np.squeeze(original_data) < NAN_THRESHOLD
        original_data[0][cloud_mask] = replacement_values[cloud_mask]
    return original_data
    

def dilate_binary_mask(mask):
    """Dilates a binary mask by one pixel using binary dilation.

    Args:
        mask (np.ndarray): A 2D numpy array with binary values (0s and 1s), representing the mask.

    Returns:
        np.ndarray: A 2D numpy array of the same shape as input, where 1s have been dilated.
    """
    # Using the default structuring element for binary dilation which is connectivity-1 (cross-shaped)
    dilated_mask = binary_dilation(mask).astype(mask.dtype)
    return dilated_mask


def find_unknown_filename_portion(folder, prefix, suffix):
    """
    Find the file in the specified directory that starts with 'prefix' and ends with 'suffix',
    and return the middle part of the file name which is unknown.

    Args:
    folder (str): The directory to search in.
    prefix (str): The known starting part of the file name.
    suffix (str): The known ending part of the file name.

    Returns:
    str: The unknown middle part of the file name.

    Raises:
    FileNotFoundError: If no file matches the specified pattern.
    """
    # List all files in the given directory
    for filename in os.listdir(folder):
        # Check if the filename starts with the prefix and ends with the suffix
        if filename.startswith(prefix) and filename.endswith(suffix):
            # Calculate the start and end index of the unknown part
            start_index = len(prefix)
            end_index = len(filename) - len(suffix)
            # Return the substring which is the unknown part
            return filename[start_index:end_index]
    
    # Raise an error if no matching file is found
    raise FileNotFoundError("No file matches the specified prefix and suffix in the directory.")


def update_shapefile(shapefile_datastructure, original_crop_name, original_crop_code, new_crop_name, new_crop_code, new_shapefile_filename):
    """
    Updates the 'CODE' column in the shapefile data, replacing specified original values with a new value, and saves it as a new shapefile.
    
    Args:
    shapefile_datastructure (GeoDataFrame): The GeoDataFrame containing the shapefile data.
    original_crop_name (str): The original value in the 'Class' column to replace.
    original_crop_code (int or str): The original value in the 'CODE' column to replace.
    new_crop_name (str): The new name to replace the original value in the 'Class' column.
    new_crop_code (int or str): The new value to replace the original value in the 'CODE' column.
    new_shapefile_filename (str): The name of the new shapefile to be created with updates.
    
    Returns:
    shapefile_datastructure (GeoDataFrame): the updated shapefile data structure
    """
    # 'CODE': numerical code from the type of crop
    # Update the 'CODE' column where it matches the original_code
    shapefile_datastructure['CODE'] = shapefile_datastructure['CODE'].replace(original_crop_code, new_crop_code)
    # 'Class': string with the type of crop name
    # Update the 'Class' column where it matches the original_code
    shapefile_datastructure['Class'] = shapefile_datastructure['Class'].replace(original_crop_name, new_crop_name)
    # Save the updated GeoDataFrame as a new shapefile
    shapefile_datastructure.to_file(new_shapefile_filename)


def crop_swap_CyL(crop_index_orig, crop_index_dest, date_str, mainS2path, mainS1path, output_folder):
    """
    This function sweeps over all geotiff files for the indicated crop type and
    date; it localizes all the labelled crops according the shapefile and substitutes 
    the content by the synthetic texture pixels in a random location of the texture 
    domain. 
    Orientation is preserved (to maintain properties in radar signal).
    We fill in the content with an additional pixel margin, to avoid later any issue. 

    Args:
        crop_name (str): Crop name according to ROMANIA identifier 
        crop_type_ID (str): Crop type according to ROMANIA identifier. 
        date_str (str): Desired date (when the satellite took the images)
                        In the case of Sentinel 1 only the year-month are considered
        mainS2path (str): Input data path with slots for bands
        mainS1path (str): Input data path (we only take selected bands)
        output_folder (str): Folder where the geotiff files will be saved

    Returns:
        Modified geotiff files (S110m, S220m and SAR) are saved in the indicated folder


    """
    print('Start...')
    # Initialization
    crop_name_orig = Romania_Crop_Codes[crop_index_orig][0]
    crop_type_ID_orig = Romania_Crop_Codes[crop_index_orig][1]
    crop_name_dest = Romania_Crop_Codes[crop_index_dest][0]
    crop_type_ID_dest = Romania_Crop_Codes[crop_index_dest][1]
    month_year_str = full_date_str[:-2]
    geotiff_S2_base_full_filename = mainS2path + date_str +'/SD4EO_Spain_Crops_' + date_str + '_{}_CLM_10m.tif'
    geotiff_S1_base_full_path = mainS1path + month_year_str + '/'
    geotiff_S1_base_full_filename = find_tif_file(geotiff_S1_base_full_path)
        
    # Load the shapefile
    print('Load shapefile')
    shapefile_full_path = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/Fields_GTData/SD4EO_GTData_Training_32630.shp'  
    shapefile_gdf = load_shapefile(shapefile_full_path)
    # Extract polygons for a specific crop type, e.g., "Wheat"
    shapefile_vector_crop_polygons = extract_polygons_by_CyL_crop_type_ID(shapefile_gdf, crop_type_ID_orig)
    
    # Print the result or do further processing
    print(shapefile_vector_crop_polygons)

    # Let's extract the rastered polygon mask and the content for each crop, time sample and layer. It is not easy, because CRS must be compatible
    # Let's check the reference system
    crs_shapefile = shapefile_vector_crop_polygons.crs

    # Load multi-spectral texture
    folder_mc_texture = 'C:/DATASETs/OUT_HighOrder/'
    prefix_filename = f'HO_{crop_name_dest}_{date_str[:-2]}01_'
    suffix_filename = '_0.nc'
    print(f'   -->  {folder_mc_texture}{prefix_filename}*{suffix_filename}')
    middle_filename = find_unknown_filename_portion(folder_mc_texture, prefix_filename, suffix_filename)
    full_filename_mc_texture = prefix_filename + middle_filename + suffix_filename
    multispectral_texture, max_value, max_visible_value, metadata = load_from_netCDF(folder_mc_texture + full_filename_mc_texture)

    # Swap content inside the projected vector crop plot
    swap_rastered_polygons(shapefile_vector_crop_polygons, crs_shapefile,  geotiff_S1_base_full_filename, geotiff_S2_base_full_filename, multispectral_texture, date_str, output_folder, crop_name_orig, crop_name_dest, flag_display=False, flag_replace_clouds=False)

    # Save the corresponding shapefile in the same output folder
    output_shapefile_filename = output_folder + 'SD4EO_GTData_Training_modified.shp'
    update_shapefile(shapefile_gdf, crop_name_orig, crop_type_ID_orig, crop_name_dest, crop_type_ID_dest, output_shapefile_filename)






if __name__ == "__main__":  
    mc_texture_folder = 'C:/DATASETs/OUT_HighOrder/'
    Sentinel2_base_folder = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/S2Images_Oct2017-Sep2018_T30TUM/S2_CroppedAOI_WithCloudMask/'
    Sentinel1_base_folder = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/S1Images_Oct2017-Sep2018_T30TUM/Monthly-Composites/'
    output_base_folder = 'C:/DATASETs/OUT_SwappedSynthCrops/'
    
    index_date = 0       # Index for the selected date 
    crop_index_orig = 7  # Type of crop to be replaced, the one we will substitute
    crop_index_dest = 6  # Type of crop of the multiespectral texture we are going 
                         # to use to replace the original crop 
    full_date_str = CyL_date_S2_list[index_date]

    crop_swap_CyL(crop_index_orig, crop_index_dest, full_date_str, Sentinel2_base_folder, Sentinel1_base_folder, output_base_folder)
















