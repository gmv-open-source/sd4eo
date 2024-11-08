# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     plotExploration.py
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
# This is a small script to study plot boundaries and labels

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

# CyL shapefile fields:
# --------------------
# ['CODE', 'Samples', 'Class', 'Area_ha', 'uuid', 'geometry']
# 'CODE': numerical code from the type of crop
# 'Class': string with the type of crop name

# Catalonia shapefile fields
# --------------------------
# ['Campanya', 'Provincia', 'Comarca', 'Municipi', 'ID_MUN', 'Grup', 'Cultiu', 'Seca_Regad', 'HA', 'BUFF_DIST', 'ORIG_FID', 'geometry']


# ROMANIA CROP CODES
# ------------------
# Barley : 5
# Wheat : 1
# Other grain leguminous : 39
# Peas : 40
# Fallow & Bare soil : 21
# Vetch : 52
# Alfalfa or lucerne : 60
# Sunflower : 33
# Oats : 8
# En total hay 27 tipos de cultivo en este dataset: [1, 4, 5, 6, 7, 8, 10, 21, 33, 35, 39, 40, 52, 60, 61, 82, 85, 90, 94, 100, 101, 104, 105, 193, 202, 203, 204]

# CATALONIA CROP CODES
# ------------------
# Barley : 'ORDI'
# Wheat : 'BLAT TOU', 'BLAT DUR', 'BLAT KHORASAN'
# Other grain leguminous : 'CIGRONS SIE', 'LLENTIES SIE', 'ERBS (LLEGUM) SIE', 'CIGRONS NO SIE', 'MONGETA SIE', 'MONGETA NO SIE', 'LLENTIES NO SIE', 'ERBS (LLEGUM) NO SIE', 'GUIXES NO SIE O PEDREROL NO S*', 'GUIXES SIE O PEDREROL SIE'
# Peas : 'PÃˆSOLS SIE', 'PÃˆSOLS NO SIE'
# Fallow & Bare soil : 'GUARET NO SIE/ SUP. LLIURE SE*', 'GUARET SIE/ SUP. LLIURE SEMBRA', 'GUARET SIE AMB ESPÃˆCIES MEL.L*'
# Vetch : 'VECES NO SIE', 'VECES SIE'
# Alfalfa or lucerne : 'ALFALS SIE', 'ALFALS NO SIE'
# Sunflower : 'GIRA-SOL'
# Oats : 'CIVADA'


convertCataloniaCodetoCyL = {
    'GUARET NO SIE/ SUP. LLIURE SE*': 21,
    'ORDI': 5,
    'CIVADA': 8,
    'GUARET SIE/ SUP. LLIURE SEMBRA': 21,
    'PÃˆSOLS SIE': 40,
    'BLAT TOU': 1,
    'VECES SIE': 52,
    'ALFALS SIE': 60,
    'PÃˆSOLS NO SIE': 40,
    'VECES NO SIE': 52,
    'ALFALS NO SIE': 60,
    'CIGRONS SIE': 39,
    'LLENTIES SIE': 39,
    'ERBS (LLEGUM) SIE': 39,
    'CIGRONS NO SIE': 39,
    'MONGETA SIE': 39,
    'MONGETA NO SIE': 39,
    'LLENTIES NO SIE': 39,
    'GIRA-SOL': 33,
    'ERBS (LLEGUM) NO SIE': 39, 
    'BLAT DUR': 1,
    'BLAT KHORASAN': 1,
    'GUIXES NO SIE O PEDREROL NO S*': 39,
    'GUIXES SIE O PEDREROL SIE': 39,
    'GUARET SIE AMB ESPÃˆCIES MEL.L*': 21
    }

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

def get_column_names(gdf):
    """
    Get the names of the columns in a GeoDataFrame or DataFrame.

    Args:
        gdf (GeoDataFrame or DataFrame): The GeoDataFrame or DataFrame from which to extract column names.

    Returns:
        list: A list containing the names of the columns.
    """
    return gdf.columns.tolist()

def get_elements_for_shapefile_column(gdf, field_str):
    """
    Get all unique elements in a column in a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame containing the crop type data.
        field_str (str): name of field

    Returns:
        numpy.ndarray: An array of unique crop types.

    """
    if field_str in gdf.columns:
        return gdf[field_str].unique()
    else:
        raise ValueError(f"The GeoDataFrame does not contain a '{field_str}' column.")



def exhaustive_crop_search(list_crop_registers, key_code):
    crop_register = None
    for crop_reg_i in list_crop_registers:
        if crop_reg_i.uuid == key_code:
            crop_register = crop_reg_i
    return crop_register


def initial_S2_analysis_on_rastered_polygons(unsorted_polygons, crs_shapefile,  base_full_GeoTIFF_filename, bandname_10m, bandname_20m, subset_str, date_str, flag_display=False):
    """
    Analyzes rastered poligons in a subset of GeoTIFF key bands: 10m & 20m and creates the initialized list of CropRegisterClass for thoise polygons

    Args:
        unsorted_polygons (shapefile geopandas table): polygons 
                    of the selected crop type, there may be more crops in the
                    shapefile than in the sorted_rastered_polygons, because they 
                    are not present in this tile or they were discarded by size
        crs_shapefile (crs): CRS del shapefile
        base_full_GeoTIFF_filename (str): path of S2 geotiff filename with a slot
                    to indicate the band name
        bandname_10m (str): reference 10m band name
        bandname_20m (str): auxiliar 20m band to check projections
        subset_str (str): identifier of the dataset
        date_str (str): existing date for satellite data
        flag_display (bool):

    Returns:
        An initialized list CropDeck (of CropRegisterClass elements)

    Warning: 
        It must be called before many other funtions
    """
    max_S2_num_bands = len(ALL_BAND_CODE)
    rastered_polygons = CropDeck([],subset_str, date_str, bandname_10m)
    band10m_crs_raster = None
    band10m_projection = None
    if subset_str == 'CyL':
        band10m_geotiff_full_path = base_full_GeoTIFF_filename.format(bandname_10m)
        band20m_geotiff_full_path = base_full_GeoTIFF_filename.format(bandname_20m)
        key_for_crop_code = 'CODE'
        key_for_crop_area = 'Area_ha'
        key_for_crop_ID = 'uuid'
    else: # In catalonia dataset, we only have the start of the filename
        band10m_geotiff_full_path = find_incompletefile_path(base_full_GeoTIFF_filename, bandname_10m)
        band20m_geotiff_full_path = find_incompletefile_path(base_full_GeoTIFF_filename, bandname_20m)
        key_for_crop_code = 'Cultiu'
        key_for_crop_area = 'HA'
        key_for_crop_ID = 'ORIG_FID'

    # Step 1: First sweep with the 10m/pixel reference GeoTIFF file
    # print(f'fichero parcial: >>{base_full_GeoTIFF_filename}<<  >>{bandname_10m}<<')
    # print(f'fichero completo: >>{band10m_geotiff_full_path}<<')
    with rasterio.open(band10m_geotiff_full_path) as src:
        # Let's check the reference system
        band10m_crs_raster = src.crs
        band10m_projection = src.transform
        print(f"CRS GeoTIFF for 10m band: {band10m_crs_raster}")
        # If it's different, we transform it just in the polygons
        if crs_shapefile != band10m_crs_raster:
            polygons = unsorted_polygons.to_crs(band10m_crs_raster)
        else:
            polygons = unsorted_polygons

        # For each polygon in the shapefile
        for _, polygon in polygons.iterrows():
            crop_internal_index = int(_)
            crop_raw_code = polygon[key_for_crop_code]
            if subset_str == 'CyL':
                crop_type_ID = crop_raw_code
            else: # Catalonia dataset
                crop_type_ID = convertCataloniaCodetoCyL[crop_raw_code]
            crop_area_ha = polygon[key_for_crop_area]
            crop_uuid = polygon[key_for_crop_ID]
            # Extract the portion of the image corresponding to the polygon
            try:
                out_image, out_transform = mask(src, [polygon['geometry']], crop=True)
            except:
                # If input shapes do not overlap raster, 
                # then we do not consider this crop for this set
                continue 
            # Convert the image portion to a 2D numpy array
            np_out_image_array = np.squeeze(np.array(out_image.data)) # 2D
            if len(np_out_image_array.shape) < 2:
                # We do not support vector (1D) crops
                continue
            crop_b10m_rows = np_out_image_array.shape[0]
            crop_b10m_cols = np_out_image_array.shape[1]
            # Initialze the 3D array
            crop_array3D = np.zeros((crop_b10m_rows, crop_b10m_cols, max_S2_num_bands))
            crop_array3D[:,:,S_BAND_INDX[bandname_10m]] = np_out_image_array[:,:]
            # Check if there is some crop pixels below the cloud
            crop_mask = np_out_image_array > NAN_THRESHOLD
            crop_area_px = np.count_nonzero(crop_mask)
            if crop_area_px > MIN_CROP_SIZE_PIXELS:  # We only consider minimum sized crops in 10m band, not just a pair of scattered pixels
                # Let's see each image portion 
                if flag_display:
                    plt.imshow(np.squeeze(np_out_image_array), interpolation='none')
                    plt.colorbar()
                    plt.title(f'Crop  # {_} : {crop_area_px} px')
                    plt.show()
                # Only in the reference 10m band mask we calculate the inner crop distance 
                distance_on_mask = calculate_crop_distance_transform(np_out_image_array)
                if flag_display:
                    plt.imshow(distance_on_mask, interpolation='none')
                    plt.colorbar()
                    plt.title(f'Distance  # {_} : {crop_area_px} px')
                    plt.show()
                crop_register = CropRegisterClass(crop_internal_index, crop_uuid, crop_type_ID, crop_area_ha, crop_area_px, crop_b10m_rows, crop_b10m_cols, crop_mask, distance_on_mask, crop_array3D, None)
                rastered_polygons.crop_in_region.append(crop_register)

    # Step 2: Now we analyse the 20m/pixel band reference GeoTIFF file 
    with rasterio.open(band20m_geotiff_full_path) as src:
        # Let's check the reference system
        band20m_crs_raster = src.crs
        band20m_projection = src.transform
        print(f"CRS GeoTIFF for 20m band: {band20m_crs_raster}")
        # If it's different, we transform it just in the polygons
        if crs_shapefile != band20m_crs_raster:
            polygons = unsorted_polygons.to_crs(band20m_crs_raster)
        else:
            polygons = unsorted_polygons
        # Check if CRS is different between 10m and 20m bands
        if band10m_crs_raster != band20m_crs_raster:
            print("WARNING: 10m and 20m CRS bands are DIFFERENT")
        if band10m_projection != band20m_projection:
            print("WARNING: 10m and 20m projections are DIFFERENT")
            print("· 10m band projection:")
            print(band10m_projection)
            print("· 20m band projection:")
            print(band20m_projection)
        # Revise each polygon in the shapefile and meet the correspondence in same order rastered_polygons CropDeck
        num_discarded_large_crops = 0
        num_registered_crops = len(rastered_polygons.crop_in_region)
        if num_registered_crops == 0:
            print ("************************************************")
            print ("Severe Error: NO CROPS OF THIS TYPE IN THIS TILE")
            print ("************************************************")
            return None, None
        else:
            for _, polygon in polygons.iterrows():
                # The main objective is to understand which simple strategy with rasterio.mask allow us to fit the rasterized polygon into the 2D array limits for each 20m band
                crop_register = exhaustive_crop_search(rastered_polygons.crop_in_region, polygon[key_for_crop_ID])
                if crop_register is None:
                    # Not present in this region 
                    continue
                # 2.2) Let's start by extracting the portion of the image corresponding to the polygon with default values
                b20m_out_image, out_transform = mask(src, [polygon['geometry']], crop=True)
                # Convert the image portion to a 2D numpy array
                np_b20m_out_image_array = np.squeeze(np.array(b20m_out_image.data)) # 2D
                orig_crop_b20m_rows = np_b20m_out_image_array.shape[0]
                orig_crop_b20m_cols = np_b20m_out_image_array.shape[1]
                # 2.3) Are same dimensions?
                if orig_crop_b20m_rows == crop_register.b10m_rows and orig_crop_b20m_cols == crop_register.b10m_cols:
                    crop_register.b20m_margin_ID = b20mFIT.PERFECT_MATCH
                    continue
                # 2.4) Is it big enough?
                elif orig_crop_b20m_rows > crop_register.b10m_rows and orig_crop_b20m_cols > crop_register.b10m_cols:
                    crop_register.b20m_margin_ID = b20mFIT.BIGGER_THAN_10m
                    continue
                # Then, 20m band array is smaller in at least 1 dimension
                # 2.5) Let's pad it
                pad_b20m_out_image, out_transform = mask(src, [polygon['geometry']], pad= True, crop=True)
                pad_crop_b20m_rows = pad_b20m_out_image.shape[1]
                pad_crop_b20m_cols = pad_b20m_out_image.shape[2]
                if pad_crop_b20m_rows >= crop_register.b10m_rows and pad_crop_b20m_cols >= crop_register.b10m_cols:
                    crop_register.b20m_margin_ID = b20mFIT.PAD_NEEDED
                    continue
                # 2.6) Let's increase the way pixels are taken into account
                alltouched_b20m_out_image, out_transform = mask(src, [polygon['geometry']], all_touched= True, crop=True)
                alltouched_crop_b20m_rows = alltouched_b20m_out_image.shape[1]
                alltouched_crop_b20m_cols = alltouched_b20m_out_image.shape[2]
                if alltouched_crop_b20m_rows >= crop_register.b10m_rows and alltouched_crop_b20m_cols >= crop_register.b10m_cols:
                    crop_register.b20m_margin_ID = b20mFIT.ALLTOUCHED_NEEDED
                    continue
                # 2.7) We avoid to combine both all touched and pad as it may introduce large distortions
                crop_register.b20m_margin_ID = b20mFIT.NOT_VALID
                num_discarded_large_crops = num_discarded_large_crops + 1
        # At this point we should have a complete analysis for both kinds of bands and we get an initialized list of CropRegisterClass.
        # In order to speed up later computation, we also prepare a lookup table to access the indexes of the list sorted by actual pixel area
        unsorted_list_of_crop_regs = []
        for orig_index, crop in enumerate(rastered_polygons.crop_in_region):
            unsorted_list_of_crop_regs.append((orig_index, crop.area_px))
        sorted_list_of_index_crop_regs = sorted(unsorted_list_of_crop_regs, key=lambda x: x[1], reverse=True)
        if num_discarded_large_crops:
            print(f"Num discarded crops : {num_discarded_large_crops}")
        # print(sorted_list_of_index_crop_regs)
        return rastered_polygons, sorted_list_of_index_crop_regs
    


def combine_masked_arrays(normal_crop_image, inverted_crop_image, local_mask):
    """
    Combine two 2D numpy arrays based on validity indicated by a mask array.
    
    Parameters:
    - normal_crop_image: numpy array containing the first set of values.
    - inverted_crop_image: numpy array of the same size as normal_crop_image, containing the second set of values.
    - local_mask: boolean numpy array indicating valid values in normal_crop_image.
    
    Returns:
    - combined_array: numpy array with combined values from both input arrays.
    """  
    # Invert the mask to use for selecting values from the inverted_crop_image
    inverted_mask = ~local_mask
    
    # Initialize the output array with the same shape as the input arrays
    combined_array = np.empty_like(normal_crop_image)
    
    # Fill in values from the normal_crop_image where the mask is True
    combined_array[local_mask] = normal_crop_image[local_mask]
    
    # Fill in values from the inverted_crop_image where the inverted mask is True (original mask is False)
    combined_array[inverted_mask] = inverted_crop_image[inverted_mask]
    
    return combined_array


def simple_count_below_threshold(array, threshold=NAN_THRESHOLD):
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


def masked_count_below_threshold(matrix, mask, threshold=NAN_THRESHOLD):
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


def fill_content_for_allS2_bands(subset_str, unsorted_rastered_polygons, sorted_list_of_index_crop_regs, shapefile_unsorted_polygons, crs_shapefile, base_full_GEoTIFF_filename):
    """
    This function fills all the missing bands of Sentinel 2 in the corresponding 3D array

    Args:
        subset_str (str): identifier of the dataset
        unsorted_rastered_polygons (CropDeck): a cropdeck already initialized, 
                    it should contain the refreence band content (usually B02)
        sorted_list_of_index_crop_regs (list->tuple): This list of pairs with
                    indexes is used to sort the resulting crop_deck
        shapefile_unsorted_polygons (shapefile geopandas table): polygons 
                    of the selected crop type, there may be more coprs in the
                    shapefile than in the sorted_rastered_polygons, because they 
                    are not present in this tile or they were discarded by size
        crs_shapefile (crs): CRS del shapefile
        base_full_GEoTIFF_filename (str): path of S2 geotiff filename with a slot
                    to indicate the band name

    Returns:
        sorted_rastered_polygons (CropDeck): the final crop_deck with the 
                Sentinel 2 band content in it, and sorted by pixel-size of the crops
    """
    if subset_str == 'CyL':
        key_for_crop_ID = 'uuid'
    else: # In catalonia dataset, we only have the start of the filename
        key_for_crop_ID = 'ORIG_FID' 

    # First, 10m bands
    for band_10m_name in S2BAND10m_CODE:
        if band_10m_name == unsorted_rastered_polygons.reference_band:
            # Already done in the initializatino step, we continue with other band
            continue

        if subset_str == 'CyL':
            band10m_geotiff_full_path = base_full_GEoTIFF_filename.format(band_10m_name)
        else: # In catalonia dataset, we only have the start of the filename
            band10m_geotiff_full_path = find_incompletefile_path(base_full_GEoTIFF_filename, band_10m_name)     

        with rasterio.open(band10m_geotiff_full_path) as src:
            # Let's check the reference system
            band10m_crs_raster = src.crs
            print(f"CRS GeoTIFF for {band_10m_name} 10m band: {band10m_crs_raster}")
            # If it's different, we transform it just in the polygons
            if crs_shapefile != band10m_crs_raster:
                polygons = shapefile_unsorted_polygons.to_crs(band10m_crs_raster)
            else:
                polygons = shapefile_unsorted_polygons

            # For each polygon in the shapefile
            num_registered_crops = len(unsorted_rastered_polygons.crop_in_region)
            for _, polygon in polygons.iterrows():
                crop_register = exhaustive_crop_search(unsorted_rastered_polygons.crop_in_region, polygon[key_for_crop_ID])
                if crop_register is None:
                    # Not present in this region 
                    continue

                # Extract the portion of the image corresponding to the polygon
                incomple_crop_image, out_transform = mask(src, [polygon['geometry']], crop=True)
                # Convert the image portion to a 2D numpy array
                np_out_image_array = np.squeeze(np.array(incomple_crop_image.data)) # 2D
                crop_b10m_rows = np_out_image_array.shape[0]
                crop_b10m_cols = np_out_image_array.shape[1]                
                # We check it has the same size as the 10m reference band
                if crop_register.b10m_rows != crop_b10m_rows or crop_register.b10m_cols != crop_b10m_cols:
                    print(f"ERROR: Serius error mistmatch in 10m band {band_10m_name} for polygon # {_}. Expected size {crop_register.b10m_rows}x{crop_register.b10m_cols} and real size {crop_b10m_rows}x{crop_b10m_cols}")
                    # TODO: Exception !!
                # We insert the band content
                crop_register.array3D[:,:,S_BAND_INDX[band_10m_name]] = np_out_image_array[:,:]
    # Second, 20m bands
    for band_20m_name in S2BAND20m_CODE:

        if subset_str == 'CyL':
            band20m_geotiff_full_path = base_full_GEoTIFF_filename.format(band_20m_name)
        else: # In catalonia dataset, we only have the start of the filename
            band20m_geotiff_full_path = find_incompletefile_path(base_full_GEoTIFF_filename, band_20m_name)

        with rasterio.open(band20m_geotiff_full_path) as src:
            # Let's check the reference system
            band20m_crs_raster = src.crs
            print(f"CRS GeoTIFF for {band_20m_name} 20m band: {band20m_crs_raster}")
            # If it's different, we transform it just in the polygons
            if crs_shapefile != band20m_crs_raster:
                polygons = shapefile_unsorted_polygons.to_crs(band20m_crs_raster)
            else:
                polygons = shapefile_unsorted_polygons

            # For each polygon in the shapefile
            num_registered_crops = len(unsorted_rastered_polygons.crop_in_region)
            for _, polygon in polygons.iterrows():
                crop_register = exhaustive_crop_search(unsorted_rastered_polygons.crop_in_region, polygon[key_for_crop_ID])
                if crop_register is None:
                    # Not present in this region 
                    continue

                # Let's proceed according the designated strategy to obtain pixels in all canvas space
                # array_non_valid_values = np.ones((crop_register.b10m_rows, crop_register.b10m_cols))*NON_VALID_VALUE
                match crop_register.b20m_margin_ID:
                    case b20mFIT.PERFECT_MATCH:
                        normal_crop_image, out_transform = mask(src, [polygon['geometry']], crop=True, filled=False)
                        normal_image_array = np.squeeze(np.array(normal_crop_image.data)) # 2D
                        crop_register.array3D[:,:,S_BAND_INDX[band_20m_name]] = normal_image_array[:,:]
                    case b20mFIT.BIGGER_THAN_10m:
                        normal_crop_image, out_transform = mask(src, [polygon['geometry']], crop=True, filled=False)
                        normal_image_array = np.squeeze(np.array(normal_crop_image.data)) # 2D
                        crop_register.array3D[:,:,S_BAND_INDX[band_20m_name]] = normal_image_array[:crop_register.b10m_rows,:crop_register.b10m_cols]
                    case b20mFIT.PAD_NEEDED:
                        normal_crop_image, out_transform = mask(src, [polygon['geometry']], crop=True, pad=True, filled=False)
                        normal_image_array = np.squeeze(np.array(normal_crop_image.data)) # 2D
                        crop_register.array3D[:,:,S_BAND_INDX[band_20m_name]] = normal_image_array[:crop_register.b10m_rows,:crop_register.b10m_cols]
                    case b20mFIT.ALLTOUCHED_NEEDED:
                        normal_crop_image, out_transform = mask(src, [polygon['geometry']], crop=True, all_touched=True, filled=False)
                        normal_image_array = np.squeeze(np.array(normal_crop_image.data)) # 2D
                        crop_register.array3D[:,:,S_BAND_INDX[band_20m_name]] = normal_image_array[:crop_register.b10m_rows,:crop_register.b10m_cols]  
                # Re-check the mask2D and update it if needed
                num_non_valid_values_in_20m = simple_count_below_threshold(crop_register.array3D[:,:,S_BAND_INDX[band_20m_name]])
                if num_non_valid_values_in_20m > 0:
                    # We can have a problem because of sneaking cloud masks
                    sneacked_cloud_pixels = masked_count_below_threshold(crop_register.array3D[:,:,S_BAND_INDX[band_20m_name]], crop_register.mask2D)
                    if sneacked_cloud_pixels > 0:
                        # We need to update the mask2D
                        mask_20m = crop_register.array3D[:,:,S_BAND_INDX[band_20m_name]] > NAN_THRESHOLD
                        combined_mask = mask_20m & crop_register.mask2D
                        crop_register.mask2D = combined_mask
                        # We must also update the distance mask !!
                        crop_register.distance_map_2D = calculate_crop_distance_transform_from_inv_binary_mask(~combined_mask)

                # Apply mask on the copied "slice"
                # --> Commented because it produces a seldom slight gap 
                # masked_output = combine_masked_arrays(crop_register.array3D[:,:,S_BAND_INDX[band_20m_name]], array_non_valid_values, crop_register.mask2D)
                # crop_register.array3D[:,:,S_BAND_INDX[band_20m_name]] = masked_output

    # Third, we prepare a new CropDeck list with valid polygons sorted by size
    ### unsorted_rastered_polygons, sorted_list_of_index_crop_regs
    sorted_rastered_polygons = CropDeck([],unsorted_rastered_polygons.AOI_ID, unsorted_rastered_polygons.date_str, unsorted_rastered_polygons.reference_band)
    for (index, area) in sorted_list_of_index_crop_regs:
        candidate_crop_register = unsorted_rastered_polygons.crop_in_region[index]
        if candidate_crop_register.b20m_margin_ID != b20mFIT.NOT_VALID:
                sorted_rastered_polygons.crop_in_region.append(candidate_crop_register)
    return sorted_rastered_polygons


def fill_content_for_S1_bands(subset_str, sorted_rastered_polygons, shapefile_unsorted_polygons, crs_shapefile, geotiff_S1_base_full_filename):
    """
    This function fills in the layer of Sentinel 1 in the corresponding 3D array

    Args:
        subset_str (str): identifier of the dataset
        sorted_rastered_polygons (CropDeck): a cropdeck of the already selected 
                    and sorted crop_registers with all the content of the bands 
                    but the missing content of the S1 polarizations
        shapefile_unsorted_polygons (shapefile geopandas table): polygons 
                    of the selected crop type, there may be more coprs in the
                    shapefile than in the sorted_rastered_polygons, because they 
                    are not present in this tile or they were discarded by size
        crs_shapefile (crs): CRS del shapefile
        geotiff_S1_base_full_filename (str): full path of S1 geotiff filename
                    (or at least what it is needed to locate it)

    Returns:
        sorted_rastered_polygons (CropDeck): the final crop_deck with the last
                    slices of the arrays filled in with the Sentinel 1 content

    Note:
        For now we suppose only one Sentinel 1 GeoTIFF file is in the folder,
        we conly support HV polarization by petition of NSL client
    """
    if subset_str == 'CyL':
        key_for_crop_ID = 'uuid'
    else: # In catalonia dataset, we only have the start of the filename
        key_for_crop_ID = 'ORIG_FID'

    for band_S1_name in S1BAND_CODE:
        with rasterio.open(geotiff_S1_base_full_filename) as src:
            # Let's check the reference system
            S1_crs_raster = src.crs
            print(f"CRS GeoTIFF for {band_S1_name} S1 band: {S1_crs_raster}")
            # If it's different, we transform it just in the polygons
            if crs_shapefile != S1_crs_raster:
                polygons = shapefile_unsorted_polygons.to_crs(S1_crs_raster)
            else:
                polygons = shapefile_unsorted_polygons
       
            num_registered_crops = len(sorted_rastered_polygons.crop_in_region)
            for _, polygon in polygons.iterrows():
                # For each polygon in the shapefile we look for the corresponding one in the CropDeck
                # If it is not listed, we skip it
                found_crop_flag = False
                for crop_deck_indx in range(num_registered_crops):
                    crop_register = sorted_rastered_polygons.crop_in_region[crop_deck_indx]
                    if (crop_register.uuid == polygon[key_for_crop_ID]):
                        found_crop_flag = True
                        break
                if found_crop_flag == False:
                    continue
                #) Analyze how the S1 mask area fits in S2 20m band room
                # array_non_valid_values = np.ones((crop_register.b10m_rows, crop_register.b10m_cols))*NON_VALID_VALUE
                # 2.1) Let's start by extracting the portion of the image corresponding to the polygon with default values
                S1_out_image, out_transform = mask(src, [polygon['geometry']], crop=True, filled=False)
                # Convert the image portion to a 2D numpy array
                np_S1_out_image_array = np.squeeze(np.array(S1_out_image.data)) # 2D
                orig_crop_S1_rows = np_S1_out_image_array.shape[0]
                orig_crop_S1_cols = np_S1_out_image_array.shape[1]
                # 2.3) Are same dimensions?
                if orig_crop_S1_rows == crop_register.b10m_rows and orig_crop_S1_cols == crop_register.b10m_cols:
                    crop_register.S1_margin_ID = b20mFIT.PERFECT_MATCH
                    crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]] = np_S1_out_image_array[:,:]
                    # masked_output = combine_masked_arrays(crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]], array_non_valid_values, crop_register.mask2D)
                    # crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]] = masked_output
                    # debug_view(np_S1_out_image_array, crop_register.S1_margin_ID) 
                    continue
                # 2.4) Is it big enough?
                elif orig_crop_S1_rows > crop_register.b10m_rows and orig_crop_S1_cols > crop_register.b10m_cols:
                    crop_register.S1_margin_ID = b20mFIT.BIGGER_THAN_10m
                    crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]] = np_S1_out_image_array[:crop_register.b10m_rows,:crop_register.b10m_cols]
                    # masked_output = combine_masked_arrays(crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]], array_non_valid_values, crop_register.mask2D)
                    # crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]] = masked_output
                    # debug_view(np_S1_out_image_array, crop_register.S1_margin_ID) 
                    continue
                # Then, S1 band array is smaller in at least 1 dimension
                # 2.5) Let's pad it
                pad_S1_out_image, out_transform = mask(src, [polygon['geometry']], pad= True, crop=True, filled=False)
                pad_crop_S1_rows = pad_S1_out_image.shape[1]
                pad_crop_S1_cols = pad_S1_out_image.shape[2]
                if pad_crop_S1_rows >= crop_register.b10m_rows and pad_crop_S1_cols >= crop_register.b10m_cols:
                    np_pad_S1_out_image = np.squeeze(np.array(pad_S1_out_image.data)) # 2D
                    crop_register.S1_margin_ID = b20mFIT.PAD_NEEDED
                    crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]] = np_pad_S1_out_image[:crop_register.b10m_rows,:crop_register.b10m_cols]
                    # masked_output = combine_masked_arrays(crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]], array_non_valid_values, crop_register.mask2D)
                    # crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]] = masked_output
                    # debug_view(np_pad_S1_out_image, crop_register.S1_margin_ID) 
                    continue
                # 2.6) Let's increase the way pixels are taken into account
                alltouched_S1_out_image, out_transform = mask(src, [polygon['geometry']], all_touched= True, crop=True, filled=False)
                alltouched_crop_S1_rows = alltouched_S1_out_image.shape[1]
                alltouched_crop_S1_cols = alltouched_S1_out_image.shape[2]
                if alltouched_crop_S1_rows >= crop_register.b10m_rows and alltouched_crop_S1_cols >= crop_register.b10m_cols:
                    np_alltouched_S1_out_image = np.squeeze(np.array(pad_S1_out_image.data)) # 2D
                    crop_register.S1_margin_ID = b20mFIT.ALLTOUCHED_NEEDED
                    crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]] = np_alltouched_S1_out_image[:crop_register.b10m_rows,:crop_register.b10m_cols]
                    # masked_output = combine_masked_arrays(crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]], array_non_valid_values, crop_register.mask2D)
                    # crop_register.array3D[:,:,S_BAND_INDX[band_S1_name]] = masked_output
                    # debug_view(np_alltouched_S1_out_image, crop_register.S1_margin_ID)                   
                    continue
    return sorted_rastered_polygons

def debug_view(array2D, state):
    match state:
        case b20mFIT.PERFECT_MATCH:
            title = 'PERFECT_MATCH'
        case b20mFIT.BIGGER_THAN_10m:
            title = 'BIGGER_THAN_10m'
        case b20mFIT.PAD_NEEDED:
            title = 'PAD_NEEDED'            
        case b20mFIT.ALLTOUCHED_NEEDED:
            title = 'ALLTOUCHED_NEEDED'
    if state == b20mFIT.ALLTOUCHED_NEEDED:
        plt.figure()
        plt.imshow(array2D)
        plt.title(title)
        plt.show()


def extract_rastered_polygons(unsorted_polygons, crs_shapefile, mainS2path, base_filename, flag_sort_by_area = False, flag_save_tif_files=False, flag_display=False):
    """
    Extract rastered poligons from GeoTIFF and shapefile polygons

    Args:
        unsorted_polygons (shapefile): original poligons from shapefile table
        crs_shapefile (crs): CRS of the shapefile
        mainS2path (str): base path for the folder we need to explore
        base_filename (str): string with the content we need to locate the geotiff file
        flag_sort_by_area (bool): If we want to sort the output
        flag_save_tif_files (bool): If we want to save intermediate images 
                                    to check output (just for debugging purposes)
        flag_display (bool): Display the crop shapes 

    Returns:
        Two lists: 
        rastered_polygons : one with rasterized crops (should be all bands !)
        inner_distance_crops : one with corresponding distance inner crop transform
    """

    # Step 1: Open the GeoTIFF file with rasterio
    rastered_polygons = []
    rastered_polygons_for_this_band = []
    inner_distance_crops = []
    max_num_bands = len(ALL_BAND_CODE)
    crs_raster = None
    master_projection = None

    for band_index in range (max_num_bands):
        band_name = ALL_BAND_CODE[band_index]
        geotiff_full_path = mainS2path + f"{date_str}/{base_filename}_{date_str}_{band_name}_CLM_10m.tif"
        with rasterio.open(geotiff_full_path) as src:
            # Let's check the reference system
            if band_index == 0: # We consider all as the same projection
                crs_raster = src.crs
                master_projection = src.transform
            # else:
            #     src.transform = master_projection
            print(f"CRS del GeoTIFF: {crs_raster}")
            # If it's different, we transform it just in the polygons
            if crs_shapefile != crs_raster:
                projected_polygons = unsorted_polygons.to_crs(crs_raster)
            else:
                projected_polygons = unsorted_polygons

            flag_sort_by_area = False
            if flag_sort_by_area:  # It is note working as expected
                # Calculate the area of each polygon and store it in a new column
                projected_polygons['Area_ha'] = projected_polygons.area
                # Sort the GeoDataFrame by the area column from largest to smallest
                polygons = projected_polygons.sort_values(by='Area_ha', ascending=False)
            else:
                polygons = projected_polygons

            # For each polygon in the shapefile
            new_index = 0
            sorted_crop_index = 0
            num_errors = 0
            for _, polygon in polygons.iterrows():
                # Step 3: Extract the portion of the image corresponding to the polygon
                out_image, out_transform = mask(src, [polygon['geometry']], crop=True)
                # Convert the image portion to a numpy array
                out_image_array = out_image.data
                
                # # Is it a numpy array ?
                # print(out_image.shape)
                # print(out_image)
                # print(np.max(np.max(out_image_array)))

                # We could save the image portion as a new GeoTIFF file
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image_array.shape[1],
                    "width": out_image_array.shape[2],
                    "transform": out_transform
                })

                # Check if there is some crop pixels below the cloud
                np_out_image_array = np.array(out_image_array)
                dummy_crop_mask = np.squeeze(np_out_image_array[0,:,:]) > NAN_THRESHOLD
                crop_num_pixels = np.count_nonzero(dummy_crop_mask)
                if crop_num_pixels > MIN_CROP_SIZE_PIXELS:       
                    # print(f'# {_} : {crop_num_pixels} px') 
                    if flag_save_tif_files:
                        # Let's see each image portion before saving it into a file
                        if flag_display:
                            # TODO: Only supports one-dim for now -> update leater
                            plt.imshow(np.squeeze(np_out_image_array), interpolation='none')
                            plt.colorbar()
                            plt.show()              
                        with rasterio.open(f'./pieces/piece_{new_index}_{_}.tif', 'w', **out_meta) as dest:
                            dest.write(np_out_image_array)
                            new_index = new_index + 1
                    # The single layer/band crop
                    single_layer_crop = np.squeeze(np_out_image_array) 
                    if band_index == 0: # It is the same for all bands
                        # Get the inner crop distance 
                        distance_on_mask = calculate_crop_distance_transform(np.squeeze(np_out_image_array))
                        if flag_display:
                            plt.imshow(distance_on_mask, interpolation='none')
                            plt.colorbar()
                            plt.show() 
                        inner_distance_crops.append(distance_on_mask)
                        multispectral_crop = np.zeros((single_layer_crop.shape[0], single_layer_crop.shape[1],max_num_bands))
                        multispectral_crop[:,:,0] = single_layer_crop
                        rastered_polygons.append(multispectral_crop)
                    else: # the rest of the bands
                        # We recover the multispectral_crop datastructure
                        multispectral_crop = rastered_polygons[sorted_crop_index]
                        # We check it is the expected one
                        if (single_layer_crop.shape[0] != multispectral_crop.shape[0]) or (single_layer_crop.shape[1] != multispectral_crop.shape[1]):
                            print(f"ERROR: Crop mismatch {num_errors} !!!")
                            num_errors = num_errors + 1
                        else:
                            multispectral_crop[:,:,band_index] = single_layer_crop
                        sorted_crop_index = sorted_crop_index + 1
        # Compose the bands (different in the fisrt one than in the others)            
    return rastered_polygons, inner_distance_crops


def check_geotiff_transformations_and_crs(file_paths):
    """
    Checks if GeoTIFF files in the provided list have the same transformations and CRS.
    
    Args:
        file_paths (list of str): List of paths to GeoTIFF files.
    
    Returns:
        tuple of lists: (matching_files, non_matching_files) where:
            matching_files (list of str): Files that have the same transformation and CRS as the first file.
            non_matching_files (list of str): Files that do not have the same transformation and CRS as the first file.
    """
    if not file_paths:
        return [], []  # Return empty lists if no file paths are provided
    
    # Initialize lists to store matching and non-matching file paths
    matching_files = []
    non_matching_files = []
    
    # Open the first file to get its transformation and CRS as a reference
    with rasterio.open(file_paths[0]) as first_file:
        reference_transformation = first_file.transform
        reference_crs = first_file.crs
        print(f'Reference file : {Path(file_paths[0]).name}')
        print(reference_transformation)
        print(reference_crs)
        print('---')
    
    # Check each file against the reference
    for path in file_paths:
        with rasterio.open(path) as current_file:
            if current_file.transform == reference_transformation and current_file.crs == reference_crs:
                matching_files.append(path)
                print(f'Studied file {Path(path).name} : Transformation & CRS matches')
                print('---')
            else:
                non_matching_files.append(path)
                print(f'Studied file {Path(path).name} : DOES NOT MATCH')
                print(current_file.transform)
                print(current_file.crs)
                print('---')


    return matching_files, non_matching_files

def test_geotiff_transformations_and_crs():
    """
    Just a test
    """
    base_full_filename = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/S2Images_Oct2017-Sep2018_T30TUM/S2_CroppedAOI_WithCloudMask/20171029/SD4EO_Spain_Crops_20171029_{}_CLM_10m.tif'
    list_geotiff_filenames = []
    for band_name in ALL_BAND_CODE:
        geotiff_filename  = base_full_filename.format(band_name)
        list_geotiff_filenames.append(geotiff_filename)
    matching_files, non_matching_files = check_geotiff_transformations_and_crs(list_geotiff_filenames)
    print("# Matching files:")
    for filename in matching_files:
        print(Path(filename).name)
    print("# Non matching files:")
    for filename in non_matching_files:
        print(Path(filename).name)      



def plot_geotiff_histogram(file_path):
    """
    Opens a GeoTIFF file, plots a histogram of the values it contains, and prints
    the maximum value which might correspond to areas affected by clouds.

    Args:
        file_path (str): The path to the GeoTIFF file.

    Returns:
        None.
    """
    # Open the GeoTIFF file.
    with rasterio.open(file_path) as src:
        # Read the first band of the GeoTIFF.
        band1 = src.read(1)
        
        # Calculate the maximum value in the band.
        max_value = np.max(band1)
        print(f"The maximum value (possibly representing cloud-affected areas) is: {max_value}")
        
        print(band1[455:469,459:471])

        # Flatten the array for histogram representation and filter out no-data values if necessary.
        # Assuming no-data values are represented as np.nan or a specific value e.g., -9999.
        # You might need to adjust this depending on how no-data is represented in your GeoTIFF.
        flat_band = band1.flatten()
        flat_band = flat_band[~np.isnan(flat_band)]  # Remove no-data values represented as np.nan
        # flat_band = flat_band[flat_band != -9999]  # Uncomment if no-data values are represented as -9999
        
        # Plot the histogram.
        plt.hist(flat_band, bins=50, color='blue', edgecolor='black')
        plt.title("Histogram of GeoTIFF Values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()


def calculate_crop_distance_transform(polygon_raster, offset = 5):
    """Calculates the distance transform of a rasterized polygon.

    Args:
        polygon_raster (np.array): A NumPy array representing the rasterized polygon.                  

    Returns:
        np.array: An array of the same dimensions as `polygon_raster` with OUR distance transform calculated.
    """
    # Calculate the mask around the crop in the array domain (valid crop area has values greater than NAN_THRESHOLD).
    binary_mask = polygon_raster < NAN_THRESHOLD # Pixels in the exterior of the crop should have a non-zero value, and pixels inside should be zero.
    inverted_binary_mask = polygon_raster > NAN_THRESHOLD
    # Calculate the standard Euclidean distance transform. Pixels that are zero are considered the "border"
    # to which the distance is calculated.
    standard_distance_transform = distance_transform_edt(binary_mask == 0)
    max_inner_distance = np.max(np.max(standard_distance_transform))
    # Our inverted crop distance: Subtract max distance value at crop valid pixels
    our_distance_transform = copy.deepcopy(standard_distance_transform)
    our_distance_transform[inverted_binary_mask] -= (max_inner_distance + offset)
    our_distance_transform = np.abs(our_distance_transform)
    return our_distance_transform


def calculate_crop_distance_transform_from_inv_binary_mask(mask_below_NAN, offset = 5):
    """Calculates the distance transform of a rasterized polygon.

    Args:
        binary_mask (np.array): A NumPy array mask with 1 in values BELOW NAN_THRESHOLD

    Returns:
        np.array: An array of the same dimensions as `polygon_raster` with OUR distance transform calculated.
    """
    binary_mask = mask_below_NAN
    reinverted_binary_mask = ~binary_mask
    # Calculate the standard Euclidean distance transform. Pixels that are zero are considered the "border"
    # to which the distance is calculated.
    standard_distance_transform = distance_transform_edt(binary_mask == 0)
    max_inner_distance = np.max(np.max(standard_distance_transform))
    # Our inverted crop distance: Subtract max distance value at crop valid pixels
    our_distance_transform = copy.deepcopy(standard_distance_transform)
    our_distance_transform[reinverted_binary_mask] -= (max_inner_distance + offset)
    our_distance_transform = np.abs(our_distance_transform)
    return our_distance_transform

def provide_less_overlapping_coordinates(extended_mask, crop_content, crop_mask2D, margin, num_attempts=10):
    """
    Args:
        extended_mask (np.ndarray): A binary 2D array with the mask of the inserted crops. It is a bit larger than the slices of multispectral_domain becuase it includes a margin for dealing with boundary conditions
        crop_content (np.ndarray): The smaller 3D array containing content to insert.
        crop_mask2D (np.ndarray): The smaller 2D array mask that takes into account both 10m and 20m bands.
        margin (int): additional size in each side of the extended domain
        num_attempts (int): number of attempts for obtaining the most suitable
                            randomized location
    
    Returns:
        best_coord_row(int): best position according to the previously occupied mask
        best_coord_col(int): best position according to the previously occupied mask
    """
    best_coord_row = 0
    best_coord_col = 0
    best_overlapping_factor = 100000000 # the lower the better
    extended_domain_side = extended_mask.shape[0]
    crop_rows = crop_content.shape[0]
    crop_cols = crop_content.shape[1]
    for i in range(num_attempts):
        rand_coord_row_m = random.randint(margin, extended_domain_side - crop_rows - margin)
        rand_coord_col_m = random.randint(margin, extended_domain_side - crop_cols - margin)
        current_mask_in_that_area = extended_mask[rand_coord_row_m:rand_coord_row_m+crop_rows, rand_coord_col_m:rand_coord_col_m+crop_cols]
        # crop_mask = np.squeeze(crop_content[:,:,0]) > NAN_THRESHOLD <-- DEPRECATED !!!
        intersection = np.logical_and(current_mask_in_that_area, crop_mask2D)
        overlapping_factor = np.count_nonzero(intersection)
        if overlapping_factor < best_overlapping_factor:
            best_overlapping_factor = overlapping_factor
            best_coord_row = rand_coord_row_m - margin
            best_coord_col = rand_coord_col_m - margin
    return best_coord_row, best_coord_col



def insert_crop_efficiently(multispectral_domain, extended_mask, crop_content, crop_mask2D, coord_row, coord_col):
    """
    Efficiently inserts content from a smaller 3D array into a larger 3D array based on specified coordinates,
    without overwriting non-zero values in the destination array.
    
    Args:
        multispectral_domain (np.ndarray): The large 3D array into which content will be inserted.
        extended_mask (np.ndarray): A binary 2D array with the mask of the inserted crops. It is a bit larger than the slices of multispectral_domain becuase it includes a margin for dealing with boundary conditions
        crop_content (np.ndarray): The smaller 3D array containing content to insert.
        crop_mask2D (np.ndarray): actual mask of the crop (not only the corresponding one for 10m bands!)
        coord_row (int): The row coordinate in the larger array where insertion begins.
        coord_col (int): The column coordinate in the larger array where insertion begins.
    
    Returns:
        multispectral_domain (np.ndarray): The updated large 3D array with the content from the smaller array inserted.
                
    Note:
        Only values in `crop_content` above NAN_THRESHOLD are copied, and existing non-zero values
        in `multispectral_domain` are not overwritten.
    """
    margin = int((extended_mask.shape[0] - multispectral_domain.shape[0])/2)
    # Some method make the calculation by imposing toroid boundary conditions
    # But we cannot support them easily, so we cut the crop extension to fit inside 
    # the current domain space
    crop_rows, crop_cols, layers = crop_content.shape
    flag_valid_pos = True
    if coord_row + crop_rows > multispectral_domain.shape[0]:
        crop_rows = multispectral_domain.shape[0] - coord_row
    if coord_col + crop_cols > multispectral_domain.shape[1]:
        crop_cols = multispectral_domain.shape[1] - coord_col
    if crop_cols <= 0 or crop_rows <= 0:
        flag_valid_pos = False

    if flag_valid_pos:
        # Creating binary masks for conditions
        mask_valid_crop_pixels_2D = crop_mask2D[:crop_rows,:crop_cols]  # This mask takes into account 10m and 20m bands!
        mask_destination_zero_2D = multispectral_domain[coord_row:coord_row+crop_rows, coord_col:coord_col+crop_cols, 0] < 0.1  # Positions that are not filled yet in multispectral_domain
        
        # Combine masks to find where both conditions are True
        mask_combined_target_2D = np.logical_and(mask_valid_crop_pixels_2D, mask_destination_zero_2D)
        
        # Using the combined mask to update values in multispectral_domain
        # This iterates over each layer and applies the mask
        for i in range(layers):
            # print(f'layer : {i}')
            destination_slice = multispectral_domain[coord_row:coord_row+crop_rows, coord_col:coord_col+crop_cols, i]
            source_slice = crop_content[:crop_rows, :crop_cols, i]
            
            # Apply the mask and update the multispectral_domain
            destination_slice[mask_combined_target_2D[:, :]] = source_slice[mask_combined_target_2D[:, :]]
            if i == 0: # Only for the first layer
                binary_update = destination_slice > 0
                extended_mask[coord_row+margin:coord_row+crop_rows+margin, coord_col+margin:coord_col+crop_cols+margin] = binary_update
    return multispectral_domain, extended_mask # updated data structures


def fill_and_update_mask(mask_extended_domain, v_texture, margin):
    """
    Optimized function to fill undefined values in the central region of v_texture with 
    the mean of their valid neighboring values and update mask_extended_domain to reflect the changes.
    
    Args:
    mask_extended_domain (np.ndarray): A 2D boolean array indicating correct (1) 
                                       and undefined (0) values in v_texture.
    v_texture (np.ndarray): A 3D array whose undefined values in the central region 
                            need to be filled.
    margin (int): Defines the start of the central region in rows and columns.
    
    Returns:
    np.ndarray: The modified v_texture array with undefined central region values filled.
    np.ndarray: The updated mask_extended_domain indicating newly filled values as 1.
    """
    # Ensure that mask and v_texture have compatible shapes
    assert mask_extended_domain.shape[0] == v_texture.shape[0] and mask_extended_domain.shape[1] == v_texture.shape[1], "mask_extended_domain and v_texture must have matching spatial dimensions"
    
    # Extract the central region of the mask
    central_mask = mask_extended_domain[margin:-margin, margin:-margin]
    
    # Find indices of undefined values in the central region
    undefined_indices = np.where(central_mask == 0)
    
    # Adjust indices to account for the margin
    undefined_indices = (undefined_indices[0] + margin, undefined_indices[1] + margin)
    
    # Iterate over each undefined value to fill with mean of neighbors
    for i, j in zip(*undefined_indices):
        for slice_index in range(v_texture.shape[2]):
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = i + dx, j + dy
                    # Ensure neighbor is within bounds and defined
                    if 0 <= nx < mask_extended_domain.shape[0] and 0 <= ny < mask_extended_domain.shape[1] and mask_extended_domain[nx, ny]:
                        neighbors.append(v_texture[nx, ny, slice_index])
            # Update the value if there are valid neighbors
            if neighbors:
                v_texture[i, j, slice_index] = np.mean(neighbors)
                # Update the mask to indicate this position is now defined
                mask_extended_domain[i, j] = 1
    
    return v_texture, mask_extended_domain


def get_optimal_central_extremes(full_side_side):
    # Find the largest power of 2 that fits within the mask's dimension
    power_of_2 = 2 ** int(np.log2(full_side_side))
    # Let's check MAX_SIDE_MARGIN
    raw_margin = full_side_side - power_of_2
    if raw_margin > MAX_SIDE_MARGIN: 
        # Not valid, we should increase the central side if we can
        power_of_2_and_3 = (power_of_2 / 2) * 3
        if power_of_2_and_3 < full_side_side:
            power_of_2 = int(power_of_2_and_3)
    # Calculate the starting and ending indices for the central square
    start_index = (full_side_side - power_of_2) // 2
    end_index = start_index + power_of_2
    return start_index, end_index, power_of_2


def get_central_mask_area(mask_extended_domain):
    # Determine the dimension of the mask
    n = mask_extended_domain.shape[0]
    start_index, end_index, power_of_2 = get_optimal_central_extremes(n)
    # Extract the central square
    central_square = mask_extended_domain[start_index:end_index, start_index:end_index]
    return central_square, power_of_2


def estimate_central_coverture(mask_extended_domain):
    """
    Calculate the percentage of 1s in the central region of a binary mask.
    The central region is the largest square with side length being a power of 2,
    well-centered within the mask.

    Args:
    mask_extended_domain (np.array): A 2D square binary array representing the mask.

    Returns:
    float: The percentage of 1s in the central square region.
    int: Effective resolution of the central square where the calculation has been performed
    np.array : The estimated central mask
    """
    # Extract the central square
    central_square, power_of_2 = get_central_mask_area(mask_extended_domain)

    # Calculate and return the percentage of 1s
    percentage_of_ones = np.mean(central_square) * 100
    return percentage_of_ones, power_of_2, central_square


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


def Initial_CyL_Sentinel_analysis(subset_str, crop_name, crop_type_ID, date_str, mainS2path, mainS1path):
    """
    This function sweeps over all geotiff files for the indicated crop type and
    date; it call the function that initializes data structures and later
    they are filled in with the corresponding mulspectral pieces (crop plots) 
    which finally are stored in a pickle file as a sorted (by pixel size) list
    of crops (CropDeck -> CropRegisterClass)

    Args:
        subset_str (str): It should be 'Cat', as this is a specialized function. 
                          We maintain this parameter as a security check
        crop_name (str): Crop name according to ROMANIA identifier 
        crop_type_ID (str): Crop type according to ROMANIA identifier. 
        date_str (str): Desired date (when the satellite took the images)
                        In the case of Sentinel 1 only the year-month are considered
        mainS2path (str): Input data path with slots for bands
        mainS1path (str): Input data path (we only take selected bands)

    Returns:
        CropDeck: The list of crops (CropDeck -> CropRegisterClass), sorted (by pixel size) with all the data for Sentinel 2 spectral bands and selected Sentinel 1 polarizations

    """
    geotiff_S2_base_full_filename = mainS2path + date_str +'/SD4EO_Spain_Crops_' + date_str + '_{}_CLM_10m.tif'
    geotiff_S1_base_full_path = mainS1path + date_str[:-2] + '/'
    geotiff_S1_base_full_filename = find_tif_file(geotiff_S1_base_full_path)
    # TODO: Support for BOTH S1 bands in Catalonia dataset
        
    crop_name = Romania_Crop_Codes[crop_index][0]
    crop_type_ID = Romania_Crop_Codes[crop_index][1]

    # Load the shapefile
    shapefile_full_path = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/Fields_GTData/SD4EO_GTData_Training_32630.shp'  
    shapefile_gdf = load_shapefile(shapefile_full_path)
    # Extract polygons for a specific crop type, e.g., "Wheat"
    crop_polygons = extract_polygons_by_CyL_crop_type_ID(shapefile_gdf, crop_type_ID)
    
    # Print the result or do further processing
    print(crop_polygons)

    # Let's extract the rastered polygon mask and the content for each crop, time sample and layer. It is not easy, because CRS must be compatible
    shapefile_unsorted_polygons = crop_polygons
    # Let's check the reference system
    crs_shapefile = shapefile_unsorted_polygons.crs
    unsorted_rastered_polygons, sorted_list_of_index_crop_regs = initial_S2_analysis_on_rastered_polygons(shapefile_unsorted_polygons, crs_shapefile, geotiff_S2_base_full_filename, S2BAND10m_CODE[0], S2BAND20m_CODE[0], subset_str, date_str, flag_display=False)
    # Because of the clouds, it is possible to not extract any valid crop
    # So we have to deal with it
    if unsorted_rastered_polygons is not None:
        sorted_rastered_polygons = fill_content_for_allS2_bands(subset_str,unsorted_rastered_polygons, sorted_list_of_index_crop_regs, shapefile_unsorted_polygons, crs_shapefile, geotiff_S2_base_full_filename)
        sorted_rastered_polygons = fill_content_for_S1_bands(subset_str, sorted_rastered_polygons, shapefile_unsorted_polygons, crs_shapefile, geotiff_S1_base_full_filename)
        # Save work up to here to split the process and speed up development and tests
        pickle_filename = f"./{subset_str}/{date_str}_{crop_name}.pickle"
        with open(pickle_filename, 'wb') as pickle_file:
            pickle.dump(sorted_rastered_polygons, pickle_file)
        return sorted_rastered_polygons
    else:
        return None


def load_crop_decks_from_single_pickle(subset_str, date_str, crop_name, cat_tile=''):
    """
    Load a the multispectral crop pieces from a pickle file.

    Args:
        subset_str (str): String that identifies the dataset ('CyL' vs 'Cat')
        date_str (str): explicit date, Sentinel 1 layer only month-year
        crop_name (str): Name of the crop according ROMANIA nomenclature
        cat_tile (str): Sentinel tile ID (only applies to Cat dataset)

    Returns:
        crop_deck (CropDeck->CropRegisterClass): crops in order of pixel size
    """    
    # Load the saved files
    if subset_str == 'CyL':
        pickle_filename = f"./{subset_str}/{date_str}_{crop_name}.pickle"
    else:
        pickle_filename = f"./{subset_str}/{cat_tile}/{date_str}_{crop_name}.pickle"
    try:
        with open(pickle_filename, 'rb') as archivo:
            # We load the serialized data structures from pickle file
            # We expect data to be a CropDeck sorted by pixel size of plots
            sorted_crop_deck = pickle.load(archivo)
    except:
        sorted_crop_deck = None
    return sorted_crop_deck


def combine_crop_deck_list(crop_deck_list):
    """
    Combines all the crop_decks into a single and sorted crop_deck (by pixel-size).

    Args:
        crop_deck_list (list): list of CropDecks that we want to combine

    Returns:
        crop_deck (CropDeck->CropRegisterClass): crops in order of pixel size

    Warning:
        There will be several crops with the same uuid, as we are integrating 
        the same area in different time samples
    """       
    combined_crop_deck = None
    if len(crop_deck_list) > 0:
        combined_crop_deck = crop_deck_list[0]
    if len(crop_deck_list) > 1:
        unsorted_crop_in_region = []
        for crop_deck_i in crop_deck_list[1:]:
            unsorted_crop_in_region = unsorted_crop_in_region + crop_deck_i.crop_in_region
        # Let's reorder it by pixel size
        unsorted_list_of_crop_regs = []
        for orig_index, crop in enumerate(unsorted_crop_in_region):
            unsorted_list_of_crop_regs.append((orig_index, crop.area_px))
            sorted_list_of_index_crop_regs = sorted(unsorted_list_of_crop_regs, key=lambda x: x[1], reverse=True)
        sorted_list_of_crop_regs = []
        for (index, area) in sorted_list_of_index_crop_regs:
            sorted_list_of_crop_regs.append(unsorted_crop_in_region[index])
        # Let's edit metadata
        combined_crop_deck.crop_in_region = sorted_list_of_crop_regs
    return combined_crop_deck



def find_pickle_files_in_folder(folder_str, start_str, end_str):
    """
    Finds files in a specified folder whose names start with start_str and end with end_str.
    
    Args:
        folder_str (str): The path to the folder where files are to be searched.
        start_str (str): The starting string of the file names of interest.
        end_str (str): The ending string of the file names of interest.
        
    Returns:
        list: A list containing the names of the files that match the criteria.
    """
    # Compile a regular expression pattern for file names that start with start_str and end with end_str
    pattern = re.compile(f"^{re.escape(start_str)}.*{re.escape(end_str)}$")
    
    # List all entries in the given folder
    entries = os.listdir(folder_str)
    
    # Filter and return the list of files that match the pattern
    return [file for file in entries if pattern.match(file) and os.path.isfile(os.path.join(folder_str, file))]


def load_crop_decks_from_combined_pickles(subset_str, date_str, crop_name, cat_tile='', mode=PickleCombination.NOT_VALID):
    """
    Load a set of first-stage pickle files, not only one.
    We could select all the files of the same crop type for a whole month, 
    all available tiles, combinations of datasets...

    Args:
        subset_str (str): String that identifies the dataset ('CyL' vs 'Cat')
        date_str (str): base of the date, depending on the mode we could take 
                        only the year and the month
        crop_name (str): Name of the crop according ROMANIA nomenclature
        cat_tile (str): Sentinel tile ID
        mode (PickleCombination): All suported modes

    Returns:
        combined_crop_deck (CropDeck->CropRegisterClass): crops in order of pixel size
    """
    print(f' --== {crop_name} ==--')
    combined_crop_deck = None
    match mode:
        case PickleCombination.NOT_VALID:
            combined_crop_deck = load_crop_decks_from_single_pickle(subset_str, date_str, crop_name, cat_tile)
        case PickleCombination.MULTITILE:
            if subset_str == 'CyL':
                # It has no effect
                combined_crop_deck = load_crop_decks_from_single_pickle(subset_str, date_str, crop_name, cat_tile)
            else: 
                # We have to load all pickles from possible tiles
                crop_deck_list = []
                for cat_tile in CatTiles:
                    crop_deck_to_add = load_crop_decks_from_single_pickle(subset_str, date_str, crop_name, cat_tile)
                    if crop_deck_to_add is not None:
                        crop_deck_list.append(crop_deck_to_add)
                combined_crop_deck = combine_crop_deck_list(crop_deck_list)
        case PickleCombination.ALL_MONTH:
            # We need to discover all the available dates for the selected month
            candidate_files = find_pickle_files_in_folder(f"./{subset_str}/", date_str[:-2], f"_{crop_name}.pickle")
            crop_deck_list = []
            for full_filename in candidate_files:
                new_date_str = full_filename[:8]
                print(new_date_str)
                crop_deck_to_add = load_crop_decks_from_single_pickle(subset_str, new_date_str, crop_name, cat_tile)
                if crop_deck_to_add is not None:
                    crop_deck_list.append(crop_deck_to_add)
            combined_crop_deck = combine_crop_deck_list(crop_deck_list)
        case PickleCombination.ALL_MONTH_MT:
            if subset_str == 'CyL':
                # It has no effect
                combined_crop_deck = load_crop_decks_from_combined_pickles(subset_str, date_str, crop_name, cat_tile, mode=PickleCombination.ALL_MONTH)
            else: # Catalonia
                # We need to discover all the available dates for the selected month in each Tile
                crop_deck_list = []
                for cat_tile in CatTiles:
                    candidate_files = find_pickle_files_in_folder(f"./{subset_str}/{cat_tile}/", date_str[:-2], f"_{crop_name}.pickle")
                    for full_filename in candidate_files:
                        new_date_str = full_filename[:8]
                        print(f"{cat_tile}  {new_date_str}")
                        crop_deck_to_add = load_crop_decks_from_single_pickle(subset_str, new_date_str, crop_name, cat_tile)
                        if crop_deck_to_add is not None:
                            crop_deck_list.append(crop_deck_to_add)
                combined_crop_deck = combine_crop_deck_list(crop_deck_list)                
        case PickleCombination.ALL_MONTH_BOTH_DS:
            # We go for both datasets, so original subset_str does not mind
            crop_deck_list = []
            subset_str = 'CyL'
            candidate_files = find_pickle_files_in_folder(f"./{subset_str}/", date_str[:-2], f"_{crop_name}.pickle")
            crop_deck_list = []
            for full_filename in candidate_files:
                new_date_str = full_filename[:8]
                print(new_date_str)
                crop_deck_to_add = load_crop_decks_from_single_pickle(subset_str, new_date_str, crop_name, cat_tile)
                if crop_deck_to_add is not None:
                    crop_deck_list.append(crop_deck_to_add)
            # Now we switch the dataset
            subset_str = 'Cat'
            for cat_tile in CatTiles:
                candidate_files = find_pickle_files_in_folder(f"./{subset_str}/{cat_tile}/", date_str[:-2], f"_{crop_name}.pickle")
                for full_filename in candidate_files:
                    new_date_str = full_filename[:8]
                    print(f"{cat_tile}  {new_date_str}")
                    crop_deck_to_add = load_crop_decks_from_single_pickle(subset_str, new_date_str, crop_name, cat_tile)
                    if crop_deck_to_add is not None:
                        crop_deck_list.append(crop_deck_to_add)
            combined_crop_deck = combine_crop_deck_list(crop_deck_list)   
    return combined_crop_deck


def chaotic_aggregation(sorted_crop_deck, domain_side, chaotic_percentage, threshold_to_save, subset_str, date_str, flag_view=False, flag_view_save=False):
    """
    Chaotic aggregation algorithm

    Args:
        sorted_crop_deck (CropDeck->CropRegisterClass): crops in order of pixel size
        domain_side (int): target size for the 2^P domain
        chaotic_percentage (float): percentage for the basic chaotic scheme and 
                                    100-percentage will follow the more complex 
                                    ditance-based variant
        threshold_to_save (float): We only save the file if the result achieves 
                                   a coberture at least as high as the threshold value
        flag_view (bool): Just a flag to visualize the process
        flag_view_save (bool): Just a flag to save the visualization
        v_texture (np.array): A 3D (multispectral) square array 

    Returns:
        coverture (float): achieved percentage of coverture 
        v_texture (np.array) : The composed 3D (multispectral) square array 
    """

    num_total_available_crops = len(sorted_crop_deck.crop_in_region) 
    print(f'Num available crops: {num_total_available_crops}')
    # Step 0: Create the 2^P x 2^P x \xi domain
    num_bands = len(ALL_BAND_CODE)
    domain_margin = 1
    extended_domain_side = domain_side + domain_margin*2
    mask_extended_domain = np.zeros((extended_domain_side, extended_domain_side))
    # Boundary conditions
    mask_extended_domain[0,:] = 1
    mask_extended_domain[-1,:] = 1
    mask_extended_domain[:,0] = 1
    mask_extended_domain[:,-1] = 1
    v_texture = np.zeros((extended_domain_side, extended_domain_side, num_bands))
    # Step 1: Throw some large crops 
    num_random_crops = num_total_available_crops*(float(chaotic_percentage)/100)
    target_coords_per_crop = []
    counter_check = 0
    for i in range(0,num_total_available_crops):
        crop_content = sorted_crop_deck.crop_in_region[i].array3D  
        rows = crop_content.shape[0]
        cols = crop_content.shape[1]
        crop_mask2D = sorted_crop_deck.crop_in_region[i].mask2D
        if i < num_random_crops: # If it is one of the first crops we go for the chaotic alg
            coord_row, coord_col = provide_less_overlapping_coordinates(mask_extended_domain, crop_content, crop_mask2D, domain_margin, num_attempts=250)
        else: # we go for the distance guided approach
            coord_row, coord_col = get_coords_by_correlation_on_masked_domain(mask_extended_domain, sorted_crop_deck.crop_in_region[i])
        v_texture, mask_extended_domain = insert_crop_efficiently(v_texture, mask_extended_domain, crop_content, crop_mask2D, coord_row, coord_col)
        crop_coords = CropCoords(coord_row, coord_col, sorted_crop_deck.crop_in_region[i].internal_index, sorted_crop_deck.crop_in_region[i].uuid)
        target_coords_per_crop.append(crop_coords)
        counter_check = counter_check + 1
        if counter_check >= 3000:
            # We try to speed up massive datasets where initial larger crops cover the requested 
            # surface, so we do not need to map all possible (and small) final crops
            counter_check = 0
            coverture, _1, _2 = estimate_central_coverture(mask_extended_domain)
            if coverture > 99.95:
                print(f'Skipped a portion of smaller crops, puzzle completed: {i}/{num_total_available_crops}')
                break

    coverture, effective_res, central_mask = estimate_central_coverture(mask_extended_domain)
    if coverture > 99.5 and coverture < 100.0: # We try to fill in the last pixels
        # Let's look for small "one-pixel-size" holes
        side_half_margin = int((mask_extended_domain.shape[0] - effective_res)/2)
        v_texture, mask_extended_domain = fill_and_update_mask(mask_extended_domain, v_texture, side_half_margin)
        coverture_new, effective_res, central_mask = estimate_central_coverture(mask_extended_domain)
        print(f"Corrected coverture from {coverture} to {coverture_new}")
        coverture = coverture_new 

    if flag_view or flag_view_save:
        # Figure to show the result and validate it visually
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  
        # Show the extended area with the crop layout
        composed_crop_img = axes[0].imshow(np.squeeze(v_texture[:,:,0]), interpolation='none', cmap='viridis', aspect='equal')  
        axes[0].set_title(f'Extended resolution : {domain_side} x {domain_side}')
        cbar = fig.colorbar(composed_crop_img, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label('Band #7')  # Etiqueta opcional para el colorbar
        # Show the coveture mask 
        axes[1].imshow(central_mask, interpolation='none', aspect='equal')
        axes[1].set_title(f'Effective res : {effective_res} x {effective_res}  Coverture: {coverture:.3f}')
        # plt.tight_layout()  # Ajusta automáticamente los subplots para que se ajusten al área de la figura
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajusta el layout de la figura
        plt.suptitle(f'Crop : {crop_name}    # plots : {num_total_available_crops}    Dataset : {subset_str} \n')
        if flag_view_save:
            fig.savefig(f'./imgs/{subset_str}_{date_str}_{crop_name}_{domain_side}_{effective_res}.jpg', format='jpg')
        if flag_view:
            plt.show()
        plt.close(fig)
    # Save data structures 
    if coverture >= threshold_to_save:
        save_final_result_in_netCDF(v_texture, coverture, subset_str, date_str, crop_name)
    return coverture, v_texture


def select_2P_central_portion_3D(v_texture):
    """
    Extract the largest square central region with side length being a power of 2 
    inside the multidimensional array (we consider the 2 fisrt dimensions)

    Args:
        v_texture (np.array): A 3D (multispectral) square array 

    Returns:
        np.array : The estimated central mask
    """
    # Determine the dimension of the mask
    n = v_texture.shape[0]
    start_index, end_index, power_of_2 = get_optimal_central_extremes(n)

    # Extract the central square
    central_square = v_texture[start_index:end_index, start_index:end_index, :]
    return central_square


def distance_transform_for_masked_domain(masked_domain, offset = 5, flag_view=False):
    """
    This function calculates our distance transform for the full extended domain,
    where several plots have been already marked

    Args:
        masked_domain (np.array): 2D array with masked crop plot already projected 
                                  on the extended domain
        offset (int):             minimal distance in the inverese transformation
        flag_view (bool):         Just a flag to visualize the process

    Returns:
        (row,col): tuple of ints with the coordinates of the optimal position 
                   of the crop in the already populated domain
    """

    binary_masked_domain = masked_domain == 0
    inverted_binary_mask = ~binary_masked_domain
    # Calculate the standard Euclidean distance transform. 
    # Pixels that are zero are considered the "border"
    # to which the distance is calculated.
    standard_distance_transform = distance_transform_edt(binary_masked_domain)
    max_inner_distance = np.max(np.max(standard_distance_transform))
    # Our inverted crop distance: Subtract max distance value at crop valid pixels
    our_distance_transform = copy.deepcopy(standard_distance_transform)
    our_distance_transform[binary_masked_domain] -= (max_inner_distance + offset)
    zero_img = np.zeros(our_distance_transform.shape)
    our_distance_transform[inverted_binary_mask] = zero_img[inverted_binary_mask]
    our_distance_transform = np.abs(our_distance_transform)
    # TODO: Introduce a slight non-linearity
    if flag_view:
        plt.figure()
        plt.imshow(our_distance_transform)
        plt.colorbar()
        plt.show()
    return our_distance_transform


def find_best_correlation_position(distance_domain, distance_crop):
    """
    Find the best position to place `distance_crop` on `distance_domain` based on maximum correlation.
    This function uses the Fourier transform to efficiently calculate the cross-correlation.

    Args:
        distance_domain (np.ndarray): The larger 2D square array with sides that are powers of 2.
        distance_crop (np.ndarray): The smaller 2D array whose position within `distance_domain` is sought.

    Returns:
        tuple: The row and column (y, x) indicating where `distance_crop` best fits within `distance_domain`.
    """

    # Calculate the cross-correlation using FFT
    correlation = fftconvolve(distance_domain, distance_crop[::-1, ::-1], mode='same')
    # plt.figure()
    # plt.imshow(correlation)
    # plt.colorbar()
    # Find the position of the maximum correlation
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    return (y, x)

def get_coords_by_correlation_on_masked_domain(masked_domain, crop_register, flag_view=False):
    """
    This function estimates the 2D coordinates of the distance-transformed AABB crop 
    inside the updated distance-transformed masked domain in Fourier space to perform
    the correlation more efficiently.

    Args:
        masked_domain (np.array): 2D array with masked crop plot already projected 
                                  on the extended domain
        crop_register (CropDeck): crop to be located into the domain
        flag_view (bool):         Just a flag to visualize the process

    Returns:
        (row,col): tuple of ints with the coordinates of the optimal position 
                   of the crop in the already populated domain
    
    """
    standard_distance_transform = distance_transform_for_masked_domain(masked_domain, offset = 5, flag_view=False)
    crop_dist_map = crop_register.distance_map_2D
    row, col = find_best_correlation_position(standard_distance_transform, crop_dist_map)
    if flag_view:
        plt.figure()
        plt.imshow(crop_register.distance_map_2D)
        plt.title('Crop mask distance')
        plt.figure()
        plt.imshow(standard_distance_transform) 
        plt.title('Distance transform original')
        plt.figure()
        new_standard_distance_transform = copy.deepcopy(standard_distance_transform)
        new_standard_distance_transform[row:row+crop_dist_map.shape[0], col:col+crop_dist_map.shape[1]] = crop_dist_map[:,:]
        plt.imshow(new_standard_distance_transform) 
        plt.title('Copy/Pasted crop in distance transform')
        plt.show()
    return (row,col)



def save_final_result_in_pickle(v_texture, coverture, subset_str, date_str, crop_name, domain_side, target_coords_per_crop):
    """
    Save the synthetized multiespectral image as a texture like a numpy array with an
    additional list of the target coordinates for each crop according to its order in the
    considered list of crops. It is just a reference when we has merged several lists
    (like a whole set of tiles or dates within a month), as far as the same crop may
    appear multiple times with different shapes because of the cloud masking effect. So,
    have no mechanisms to disntinguish among them. However, it will be quite clear when 
    we deal with only one list of crops as the base for the polygon covering result.
    Additional useful information is also recorded in the pickle file.

    Args:
        v_texture (np.array): multidimensional (multi-espectral) 3D array
        coverture (float): achieved crop coverture after the polygon covring stage
        subset_str (str): name of the main dataset
        date_str (str): reference date (it could be just a year-month)
        crop_name (str): crop name according to ROMANIA (Sen4AgriNet) nomenclature
        domain_side (int): size of the side of the squared domain
        target_coords_per_crop (list[CropCoords]): coordinates of the left upper corner
           of the crop axis-aligned bounding box in the squared extended domain (not
           the 2-power cropped one), it also contains the associated crop identifier.

    Returns:
        The output is a file stored in the hard disk according to prior naming rules
    """    
    central_2P_texture = select_2P_central_portion_3D(v_texture)
    effective_res = central_2P_texture.shape[0]
    data_to_save = {
        "texture" : central_2P_texture,
        "coverture" : coverture,
        "dataset" : subset_str, 
        "date" : date_str, 
        "crop_name" : crop_name,
        "side_size" : domain_side,
        "list_coords" : target_coords_per_crop,
        "bands" : ALL_BAND_CODE
    }
    # Save structure in pickle file
    with open(f'./{subset_str}/{date_str}_{crop_name}_{effective_res}.pickle', 'wb') as pickle_file:
        pickle.dump(data_to_save, pickle_file)


def save_final_result_in_netCDF(v_texture, coverture, subset_str, date_str, crop_name):
    """
    Save the synthetized multiespectral image as a texture in a xarray labelled data
    structure, which is stored in a standarized netCDF file, so it will be used easily
    by the next algorithms in family 2.x
    Additional useful information is also recorded as metadata.

    Args:
        v_texture (np.array): multidimensional (multi-espectral) 3D array
        coverture (float): achieved crop coverture after the polygon covring stage
        subset_str (str): name of the main dataset
        date_str (str): reference date (it could be just a year-month)
        crop_name (str): crop name according to ROMANIA (Sen4AgriNet) nomenclature

    Returns:
        The output is a file stored in the hard disk according to prior naming rules
    """
    central_2P_texture = select_2P_central_portion_3D(v_texture)
    effective_res = central_2P_texture.shape[0]
    composed_crop = xr.DataArray(central_2P_texture, dims=("x", "y", "band"), coords={"band": ALL_BAND_CODE})
    composed_crop.attrs["long_name"] = crop_name
    composed_crop.attrs["date"] = date_str
    composed_crop.attrs["dataset"] = subset_str
    composed_crop.attrs["coverture"] = coverture
    filename_nc = f'./{subset_str}/{date_str}_{crop_name}_{effective_res}.nc'
    composed_crop.to_netcdf(filename_nc)
    print(f'Saved :   {filename_nc}')



def extract_polygons_by_Catalonia_crop_type_ID(shapefile, crop_type_ID):
    """
    Filters a GeoDataFrame based on specified crop types.

    Args:
        shapefile (gpd.GeoDataFrame): A GeoDataFrame representing the loaded shapefile data.
        crop_type_ID (int):

    Returns:
        gpd.GeoDataFrame: A filtered GeoDataFrame containing only the rows that match the specified crop types.
    """
    # First, we transfor the crop_type in CyL to the list of identifiers for Catalonia dataset
    match crop_type_ID:  # It could have been implemented with a dictionary, but it will be less explicit
        case 5 : # Barley
            crop_types = ['ORDI']
        case 1: # Wheat
            crop_types = ['BLAT TOU', 'BLAT DUR', 'BLAT KHORASAN']
        case 39: # OtherGrainLeguminous
            crop_types = ['CIGRONS SIE', 'LLENTIES SIE', 'ERBS (LLEGUM) SIE', 'CIGRONS NO SIE', 'MONGETA SIE', 'MONGETA NO SIE', 'LLENTIES NO SIE', 'ERBS (LLEGUM) NO SIE', 'GUIXES NO SIE O PEDREROL NO S*', 'GUIXES SIE O PEDREROL SIE']
        case 40: # Peas
            crop_types = ['PÃˆSOLS SIE', 'PÃˆSOLS NO SIE']
        case 21: # FallowAndBareSoil
            crop_types = ['GUARET NO SIE/ SUP. LLIURE SE*', 'GUARET SIE/ SUP. LLIURE SEMBRA', 'GUARET SIE AMB ESPÃˆCIES MEL.L*']
        case 52: # Vetch
            crop_types = ['VECES NO SIE', 'VECES SIE']
        case 60: # Alfalfa
            crop_types = ['ALFALS SIE', 'ALFALS NO SIE']
        case 33: # Sunflower
            crop_types = ['GIRA-SOL']
        case 8: # Oats
            crop_types = ['CIVADA']
            
    # s (list of str): A list of crop type identifiers to filter by.
    # Filter the GeoDataFrame based on whether the 'Cultius' column matches any of the crop types in the list.
    filtered_gdf = shapefile[shapefile['Cultiu'].isin(crop_types)]
    flag_check = False
    if flag_check:
        num_elems = 0
        for indice, fila in filtered_gdf.iterrows():
        # Aquí puedes hacer lo que necesites con la fila
        # Por ejemplo, imprimir el valor de 'Cultiu' de cada fila
            print(f" indx: {indice} : {fila['ORIG_FID']}")
            num_elems = num_elems + 1
        print(f"num elems : {num_elems}")
    return filtered_gdf


def test_catalonia_dataset():
    """
    DEPRECATED: Just for testing initial ideas
    """
    # Load the shapefile
    shapefile_full_path = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/ComplementaryData_Catalonia/PlotsBoundaries_Internal10mBuffer/SD4EO_CS2_ComplementaryData_DUNSIGPAC2918_ibuf10m.shp'  
    shapefile_gdf = load_shapefile(shapefile_full_path)
    print(shapefile_gdf.columns.tolist())
    field_str = 'Cultiu' # 'BUFF_DIST' #,'Seca_Regad' # 'ORIG_FID' #'Cultiu'
    elements = get_elements_for_shapefile_column(shapefile_gdf, field_str)
    for i in elements:
        print(f">>{i}<<")
    # Extract polygons for a specific crop type, e.g., "Wheat"
    # crop_polygons = extract_polygons_by_crop_type_ID(shapefile_gdf, crop_type_ID)       
    extract_polygons_by_Catalonia_crop_type_ID(shapefile_gdf, 33)


def find_incompletefile_path(input_path, band_name):
    """
    Finds and returns the full path of a file within a specified directory, based on an input string
    that combines the directory path and the starting characters of the file name.

    Args:
        input_path (str): The combined directory path and file start characters.

    Returns:
        str: The full path of the matching file. If no file is found, returns an empty string.

    """
    # Determine the last occurrence of the path separator to split directory and file start
    separator_index = input_path.rfind('/')
    
    # If the separator is not found, return an empty string (invalid input)
    if separator_index == -1:
        return ''

    # Extract the directory and file start from the input path
    directory = input_path[:separator_index]
    file_start = input_path[separator_index + 1:]
    
    # Construct the sea1rch pattern to include the directory, the starting characters, and any extension
    search_pattern = os.path.join(directory, file_start.format(band_name) + '*')

    # Use glob to find files that match the search pattern
    matching_files = glob.glob(search_pattern)

    # Return the first matching file's full path if any, else return an empty string
    return matching_files[0] if matching_files else ''


def Initial_Catalonia_Sentinel_analysis(subset_str, crop_name, crop_type_ID, date_str, mainS2path, mainS1path, cat_tile):
    """
    This function sweeps over all geotiff files for the indicated crop type, date
    and tile; it call the function that initializes data structures and later
    they are filled in with the corresponding mulspectral pieces (crop plots) 
    which finally are stored in a pickle file as a sorted (by pixel size) list
    of crops (CropDeck -> CropRegisterClass)

    Args:
        subset_str (str): It should be 'Cat', as this is a specialized function. 
                          We maintain this parameter as a security check
        crop_name (str): Crop name according to ROMANIA identifier (not the Catalonia one)
        crop_type_ID (str): Crop type according to ROMANIA identifier. 
                            Because one ROMANIA ID may correspond to several Catalonia ones.
        date_str (str): Desired date (when the satellite took the images)
                        In the case of Sentinel 1 only the year-month are considered
        mainS2path (str): Input data path with slots for bands
        mainS1path (str): Input data path (we only take selected bands)
        cat_tile (str): Code for the title (T31T*)

    Returns:
        CropDeck: The list of crops (CropDeck -> CropRegisterClass), sorted (by pixel size) with all the data for Sentinel 2 spectral bands and selected Sentinel 1 polarizations

    """
    geotiff_S2_base_start_filename = mainS2path + date_str + '/Vegetation-Indices_{}_S2-10m_'
    geotiff_S1_base_full_path = mainS1path + date_str[:-2] + '/'
    geotiff_S1_base_full_filename = find_tif_file(geotiff_S1_base_full_path)
        
    crop_name = Romania_Crop_Codes[crop_index][0]
    crop_type_ID = Romania_Crop_Codes[crop_index][1]

    # Load the shapefile
    shapefile_full_path = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/ComplementaryData_Catalonia/PlotsBoundaries_Internal10mBuffer/SD4EO_CS2_ComplementaryData_DUNSIGPAC2918_ibuf10m.shp'  
    shapefile_gdf = load_shapefile(shapefile_full_path)
    # Extract polygons for a specific crop type, e.g., "Wheat"
    crop_polygons = extract_polygons_by_Catalonia_crop_type_ID(shapefile_gdf, crop_type_ID)
    
    # Print the result or do further processing
    print(crop_polygons)

    # Let's extract the rastered polygon mask and the content for each crop, time sample and layer. It is not easy, because CRS must be compatible
    shapefile_unsorted_polygons = crop_polygons
    # Let's check the reference system
    crs_shapefile = shapefile_unsorted_polygons.crs
    unsorted_rastered_polygons, sorted_list_of_index_crop_regs = initial_S2_analysis_on_rastered_polygons(shapefile_unsorted_polygons, crs_shapefile, geotiff_S2_base_start_filename, S2BAND10m_CODE[0], S2BAND20m_CODE[0], subset_str, date_str, flag_display=False)
    if unsorted_rastered_polygons is not None:
        sorted_rastered_polygons = fill_content_for_allS2_bands(subset_str,unsorted_rastered_polygons, sorted_list_of_index_crop_regs, shapefile_unsorted_polygons, crs_shapefile, geotiff_S2_base_start_filename)
        sorted_rastered_polygons = fill_content_for_S1_bands(subset_str, sorted_rastered_polygons, shapefile_unsorted_polygons, crs_shapefile, geotiff_S1_base_full_filename)
        # Save work up to here to split the process and speed up development and tests
        pickle_filename = f"./{subset_str}/{cat_tile}/{date_str}_{crop_name}.pickle"
        with open(pickle_filename, 'wb') as pickle_file:
            pickle.dump(sorted_rastered_polygons, pickle_file)
        return sorted_rastered_polygons
    else:
        return None


def domain_size_down(full_side_side):
    # Let's look for current setup and margin
    # 1) Chech if actual size is a power of 2
    # Find the largest power of 2 that fits within the mask's dimension
    power_of_2 = 2 ** int(np.log2(full_side_side))
    power_of_2_and_3 = (power_of_2 / 2) * 3
    # Let's check MAX_SIDE_MARGIN
    raw_margin = full_side_side - power_of_2
    if (full_side_side - power_of_2) <= MAX_SIDE_MARGIN:
        # The IT IS a near power of two, and we swap it by a 2/3
        margin = full_side_side - power_of_2
        new_side_size = (power_of_2/4)*3 + margin
    else:
        # Perfect, we just go for the following (lower) pure power of 2 
        margin = full_side_side - power_of_2_and_3
        new_side_size = power_of_2 + margin
    return int(new_side_size)


def generate_all_months_composed_DS_textures(crop_name, domain_side, chaotic_percentage, threshold_to_save):
    start_time = time.time()
    for month_date in year_month_list:  
        crop_deck = load_crop_decks_from_combined_pickles('', month_date, crop_name, '', PickleCombination.ALL_MONTH_BOTH_DS)
        coverture = 0.0
        adaptative_domain_side = domain_side
        while coverture < threshold_to_save:
            print(f'- Current target resolution: {adaptative_domain_side}')
            coverture, _ = chaotic_aggregation(crop_deck, adaptative_domain_side, chaotic_percentage, threshold_to_save, 'CyL_and_Cat', month_date, flag_view=False, flag_view_save=True)
            if coverture < threshold_to_save:
                adaptative_domain_side = domain_size_down(adaptative_domain_side)
        print(f'Coverture : {coverture}   Size: {adaptative_domain_side}')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{crop_name}  - {month_date[:-2]} -> Spent time: {execution_time} seconds.")
    pass


# def generate_all_months_composed_DS_textures(crop_name, domain_side, chaotic_percentage, threshold_to_save):
#     start_time = time.time()
#     month_date = year_month_list[11]
#     crop_deck = load_crop_decks_from_combined_pickles('', month_date, crop_name, '', PickleCombination.ALL_MONTH_BOTH_DS)
#     coverture = 0.0
#     adaptative_domain_side = domain_side
#     while coverture < threshold_to_save:
#         print(f'- Current target resolution: {adaptative_domain_side}')
#         coverture, _ = chaotic_aggregation(crop_deck, adaptative_domain_side, chaotic_percentage, threshold_to_save, 'CyL_and_Cat', month_date, flag_view=False, flag_view_save=True)
#         if coverture < threshold_to_save:
#             adaptative_domain_side = domain_size_down(adaptative_domain_side)
#     print(f'Coverture : {coverture}   Size: {adaptative_domain_side}')
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"{crop_name}  - {month_date[:-2]} -> Spent time: {execution_time} seconds.")
#     pass


def generate_one_month_CyL_for_testing(crop_name, domain_side, chaotic_percentage, threshold_to_save):
    start_time = time.time()
    month_date = year_month_list[1] 
    crop_deck = load_crop_decks_from_combined_pickles('CyL', month_date, crop_name, '', PickleCombination.ALL_MONTH)
    coverture = 0.0
    adaptative_domain_side = domain_side
    while coverture < threshold_to_save:
        print(f'- Current target resolution: {adaptative_domain_side}')
        coverture, _ = chaotic_aggregation(crop_deck, adaptative_domain_side, chaotic_percentage, threshold_to_save, 'CyL_and_Cat', month_date, flag_view=False, flag_view_save=True)
        if coverture < threshold_to_save:
            adaptative_domain_side = domain_size_down(adaptative_domain_side)
    print(f'Coverture : {coverture}   Size: {adaptative_domain_side}')
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{crop_name}  - {month_date[:-2]} -> Spent time: {execution_time} seconds.")
    pass

def debug_show_bands(crop_name, subset_str, date_str):
    month_date = year_month_list[1] 
    crop_deck = load_crop_decks_from_combined_pickles('CyL', month_date, crop_name, '', PickleCombination.ALL_MONTH)
    one_crop = crop_deck.crop_in_region[0].array3D
    for i in range(one_crop.shape[2]):
        plt.figure()
        plt.imshow(np.squeeze(one_crop[:,:,i]))
        plt.title(ALL_BAND_CODE[i])
        plt.colorbar()
        plt.show()




if __name__ == "__main__":  
    phase_ID = 2
    dataset_ID = 0  # 0: CyL  0: Catalonia
    domain_side =  1536 + 10 #1024*2 + 10 #1024 + 10 # 512 + 10 # 256 + 10 #  512 + 10 # 1024 + 20 # 512
    threshold_to_save = 99.999
    crop_index = 0  # Change to select the desired crop type
    chaotic_percentage = 97

    crop_name = Romania_Crop_Codes[crop_index][0]
    crop_type_ID = Romania_Crop_Codes[crop_index][1]

    # First Phase: piece generation
    if phase_ID == 1:
        if dataset_ID == 0: # CyL
            subset_str = 'CyL'
            mainS2path = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/S2Images_Oct2017-Sep2018_T30TUM/S2_CroppedAOI_WithCloudMask/'
            mainS1path = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/S1Images_Oct2017-Sep2018_T30TUM/Monthly-Composites/'
            cat_tile=''
            start_time = time.time()
            for date_str in CyL_date_S2_list:
                for crop_index in range(9):
                    crop_name = Romania_Crop_Codes[crop_index][0]
                    crop_type_ID = Romania_Crop_Codes[crop_index][1]
                    Initial_CyL_Sentinel_analysis(subset_str, crop_name, crop_type_ID, date_str, mainS2path, mainS1path)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Spent time: {execution_time} seconds.")
        else: # Catalonia
            subset_str = 'Cat'
            start_time = time.time()
            for cat_tile in CatTiles:
                # cat_tile = CatTiles[3]
                for date_str in Cat_date_S2_dic[cat_tile]: # Dates depends on Tiles
                    for crop_index in range(9):
                        crop_name = Romania_Crop_Codes[crop_index][0]
                        crop_type_ID = Romania_Crop_Codes[crop_index][1]
                        mainS2path = f'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/ComplementaryData_Catalonia/S2Images_Oct2017-Sep2018/{cat_tile}/'
                        mainS1path = f'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/ComplementaryData_Catalonia/S1Images_Oct2017-Sep2018/{cat_tile}/Monthly-Composites/'
                        Initial_Catalonia_Sentinel_analysis(subset_str, crop_name, crop_type_ID, date_str, mainS2path, mainS1path, cat_tile)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Spent time: {execution_time} seconds.")


    # if dataset_ID == 0:
    #     subset_str = 'CyL'
    #     date_str = CyL_date_S2_list[1]
    #     cat_tile = ''
    # else:
    #     subset_str = 'Cat'
    #     cat_tile = CatTiles[0]
    #     date_str = Cat_date_S2_dic[cat_tile][1]
    # mode_p = PickleCombination.ALL_MONTH_BOTH_DS
    # # crop_deck = load_crop_decks_from_single_pickle(subset_str, date_str, crop_name, cat_tile)
    # start_time = time.time()
    # crop_deck = load_crop_decks_from_combined_pickles(subset_str, date_str, crop_name, cat_tile, mode_p)
    # coverture, _ = chaotic_aggregation(crop_deck, domain_side, chaotic_percentage, threshold_to_save, flag_view=True, flag_view_save=True)
    # print(f'Coverture : {coverture}')
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Spent time: {execution_time} seconds.")
    
    # Uncomment next line!:
    # Second Phase: puzzle assembly
    if phase_ID == 2:
        # generate_one_month_CyL_for_testing(crop_name, 128+32, chaotic_percentage, threshold_to_save)
        generate_all_months_composed_DS_textures(crop_name, domain_side, chaotic_percentage, threshold_to_save)

    # if phase_ID == 3: # just for intermediate debugging
    #     date_str = CyL_date_S2_list[0]
    #     subset_str = 'CyL'
    #     debug_show_bands(crop_name, subset_str, date_str)












