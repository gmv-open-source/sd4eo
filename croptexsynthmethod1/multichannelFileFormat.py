#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
# New version for the netCDF file format with one labelled xarray


from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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

@dataclass
class MultiChannelFileMetadata:
    crop_type_name : str = ''
    max_value : float = 0.0           # Maximum value in all the bands
    max_visible_value : float = 0.0   # Recorded maximum value for RGB bands
    date : str = ''
    dataset : str = ''                # Original dataset of the real data
    synthetic_method : str = ''
    coverture : float = 100.0  # It is supposed to be 100% in all cases in these stages
    reference_band : str = 'B02'
    bands_IDs : list = None
    band_min : list = None
    band_max : list = None
    flag_normalization : bool = False
    
bands_IDs = ['B02','B03','B04','B05','B06','B07','B08','B11','B8A', 'SCL','TCI']
bands_IDs9 = ['B02','B03','B04','B05','B06','B07','B08','B11','B8A']

def show_mcArray(numpy_array, max_visible_value=0, title_str=''):
    rgb_array = np.zeros((numpy_array.shape[0], numpy_array.shape[1], 3))
    rgb_array[:,:,0] = numpy_array[:,:,2] # R
    rgb_array[:,:,1] = numpy_array[:,:,1] # G
    rgb_array[:,:,2] = numpy_array[:,:,0] # B
    if max_visible_value == 0:
        max_visible_value = np.max(np.max(rgb_array))
    rgb_array = rgb_array * 1.0/max_visible_value
    plt.imshow(rgb_array)
    plt.colorbar()
    if title_str != '':
        plt.title(title_str)
    plt.show()


def load_NC_patch(full_filename, crop_margin = 0, flag_show=False):
    data_xarray = xr.open_dataarray(full_filename, engine='netcdf4')
    if crop_margin > 0:
        full_np_array = data_xarray.values
        # numpy_array = full_np_array[crop_margin:-crop_margin,crop_margin:-crop_margin,:3] # Esto hace que sólo trabajemos con 3 bandas en el quilting y el resultado no se vea influenciado por las otras bandas que distorsionan el resultado en la zona "visible"
        numpy_array = full_np_array[crop_margin:-crop_margin,crop_margin:-crop_margin,:]
    else:
        numpy_array = data_xarray.values
    max_visible_value = data_xarray.attrs.get('max_visible_value')
    data_xarray.close()
    if max_visible_value is None:
        max_visible_value = np.max(np.max(np.max(numpy_array[:,:,:3])))
    max_value = np.max(np.max(np.max(numpy_array)))
    print(f" size : {numpy_array.shape}  max visible: {max_visible_value}  max total: {max_value}")
    if flag_show:
        show_mcArray(numpy_array)
    return numpy_array, max_value, max_visible_value


def save_new_NC_patch(base_3D_array, full_output_path, max_visible_value=0):
    # Transform as a xarray
    num_bands_in_patch = base_3D_array.shape[2]
    if num_bands_in_patch == len(bands_IDs):
        xr_array = xr.DataArray(base_3D_array, dims=['y', 'x', 'band'], coords={'band': bands_IDs})
    else:
        xr_array = xr.DataArray(base_3D_array, dims=['y', 'x', 'band'], coords={'band': bands_IDs[:num_bands_in_patch]})
    xr_array.attrs['max_visible_value'] = max_visible_value
    xr_array.to_netcdf(full_output_path)  # f"./pieces/piece_{polygon_index}_{month_i}.nc")


def normalize_array(array2D, max_val, min_val):
    """
    Normalizes the values in a 2D array to the range [0, 1] based on provided maximum and minimum values.
    
    Args:
    array2D (np.ndarray): A 2D numpy array containing numerical data.
    max_val (float): The maximum value used for normalization.
    min_val (float): The minimum value used for normalization.
    
    Returns:
    np.ndarray: A 2D numpy array where all values are normalized to the range [0, 1].
    """
    # Ensure the input array is a numpy array
    array2D = np.asarray(array2D)
    
    # Perform the linear normalization
    normalized_array = (array2D - min_val) / (max_val - min_val)
    
    return normalized_array



def load_from_netCDF(full_filename, crop_margin = 0, flag_normalization=False, flag_show=False):
    """
    Load the synthetized multiespectral image as a multidimensional numpy array from a 
    xarray labelled data structure, which is stored in a standarized netCDF file, so it 
    will be used easily by the next algorithms in family 2.x
    Additional useful information is also recorded as metadata.

    Args:
        full_filname (str): Full path where the file to load is in the HD
        crop_margin (str):  Excluded margin at tboth sides in each band
                            It is maintained to ensure retro-compatibilty with rotation
                            issues in California dataset
        flag_show (bool):   Activate the graphic representation of the visible bands,
                            mainly used for debugging purposes

    Returns:
        numpy_array (np.array):    The content of the file as a numpy 3D array
        max_value (float):         Maximum value in all the bands
        max_visible_value (float): Reported maximum value in the RGB bands, 
                                   it may not be the current max value, but 
                                   it must be preserved to represent the content 
                                   with the same dynamic range
        metadata (MultiChannelFileMetadata):   Other useful pieces of data
    """
    with xr.open_dataarray(full_filename, engine='netcdf4') as data_xarray:
        if crop_margin > 0:
            full_np_array = data_xarray.values
            # numpy_array = full_np_array[crop_margin:-crop_margin,crop_margin:-crop_margin,:3] # Esto hace que sólo trabajemos con 3 bandas en el quilting y el resultado no se vea influenciado por las otras bandas que distorsionan el resultado en la zona "visible"
            numpy_array = full_np_array[crop_margin:-crop_margin,crop_margin:-crop_margin,:]
        else:
            numpy_array = data_xarray.values
        # Extract the attributes
        max_visible_value = data_xarray.attrs.get('max_visible_value')
        crop_type_name = data_xarray.attrs.get('long_name', 'unknown_crop_type')
        date = data_xarray.attrs.get('date', 'unknown_date')
        dataset = data_xarray.attrs.get('dataset', 'unknown_dataset')
        synthetic_method = data_xarray.attrs.get('dataset', 'real_composed_data')
        coverture = data_xarray.attrs.get('coverture', 100.0)
        reference_band = data_xarray.attrs.get('reference_band', 'B02')
        if max_visible_value is None:
            max_visible_value = np.max(np.max(np.max(numpy_array[:,:,:3])))
    if flag_show:
        show_mcArray(numpy_array, max_visible_value, f"{crop_type_name}  {date}")            
    # Let's collect the max and min values of each band
    num_bands = numpy_array.shape[2]
    band_min_list = []
    band_max_list = []
    for band_i in range(num_bands):
        band_max_value = np.max(np.max(numpy_array[:,:,band_i]))
        band_min_value = np.min(np.min(numpy_array[:,:,band_i]))
        band_max_list.append(band_max_value)
        band_min_list.append(band_min_value)
        if flag_normalization:
            # We fit the dynamic range to [0,1]
            numpy_array[:,:,band_i] = normalize_array(numpy_array[:,:,band_i], band_max_value, band_min_value)

    max_value = np.max(band_max_value)
    # We extract the rest of potentially useful data
    metadata = MultiChannelFileMetadata(crop_type_name, max_value, max_visible_value, date, dataset, synthetic_method, coverture, reference_band, ALL_BAND_CODE, band_min_list, band_max_list, flag_normalization)

    # print(f" size : {numpy_array.shape}  max visible: {max_visible_value}  max total: {max_value}")
    return numpy_array, max_value, max_visible_value, metadata


def reverse_normalize_array(normalized_array, max_val, min_val):
    """
    Reverses the normalization of a 2D array from the [0, 1] range back to the original [min_val, max_val] range.
    
    Args:
    normalized_array (np.ndarray): A 2D numpy array with values normalized to the range [0, 1].
    max_val (float): The original maximum value before normalization.
    min_val (float): The original minimum value before normalization.
    
    Returns:
    np.ndarray: A 2D numpy array with values restored to the original range [min_val, max_val].
    """
    # Ensure the input array is a numpy array
    normalized_array = np.asarray(normalized_array)
    
    # Reverse the normalization process
    original_array = normalized_array * (max_val - min_val) + min_val
    
    return original_array


def save_as_netCDF(base_3D_array, full_output_path, crop_name, date_str, subset_str, synthetic_method='', max_visible_value=0, metadata=None, flag_revert_normalization=False):
    """
    Save the synthetized multiespectral image as a texture in a xarray labelled data
    structure, which is stored in a standarized netCDF file, so it will be used easily
    by the next algorithms in family 2.x
    Additional useful information is also recorded as metadata.

    Args:
        base_3D_array (np.array): multidimensional (multi-espectral) 3D array
        full_output_path (str): full path to save the file
        crop_name (str): crop name according to ROMANIA (Sen4AgriNet) nomenclature
        date_str (str): reference date (it could be just a year-month)
        subset_str (str): name of the original dataset
        synthetic_method (str): name of the method used for generating the synthetic data
        max_visible_value (float): to allow a more realistic visualization in RGB bands

    Returns:
        The output is a file stored in the hard disk according to prior naming rules

    Warning:
        This function may potentially denormalize base_3D_array, so it is designed to be used 
        at the end of the execution
    """
    num_bands_in_patch = base_3D_array.shape[2]
    if (metadata is not None and metadata.flag_normalization) or flag_revert_normalization:
            # Let's recover the dynamic range
            for band_i in range(num_bands_in_patch):
                max_val = metadata.band_max[band_i]
                min_val = metadata.band_min[band_i]
                base_3D_array[:,:,band_i] = reverse_normalize_array(base_3D_array[:,:,band_i], max_val, min_val)
            metadata.flag_normalization = False

    if num_bands_in_patch == len(ALL_BAND_CODE):
        composed_crop = xr.DataArray(base_3D_array, dims=("x", "y", "band"), coords={"band": ALL_BAND_CODE})
    else:
        composed_crop = xr.DataArray(base_3D_array[:,:,num_bands_in_patch], dims=("x", "y", "band"), coords={"band": ALL_BAND_CODE[:num_bands_in_patch]})
    composed_crop.attrs["long_name"] = crop_name
    composed_crop.attrs["date"] = date_str
    composed_crop.attrs["dataset"] = subset_str
    composed_crop.attrs["synthetic_method"] = synthetic_method
    composed_crop.attrs['max_visible_value'] = max_visible_value
    if metadata is not None:
        # Additional attributes
        composed_crop.attrs["coverture"] = metadata.coverture

    filename_nc = full_output_path  # f'.\{subset_str}\{date_str}_{crop_name}_{effective_res}.nc'
    composed_crop.to_netcdf(filename_nc)
    print(f'Saved :   {filename_nc}')

