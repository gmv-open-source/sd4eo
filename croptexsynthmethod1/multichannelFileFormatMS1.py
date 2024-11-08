#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     multichannelFileFormatMS1.py
# Author:        David Miraut
# License:       MIT License
# 
# The following script is the result of mixing and adapting several works, specially the one made by Tetsuya Odaka, 
# and source code pieces, some of them coded in matlab, to python code, in order to achieve the purpose described
# below. Here are the references to those works:
# https://github.com/rohitrango/Image-Quilting-for-Texture-Synthesis
# https://github.com/PJunhyuk/ImageQuilting
# https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli
# https://github.com/ganguli-lab/textureSynth
# https://github.com/plenoptic-org/plenoptic

# All of them are licensed by the terms of the MIT License
 
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
# Just a bunch of auxiliar functions for our xarray files with Sentinel-2 bands
# This format is now deprecated, it was used for the preliminary dataset in MS1


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


bands_IDs = ['B02','B03','B04','B05','B06','B07','B08','B11','B8A', 'SCL','TCI']
bands_IDs9 = ['B02','B03','B04','B05','B06','B07','B08','B11','B8A']

def show_mcArray(numpy_array, max_visible_value=0):
    rgb_array = np.zeros((numpy_array.shape[0], numpy_array.shape[1], 3))
    rgb_array[:,:,0] = numpy_array[:,:,2] # R
    rgb_array[:,:,1] = numpy_array[:,:,1] # G
    rgb_array[:,:,2] = numpy_array[:,:,0] # B
    if max_visible_value == 0:
        max_visible_value = np.max(np.max(rgb_array))
    rgb_array = rgb_array * 1.0/max_visible_value
    plt.imshow(rgb_array)
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
