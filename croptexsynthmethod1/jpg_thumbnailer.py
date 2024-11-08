#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     jpg_thumbnailer.py
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
# This script scans a folder for .nc files containing xarrays of crop surfaces from 2.x texture synthesis algorithms, and generates thumbnails in JPG format of the bands in the visible spectrum. 
# It applies dynamic range correction specified by the max_visible_value attribute to ensure a color space similarity when comparing images across the synthetic generation pipeline. 

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from scipy.stats import skew, kurtosis
import os
import glob

# from PIL import Image
# import sys, os
# import logging
# import argparse, copy
# import time
# import sutils
# import steerable_pyramid as steerable
# import texture_analysis as ta
from multichannelFileFormatMS1 import load_NC_patch


def find_nc_files(folder):
    """Recursively searches for all files with .nc extension in the specified directory and its subdirectories.

    Args:
        directory (str): The root directory where the search begins.

    Returns:
        list: A list with the full paths of the .nc files found.
    """
    # Build the search path to include all .nc files in directory and subdirectories
    search_path = os.path.join(folder, '**', '*.nc')
    
    # Use glob with the recursive=True parameter to search in subdirectories
    nc_files = glob.glob(search_path, recursive=True)
    
    return nc_files



def convert_into_jpg(full_input_nc_filename, full_output_jpg_filename, flag_show=False):
    numpy_array, max_value, max_visible_value = load_NC_patch(full_input_nc_filename, flag_show=False)
    rgb_array = np.zeros((numpy_array.shape[0], numpy_array.shape[1], 3))
    rgb_array[:,:,0] = numpy_array[:,:,2] # R
    rgb_array[:,:,1] = numpy_array[:,:,1] # G
    rgb_array[:,:,2] = numpy_array[:,:,0] # B
    # if max_visible_value == 0:
    max_visible_value = np.max(np.max(rgb_array))
    rgb_array = rgb_array * 1.0/max_visible_value
    plt.imsave(full_output_jpg_filename, rgb_array)
    if flag_show:
        plt.imshow(rgb_array)
        plt.colorbar()
        plt.show()
    pass


if __name__ == "__main__":
    # target_folder = 'C:/DMTA/projects/SD4EO/croptexsynthmethod1/synthetic_parcels/'
    target_folder = 'C:/DATASETs/OUT_HighOrder/'

    input_filenames = find_nc_files(target_folder)
    for input_filename in input_filenames:
        print(input_filename)
        output_filename = input_filename[:-3] + '_onlyRGBpreview.jpg'
        convert_into_jpg(input_filename, output_filename, flag_show=False)
    pass





