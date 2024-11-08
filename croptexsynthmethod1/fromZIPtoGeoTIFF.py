#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     fromZIPtoGeoTIFF.py
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
# This is a small script that invokes GDAL executables to convert JP2 images into GeoTIFF files,
# band by band, enabling the extraction of rasterized polygons with the desired bands afterward. 
# This is necessary due to the packaging of files downloaded from the Copernicus Browser into 
# a ZIP file (which has a peculiar structure).

import os
import subprocess

GDAL_PATH = 'C:/Portables/otroQGIS 3.34.3/bin/'

def check_CRS_info(jp2_full_filename):
    # & 'C:\Portables\otroQGIS 3.34.3\bin\gdalinfo.exe' .\T11SPS_20240110T182729_B02_10m.jp2
    executable_full_path = GDAL_PATH + 'gdalinfo.exe'
    result = subprocess.run([executable_full_path, jp2_full_filename], capture_output=True, text=True)
    print('CHECK CRS:')
    print('----------')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    print('')

def convert_from_jp2_to_geotiff(jp2_full_filename, geotiff_full_filename, flag_force_resolution=False, str_resx = '10980', str_resy = '10980'):
    # 'C:\Portables\otroQGIS 3.34.3\bin\gdal_translate.exe' -of GTiff .\T11SPS_20240110T182729_B02_10m.jp2 .\T11SPS_20240110T182729_B02_10m.tif
    executable_full_path = GDAL_PATH + 'gdal_translate.exe'
    if flag_force_resolution == False:
        result = subprocess.run([executable_full_path, '-of', 'GTiff', jp2_full_filename, geotiff_full_filename], capture_output=True, text=True)
    else:
        result = subprocess.run([executable_full_path, '-of', 'GTiff', '-outsize', str_resx, str_resy, '-r', 'bilinear', jp2_full_filename, geotiff_full_filename], capture_output=True, text=True)
    print('TRANSFORM JP2 to GeoTIFF:')
    print('----------')
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    print('')	


base_pkg_dat_paths = [['S2A_MSIL2A_20220105T182741_N0301_R127_T11SPS_20220105T205924.SAFE/GRANULE/L2A_T11SPS_A034158_20220105T182954/IMG_DATA/', 'T11SPS_20220105T182741'],['S2A_MSIL2A_20220204T182541_N0400_R127_T11SPS_20220204T215218.SAFE/GRANULE/L2A_T11SPS_A034587_20220204T182543/IMG_DATA/', 'T11SPS_20220204T182541'],['S2B_MSIL2A_20220301T182259_N0400_R127_T11SPS_20220301T220950.SAFE/GRANULE/L2A_T11SPS_A026036_20220301T183136/IMG_DATA/', 'T11SPS_20220301T182259'],['S2A_MSIL2A_20220405T181921_N0400_R127_T11SPS_20220406T000155.SAFE/GRANULE/L2A_T11SPS_A035445_20220405T182333/IMG_DATA/', 'T11SPS_20220405T181921'],['S2A_MSIL2A_20220505T181921_N0400_R127_T11SPS_20220506T005710.SAFE/GRANULE/L2A_T11SPS_A035874_20220505T182340/IMG_DATA/', 'T11SPS_20220505T181921'],['S2A_MSIL2A_20220604T181921_N0400_R127_T11SPS_20220605T000320.SAFE/GRANULE/L2A_T11SPS_A036303_20220604T183023/IMG_DATA/', 'T11SPS_20220604T181921'],['S2A_MSIL2A_20220704T181931_N0400_R127_T11SPS_20220705T002619.SAFE/GRANULE/L2A_T11SPS_A036732_20220704T183416/IMG_DATA/', 'T11SPS_20220704T181931'],['S2A_MSIL2A_20220823T181931_N0400_R127_T11SPS_20220826T162703.SAFE/GRANULE/L2A_T11SPS_A037447_20220823T183213/IMG_DATA/', 'T11SPS_20220823T181931'],['S2B_MSIL2A_20220917T182019_N0400_R127_T11SPS_20220917T222316.SAFE/GRANULE/L2A_T11SPS_A028896_20220917T182732/IMG_DATA/', 'T11SPS_20220917T182019'],['S2A_MSIL2A_20221002T182201_N0400_R127_T11SPS_20221003T001356.SAFE/GRANULE/L2A_T11SPS_A038019_20221002T182814/IMG_DATA/', 'T11SPS_20221002T182201'],['S2B_MSIL2A_20221106T182539_N0400_R127_T11SPS_20221106T210326.SAFE/GRANULE/L2A_T11SPS_A029611_20221106T182852/IMG_DATA/', 'T11SPS_20221106T182539'],['S2B_MSIL2A_20221206T182739_N0509_R127_T11SPS_20221206T204742.SAFE/GRANULE/L2A_T11SPS_A030040_20221206T182735/IMG_DATA/', 'T11SPS_20221206T182739']]


if __name__ == "__main__":
    bands_10m = ['B02', 'B03', 'B04', 'B08', 'TCI']
    bands_20m = ['B05', 'B06', 'B07', 'B11', 'B8A', 'SCL']
    sub_path_10m = 'R10m/'
    sub_path_20m = 'R20m/'
    full_output_path = 'C:/Temp/'
    base_input_img_path = 'C:/DATASETs/SD4EO.ImperialValley/' 
    # package_path = 'S2B_MSIL2A_20240110T182729_N0510_R127_T11SPS_20240110T205820.SAFE/GRANULE/L2A_T11SPS_A035760_20240110T182919/IMG_DATA/', 
    # base_code_name = 'T11SPS_20240110T182729'
    for package_boundle in base_pkg_dat_paths:
        package_path, base_code_name = package_boundle
        base_pkg_input_img_path = base_input_img_path + package_path
        print(base_code_name)
        # Loop for 10m bands
        for band_ID in bands_10m:
            print(band_ID)
            base_filename = f"{base_code_name}_{band_ID}_10m"
            full_image_path = base_pkg_input_img_path + sub_path_10m 
            jp2_full_filename = full_image_path + base_filename + '.jp2'
            geotiff_full_filename = full_output_path + base_filename + '.tif'
            convert_from_jp2_to_geotiff(jp2_full_filename, geotiff_full_filename, True)
        for band_ID in bands_20m:
            print(band_ID)
            input_base_filename = f"{base_code_name}_{band_ID}_20m"
            output_base_filename = f"{base_code_name}_{band_ID}_10m"
            full_image_path = base_pkg_input_img_path + sub_path_20m 
            jp2_full_filename = full_image_path + input_base_filename + '.jp2'
            geotiff_full_filename = full_output_path + output_base_filename + '.tif'
            convert_from_jp2_to_geotiff(jp2_full_filename, geotiff_full_filename, True)


    # check_CRS_info(geotiff_full_filename)
    pass




