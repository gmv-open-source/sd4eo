# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     testPlotAlignment.py
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
# This is just a small script to study what's happening with crops in different bands

import copy
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import rasterio
from rasterio.mask import mask
from scipy.ndimage import distance_transform_edt
import xarray as xr

NAN_THRESHOLD = -9990
MIN_CROP_SIZE_PIXELS = 3
S2BAND_CODE = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']
Romania_Crop_Codes = [('Barley', 5), ('Wheat', 1), ('OtherGrainLeguminous', 39), ('Peas' , 40), ('FallowAndBareSoil', 21), ('Vetch', 52), ('Alfalfa', 60), ('Sunflower', 33), ('Oats' , 8)]

def test_geotiff_transformations_and_crs():
    base_full_filename = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/S2Images_Oct2017-Sep2018_T30TUM/S2_CroppedAOI_WithCloudMask/20171029/SD4EO_Spain_Crops_20171029_{}_CLM_10m.tif'
    list_geotiff_filenames = []
    for band_name in S2BAND_CODE:
        geotiff_filename  = base_full_filename.format(band_name)
        list_geotiff_filenames.append(geotiff_filename)
    matching_files, non_matching_files = check_geotiff_transformations_and_crs(list_geotiff_filenames)
    print("# Matching files:")
    for filename in matching_files:
        print(Path(filename).name)
    print("# Non matching files:")
    for filename in non_matching_files:
        print(Path(filename).name)      

def extract_polygons_by_crop_type_ID(shapefile, crop_type_ID):
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



def check_and_extract_rastered_polygons(unsorted_polygons, crs_shapefile, mainS2path, base_filename, flag_sort_by_area = False, flag_save_tif_files=False, flag_display=False):

    # Step 1: Open the GeoTIFF file with rasterio
    rastered_polygons = []
    rastered_polygons_for_this_band = []
    inner_distance_crops = []
    max_num_bands = len(S2BAND_CODE)
    crs_raster = None
    master_projection = None

    for band_index in [0,3]:
        band_name = S2BAND_CODE[band_index]
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
                if sorted_crop_index == 4:
                    break
                # print (f"Area of crop : {polygon['Area_ha']}")
                # Step 3: Extract the portion of the image corresponding to the polygon
                out_image, out_transform = mask(src, [polygon['geometry']], crop=True, filled=False)
                # Convert the image portion to a numpy array
                out_image_array = out_image.data
                
                # # Is it a numpy array ?
                print(out_image.shape)
                print(out_image)
                print(np.max(np.max(out_image_array)))

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
                plt.figure()
                plt.imshow(np.squeeze(np_out_image_array), interpolation='none')
                plt.colorbar()
                plt.title(f'crop ID: {_}  Band: {band_name}  {np_out_image_array.shape[1]} x {np_out_image_array.shape[2]}')
                if crop_num_pixels > MIN_CROP_SIZE_PIXELS:       
                    # The single layer/band crop
                    single_layer_crop = np.squeeze(np_out_image_array) 
                    if band_index == 0: # It is the same for all bands
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
    plt.show()    
    return rastered_polygons, inner_distance_crops




if __name__ == "__main__":
    # test_geotiff_transformations_and_crs()
    domain_side = 512 + 10 #1024 512
    crop_index = 7
    date_str = '20171029'
    mainS2path = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/S2Images_Oct2017-Sep2018_T30TUM/S2_CroppedAOI_WithCloudMask/'

    crop_name = Romania_Crop_Codes[crop_index][0]
    crop_type_ID = Romania_Crop_Codes[crop_index][1]

    # Load the shapefile
    # shapefile_full_path = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/Fields_GTData/SD4EO_GTData_Testing_32630.shp'  # I took the testing set because it is smaller, later I'll switch to the training one 
    shapefile_full_path = 'C:/DATASETs/_SD4EO.original/01_CropUseCase/00_CropFields_Dataset_TO_USE/Fields_GTData/SD4EO_GTData_Training_32630.shp'  
    base_filename = 'SD4EO_Spain_Crops'

    shapefile_gdf = gpd.read_file(shapefile_full_path)

    # Extract polygons for a specific crop type, e.g., "Wheat"
    crop_polygons = extract_polygons_by_crop_type_ID(shapefile_gdf, crop_type_ID)
    
    # Print the result or do further processing
    print(crop_polygons)

    # GOAL: Identify cloud mask value --> -9999
    # plot_geotiff_histogram(geotiff_full_path)

    # GOAL: I want to be able to extract the rastered polygon mask and the content for each crop, time sample and layer. It is not easy, because CRS must be compatible
    unsorted_polygons = crop_polygons
    # Let's check the reference system
    crs_shapefile = unsorted_polygons.crs
    rastered_polygons, inner_distance_crops = check_and_extract_rastered_polygons(unsorted_polygons, crs_shapefile, mainS2path, base_filename, flag_sort_by_area = True, flag_save_tif_files=False)
    # Save work up to here to split the process and speed up development and tests

