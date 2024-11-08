# File Name:     test_crop_v2.py
# License:       Apache 2.0 License
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
		
# Description: 
# Import necessary libraries
# from qgis.core import QgsRasterLayer, QgsApplication

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_geom
from shapely.geometry import box
import time


from common_tile_utils import available_Tiles_IDs, provide_inner_tile_coords

S2bands = ['B02', 'B03', 'B04', 'B08']

def load_and_clip_raster(raster_path, output_path, bbox):
    """
    Load a raster layer using QgsRasterLayer and clip it using rasterio.mask.

    Args:
        raster_path (str): Path to the input raster file.
        output_path (str): Path to save the clipped raster file.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) for clipping.
    """
    # Read the raster file using rasterio
    with rasterio.open(raster_path) as src:
        # Define the bounding box as a shapely geometry
        minx, miny, maxx, maxy = bbox
        geom = box(minx, miny, maxx, maxy)
        # Create a GeoDataFrame with the bounding box
        gdf = gpd.GeoDataFrame({'geometry': [geom]}, crs=src.crs) #.authid()
        print('CRS:')
        print(src.crs)

        # Clip the raster using the bounding box
        out_image, out_transform = mask(src, gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Save the clipped raster to the output path
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

def check_4arrays2D_same_size(arr1, arr2, arr3, arr4):
    """
    Checks if all four 2D numpy arrays have the same dimensions.
    
    Args:
        arr1 (numpy.ndarray): First 2D array.
        arr2 (numpy.ndarray): Second 2D array.
        arr3 (numpy.ndarray): Third 2D array.
        arr4 (numpy.ndarray): Fourth 2D array.
    
    Returns:
        bool: True if all arrays have the same dimensions, False otherwise.
    """
    # Get the shape of the first array
    shape = arr1.shape
    # Compare the shape of the first array with the shapes of the other three arrays
    return shape == arr2.shape == arr3.shape == arr4.shape


def check_arrays_same_size(arrays):
    """
    Checks if all 2D numpy arrays in the list have the same dimensions.
    
    Args:
        arrays (list of numpy.ndarray): List of 2D arrays.
    
    Returns:
        bool: True if all arrays have the same dimensions, False otherwise.
    """
    if not arrays:
        # Return False if the list is empty
        return False
    
    # Get the shape of the first array
    first_shape = arrays[0].shape
    
    # Compare the shape of the first array with the shapes of the other arrays
    return all(array.shape == first_shape for array in arrays)

def find_smallest_dimensions(arrays):
    """
    Finds the smallest number of rows and columns among all 2D numpy arrays in the list.
    
    Args:
        arrays (list of numpy.ndarray): List of 2D arrays.
    
    Returns:
        tuple: A tuple containing the smallest number of rows and the smallest number of columns.
    """
    if not arrays:
        # Return None if the list is empty
        return None
    
    # Initialize min_rows and min_cols with the shape of the first array
    min_rows, min_cols = arrays[0].shape
    
    # Iterate over the arrays to find the smallest dimensions
    for array in arrays:
        rows, cols = array.shape
        if rows < min_rows:
            min_rows = rows
        if cols < min_cols:
            min_cols = cols
    
    return (min_rows, min_cols)

def S2Tilecookiecutter(S2input_base_path, S2base_filename, patch_output_base_path, extent_full_filename, currentEPSG, tile_id):
    """
    Extracts data from Sentinel-2 satellite images for specified rectangular areas:
    Load the selected S2 bands from a Tile, cuts the extent polygons and save the 
    pixel content as a numpy array into small files, which have the same codename
    than the corresponding OSM patches

     Args:
        S2input_base_path (str): route to the folder with S2 Tile JP2 files
        S2base_filename (str): incomplete base filename
        patch_output_base_path (str): folder where the numpy files will be stored
        extent_full_filename (str): full path to the file with the array of extents.
                                    containing coordinates of rectangles in the format 
                                    [min_x, min_y, max_x, max_y].
        currentEPSG (str): Coordinate system of stored extents
        tile_id (str): for new filenames
    
    Returns:
        It writes a file with the content of each band
    """
    # Get the extent data structure
    with np.load(extent_full_filename) as extent_file:
        extent_reg = extent_file['extent_reg']
    patch_n_cols = extent_reg.shape[0]
    patch_n_rows = extent_reg.shape[1]
    print(extent_reg.shape)
    # print(extent_reg)
    # plt.figure()
    # plt.imshow(np.squeeze(extent_reg[:,:,0]))
    # plt.colorbar()
    # plt.show()

    # Initialize data structures
    full_S2band_filenames = []
    result_by_band = {}
    for band_name in S2bands:
        full_S2band_filenames.append(S2input_base_path + S2base_filename.format(band_name=band_name))
        result_by_band[band_name] = []  # Initialize dictionary to store results
    # DEBUG
    for band_filename in full_S2band_filenames:
        print(band_filename)

    try:
        # Open all band files at once
        with rasterio.open(full_S2band_filenames[0]) as file_B02, \
             rasterio.open(full_S2band_filenames[1]) as file_B03, \
             rasterio.open(full_S2band_filenames[2]) as file_B04, \
             rasterio.open(full_S2band_filenames[3]) as file_B08:

            # Ensure CRS matches for all files and the rectangles
            print(file_B02.crs)
            src_crs = file_B02.crs
            if not all(src.crs == src_crs for src in [file_B03, file_B04, file_B08]):
                raise ValueError("CRS mismatch among the input files.")
            
            # Process each geometry
            for col_i in range(patch_n_cols):
                print(f'{tile_id}  cols -> {col_i} / {patch_n_cols}')
                for row_i in range(patch_n_rows):
                    # print(f'{tile_id}  cols -> {col_i} / {patch_n_cols}   rows -> {row_i} / {patch_n_rows}')
                    min_x_i_patch = extent_reg[col_i, row_i, 0]
                    min_y_i_patch = extent_reg[col_i, row_i, 1]
                    max_x_i_patch = extent_reg[col_i, row_i, 2]
                    max_y_i_patch = extent_reg[col_i, row_i, 3]
                    geom_patch = box(min_x_i_patch, min_y_i_patch, max_x_i_patch, max_y_i_patch)
                    # Check CRSs
                    if src_crs != currentEPSG:
                        # Let's apply the needed transformation
                        # Typically from EPSG:3857 to EPSG:32633
                        # print()
                        # print(geom_patch)
                        geom_patch = transform_geom(currentEPSG, src_crs, geom_patch.__geo_interface__)
                        # print(geom_patch)
                    # Cut for all the bands: 'B02', 'B03', 'B04', 'B08'
                    try:
                        out_image_B02, out_transform = mask(file_B02, [geom_patch], crop=True, filled=False) #, nodata=np.nan)
                        out_image_B03, out_transform = mask(file_B03, [geom_patch], crop=True, filled=False) #, nodata=np.nan)
                        out_image_B04, out_transform = mask(file_B04, [geom_patch], crop=True, filled=False) #, nodata=np.nan)
                        out_image_B08, out_transform = mask(file_B08, [geom_patch], crop=True, filled=False) #, nodata=np.nan)
                    except ValueError as e:
                        print(f"Skiped patch [{col_i}, {row_i}] because: {e}")
                        continue
                    # convert to numpy arrays
                    patch_B02 = np.array(out_image_B02[0])
                    patch_B03 = np.array(out_image_B03[0])
                    patch_B04 = np.array(out_image_B04[0])
                    patch_B08 = np.array(out_image_B08[0])
                    # Check the size is the same, if not we have to cut it a bit
                    # in order to avoid inventing data (?)
                    list_patch_bands = [patch_B02, patch_B03, patch_B04, patch_B08]
                    if check_arrays_same_size(list_patch_bands):
                        new_3D_array = np.zeros((patch_B02.shape[0], patch_B02.shape[1], len(list_patch_bands)))
                        for i, patch_for_band_i in enumerate(list_patch_bands):
                            new_3D_array[:,:,i] = patch_for_band_i[:,:]
                    else:
                        min_rows, min_cols = find_smallest_dimensions(list_patch_bands)
                        new_3D_array = np.zeros((min_rows, min_cols, len(list_patch_bands)))
                        for i, patch_for_band_i in enumerate(list_patch_bands):
                            new_3D_array[:,:,i] = patch_for_band_i[:min_rows,:min_cols]  
                    # Save the numpy array as a file
                    np_filename = patch_output_base_path + f'patch_{tile_id}_{col_i:03}_{row_i:03}_S2'
                    np.savez(np_filename, array3D=new_3D_array)

                    # out_meta = file_B08.meta.copy()
                    # out_meta.update({
                    #     "driver": "GTiff",
                    #     "height": out_image_B08.shape[1],
                    #     "width": out_image_B08.shape[2],
                    #     "transform": out_transform
                    # })
                    # # Save the clipped raster to the output path
                    # with rasterio.open(np_filename + '.tif', "w", **out_meta) as dest:
                    #     dest.write(out_image_B08)
                    
                    # plt.figure()
                    # plt.imshow(patch_B08)
                    # plt.colorbar()
                    # plt.show()                    

    except RasterioIOError as e:
        print(f"Error reading file: {e}")
        return None


    # # Convert lists to numpy arrays for each band
    # for band in result:
    #     result[band] = np.array(result[band])

    # return result




if __name__ == "__main__":
    start_time = time.time()
    for tile_id in available_Tiles_IDs[2:]:
        icoords_EPGS3857, icoords_EPGS4326, base_filename = provide_inner_tile_coords(tile_id)
        input_base_path = f'C:/DATASETs/UC2Training/Sentinel2/{tile_id}/'
        output_base_path = f'C:/DATASETs/UC2Training/npS2Patches/{tile_id}/'
        extent_full_filename = f'C:/DATASETs/UC2Training/npExtents/np_extent_{tile_id}_EPSG_3857.npz'
        defaultEPSG = 'EPSG:3857'
        print(base_filename)

        S2Tilecookiecutter(input_base_path, base_filename, output_base_path, extent_full_filename, defaultEPSG, tile_id)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")


