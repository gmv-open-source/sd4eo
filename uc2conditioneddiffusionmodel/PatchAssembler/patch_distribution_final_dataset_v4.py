# File Name:     patch_distribution_final_dataset_v4.py
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
# This script takes the original synthetized patches after applying 
# the conditioned # Diffusion Model on the OSM patches, and transforms them 
# into the final v3 deliverable release of UC2 dataset
#
# We also separate them into two main folders:
# - MU: medium-sized urban structures
# - RBU: rural and other bigger-sized urban structures
# which will be packed into different ZIP files
# 
# We consider NIR + RGB + 2 kinds of masks
#
# osm_assembly.py must be executed first to generate puzzle npz files

import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from skimage import io, util
import scipy.ndimage
import xarray as xr
import os

# available_cities_IDs = ['Madrid', 'Paris', 'Toulouse', 'Poitiers', 'Bordeaux', 'Limoges', 'Clermont-Ferrand', 'Troyes', 'Le Mans', 'Angers', 'Niort',] # 'Rouen']
available_cities_IDs = ['Toulouse', 'Poitiers', 'Bordeaux', 'Limoges', 'Clermont-Ferrand', 'Troyes', 'Le Mans', 'Angers', 'Niort',] # 'Rouen']


def get_mask_patch_asRGB(input_base_path, city_id, col, row):
    npz_filename = f'mask_{city_id}_c{col:03}_r{row:03}.npz'
    npz_file = np.load(input_base_path +  city_id + '/' + npz_filename)
    all_building_bmask = npz_file['all_building_binary_mask']
    non_residential_bmask = npz_file['non_residential_combined_mask']
    all_building_RGB = np.zeros((all_building_bmask.shape[0], all_building_bmask.shape[1],3))
    non_residential_RGB = np.zeros((non_residential_bmask.shape[0], non_residential_bmask.shape[1],3))
    for i in range(3):
        all_building_RGB[:,:,i] = all_building_bmask[:,:]*255.0
        non_residential_RGB[:,:,i] = non_residential_bmask[:,:]*255.0
    return all_building_RGB, non_residential_RGB


def save_rgb_image(image_array: np.ndarray, file_path: str) -> None:
    """Saves a 3D numpy array (RGB image) to disk as an image file.
    
    Args:
        image_array (np.ndarray): A 3D numpy array of type float with shape (height, width, 3).
                                  Values are expected to be in the range [0, 255].
        file_path (str): The path where the image will be saved.
        
    Raises:
        ValueError: If the input array is not a 3D array with 3 channels (RGB).
    """
    # Check if the input array has the correct shape
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("Input array must be a 3D array with shape (height, width, 3).")
    # Ensure the array values are within the expected range
    image_array = np.clip(image_array, 0, 255)
    # Convert the float array to uint8
    image_array = image_array.astype(np.uint8)
    # Create a PIL image from the numpy array
    image = Image.fromarray(image_array)
    # Save the image to the specified file path
    image.save(file_path)


def replace_rgb_with_g(input_path: str, output_path: str) -> None:
    """
    Opens a PNG file as a 3D NumPy array with RGB channels, replaces the R and B channels with the G channel,
    and saves the result as a new PNG file.

    Args:
        input_path (str): The path to the input PNG file.
        output_path (str): The path to save the modified PNG file.
    """
    # Open the image and convert it to a NumPy array
    image = Image.open(input_path)
    image_np = np.array(image)
    # Check if the image has 3 channels (RGB)
    if image_np.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")
    # Replace the R and B channels with the G channel
    g_channel = image_np[:, :, 1]
    image_np[:, :, 0] = g_channel  # Replace R with G
    image_np[:, :, 2] = g_channel  # Replace B with G
    # Convert the modified NumPy array back to an image
    modified_image = Image.fromarray(image_np)
    # Save the modified image
    modified_image.save(output_path)


def resize_image_with_filter(image: np.ndarray, new_size: tuple) -> np.ndarray:
    """
    Resizes a 3D numpy array representing an RGB image to a new specified size after applying a Gaussian filter.
    
    Args:
        image (np.ndarray): 3D numpy array of shape (height, width, 3) representing the input RGB image.
        new_size (tuple): Tuple (new_height, new_width) specifying the desired output image size.
        
    Returns:
        np.ndarray: 3D numpy array of shape (new_height, new_width, 3) representing the resized RGB image.
    """
    # Apply Gaussian filter to smooth the image and avoid aliasing artifacts
    # filtered_image = gaussian_filter(image, sigma=1)
    
    # Resize the filtered image to the new size
    resized_image = resize(image, new_size, anti_aliasing=True, mode='reflect')
    return resized_image


def file_exists(file_path):
    """Check if a file exists at the specified path.
    
    Args:
        file_path (str): The complete path to the file.
    
    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)


def load_png_as_rgb_array(file_name: str) -> np.ndarray:
    """Loads a PNG image file and returns it as an RGB numpy array.

    Args:
        file_name (str): The name of the PNG file to load.

    Returns:
        np.ndarray: The image as an RGB numpy array with shape (height, width, 3).
    """
    # Open the image file
    with Image.open(file_name) as img:
        # Convert the image to RGB if it is not already in that mode
        img = img.convert('RGB')
        # Convert the image to a numpy array
        rgb_array = np.array(img)*1.0

    return rgb_array

def scale_array(arr, min_val, max_val):
    """
    Scales the values in a 2D numpy array to a new range [min_val, max_val].

    Args:
        arr (np.ndarray): 2D array with values between 0.0 and 255.0.
        min_val (float): The minimum value of the new range.
        max_val (float): The maximum value of the new range.

    Returns:
        np.ndarray: The scaled 2D array with values between min_val and max_val.
    """
    # Check if the input array is 2D
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D")
    
    # Compute the minimum and maximum values of the input array
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # Scale the array to the range [0, 1]
    arr_normalized = (arr - arr_min) / (arr_max - arr_min)
    
    # Scale the normalized array to the new range [min_val, max_val]
    arr_scaled = arr_normalized * (max_val - min_val) + min_val
    
    return arr_scaled


def resize_array_with_antialiasing(input_array, new_rows, new_cols):
    """Resize a 2D array to new dimensions using antialiasing.
    
    Args:
        input_array (numpy.ndarray): The input 2D array with float values.
        new_rows (int): The desired number of rows in the output array.
        new_cols (int): The desired number of columns in the output array.
    
    Returns:
        numpy.ndarray: The resized 2D array with the specified dimensions.
    """
    # Calculate the zoom factors for rows and columns
    zoom_row = new_rows / input_array.shape[0]
    zoom_col = new_cols / input_array.shape[1]
    
    # Use scipy's zoom function to resize the array with antialiasing
    resized_array = scipy.ndimage.zoom(input_array, (zoom_row, zoom_col), order=3)
    
    return resized_array



if __name__ == '__main__':
    real_osm_res = 566
    target_osm_res = 512
    m10_res = int(target_osm_res/3)
    print(m10_res)
    band_dynamic_range = {'B08':(2560.0, 12580.0), 'B02': (400.0, 4440.0), 'B03': (740.0, 5200.0), 'B04': (460.0, 6300.0)}
    main_input_path = 'C:/DATASETs/UC2Synthesis/'
    xarray_output_path = 'C:/DATASETs/UC2Synthesis.v3/'
    preview_output_path = 'C:/DATASETs/UC2Synthesis.v3.preview/'
    mode_10 = 'synthetic_patches_10m/'
    mode_33 = 'synthetic_patches_3.3m/'
    selected_mus = 'medium_sized_urban_structures/'
    selected_others = 'other_urban_and_rural_structures/'

    for city_name in available_cities_IDs:
        puzzle = np.load(f'{city_name}_puzzle.npz')['puzzle']

        max_row, max_col = puzzle.shape
        max_variants = 50
        for row in range(max_row):
            for col in range(max_col):
                print(f'{city_name}: r {row} c {col}')
                selected_patch = puzzle[row, col]
                if selected_patch > 0:
                    medium_path = selected_mus
                else:
                    medium_path = selected_others
                # Resolution 3.3m PREVIEW
                all_building_RGB, non_residential_RGB = get_mask_patch_asRGB(main_input_path + 'Mask/', city_name, col, row)
                save_rgb_image(all_building_RGB, preview_output_path + mode_33 + medium_path + f'Masks/{city_name}/all_building_mask_{city_name}_c{col:03}_r{row:03}.png')
                save_rgb_image(non_residential_RGB, preview_output_path + mode_33 + medium_path + f'Masks/{city_name}/non_residential_building_mask_{city_name}_c{col:03}_r{row:03}.png')
                for var in range(max_variants):
                    NIR_name = f'SynthIR/{city_name}/synthS2_{city_name}_IRm034_c{col:03}_r{row:03}_i050_v{var:02}.png'
                    RGB_name = f'SynthRGB/{city_name}/synthS2_{city_name}_RGBm034_c{col:03}_r{row:03}_i050_v{var:02}.png'
                    if file_exists(main_input_path+RGB_name):
                        shutil.copyfile(main_input_path+RGB_name, preview_output_path + mode_33 + medium_path + f'RGB/{city_name}/RGBpatch_c{col:03}_r{row:03}_{var:02}.png')
                    if file_exists(main_input_path+NIR_name,):
                        replace_rgb_with_g(main_input_path+NIR_name, preview_output_path + mode_33 + medium_path + f'NIR/{city_name}/NIRpatch_c{col:03}_r{row:03}_{var:02}.png')
                # Resolution 10m PREVIEW 
                m10_all_building_RGB = resize_image_with_filter(all_building_RGB,(m10_res,m10_res))
                m10_non_residential_RGB = resize_image_with_filter(non_residential_RGB,(m10_res,m10_res))
                save_rgb_image(m10_all_building_RGB, preview_output_path + mode_10 + medium_path + f'Masks/{city_name}/all_building_mask_{city_name}_c{col:03}_r{row:03}.png')
                save_rgb_image(m10_non_residential_RGB, preview_output_path + mode_10 + medium_path + f'Masks/{city_name}/non_residential_building_mask_{city_name}_c{col:03}_r{row:03}.png')
    
                for var in range(max_variants):
                    both_file_exist_flag = True
                    NIR_name = f'SynthIR/{city_name}/synthS2_{city_name}_IRm034_c{col:03}_r{row:03}_i050_v{var:02}.png'
                    RGB_name = f'SynthRGB/{city_name}/synthS2_{city_name}_RGBm034_c{col:03}_r{row:03}_i050_v{var:02}.png'
                    if file_exists(main_input_path+RGB_name):
                        original_RGB_patch = load_png_as_rgb_array(main_input_path+RGB_name)
                        m10_RGB_patch = resize_image_with_filter(original_RGB_patch,(m10_res,m10_res))
                        save_rgb_image(m10_RGB_patch, preview_output_path + mode_10 + medium_path + f'RGB/{city_name}/RGBpatch_c{col:03}_r{row:03}_{var:02}.png')
                    else:
                        both_file_exist_flag = False
                    if file_exists(main_input_path+NIR_name):
                        original_NIR_patch = load_png_as_rgb_array(main_input_path+NIR_name)
                        m10_NIR_patch = resize_image_with_filter(original_NIR_patch,(m10_res,m10_res))
                        save_rgb_image(m10_NIR_patch, preview_output_path + mode_10 + medium_path + f'NIR/{city_name}/NIRpatch_c{col:03}_r{row:03}_{var:02}.png')
                    else:
                        both_file_exist_flag = False

                    if both_file_exist_flag:
                        # Let's recover the original spectral range 
                        original_B04_patch = scale_array(np.squeeze(original_RGB_patch[:,:,0]), band_dynamic_range['B04'][0], band_dynamic_range['B04'][1])
                        original_B03_patch = scale_array(np.squeeze(original_RGB_patch[:,:,1]), band_dynamic_range['B03'][0], band_dynamic_range['B03'][1])
                        original_B02_patch = scale_array(np.squeeze(original_RGB_patch[:,:,2]), band_dynamic_range['B02'][0], band_dynamic_range['B02'][1])
                        original_B08_patch = scale_array(np.squeeze(original_NIR_patch[:,:,1]), band_dynamic_range['B08'][0], band_dynamic_range['B08'][1])
                        # Get all re-sized channels together
                        res_factor = 3
                        new_resolution = [m10_res,m10_res]
                        resized_fullband_patch = np.zeros((new_resolution[0], new_resolution[1], 4))
                        resized_fullband_patch[:,:,0] = resize_array_with_antialiasing(original_B04_patch, new_resolution[0], new_resolution[1])
                        resized_fullband_patch[:,:,1] = resize_array_with_antialiasing(original_B03_patch, new_resolution[0], new_resolution[1])
                        resized_fullband_patch[:,:,2] = resize_array_with_antialiasing(original_B02_patch, new_resolution[0], new_resolution[1])
                        resized_fullband_patch[:,:,3] = resize_array_with_antialiasing(original_B08_patch, new_resolution[0], new_resolution[1])
                        # Build the xarray
                        xr_array = xr.DataArray(resized_fullband_patch, dims=['y', 'x', 'band'], coords={'band': ['B04', 'B03', 'B02', 'B08']})
                        # Store the netCDF file 
                        xr_array.to_netcdf(xarray_output_path + mode_10 + medium_path + f'BandPatches/{city_name}/patch_RGBandNIR_{city_name}_c{col:03}_r{row:03}.nc')
            
                # Resolution 10m xarray
                # Build the xarray for masks
                mask_bulding = np.zeros((m10_res, m10_res, 2))
                mask_bulding[:,:,0] = m10_all_building_RGB[:,:,1]/255.0
                mask_bulding[:,:,1] = m10_non_residential_RGB[:,:,1]/255.0
                xr_mask_array = xr.DataArray(mask_bulding, dims=['y', 'x', 'mask'], coords={'mask': ['all_building_mask', 'non_residential_building_mask']})
                # Store the netCDF file 
                xr_mask_array.to_netcdf(xarray_output_path + mode_10 + medium_path + f'Masks/{city_name}/mask_set_{city_name}_c{col:03}_r{row:03}.nc')









