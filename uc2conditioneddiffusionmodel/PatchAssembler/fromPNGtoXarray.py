# File Name:     fromPNGtoXarray.py
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

# This script performs the final step in the pipeline after assembly.
# Once all assembled images with artifacts have been removed, they are taken 
# in pairs (RGB+NIR) in order, combined into a single 3D numpy array with 
# 4 channels, and converted into an xarray with the corresponding labels.
# The dynamic range is adjusted, and the scale in the number of pixels 
# is reduced to match the Sentinel-2 resolution of 10m/pixel.
# The output consists of netCDF files containing the xarrays and renamed 
# PNG files ready for publication.


import numpy as np
import os
from PIL import Image
import scipy.ndimage
import shutil
import xarray as xr


available_cities_IDs = ['Madrid', 'Paris', 'Toulouse', 'Poitiers', 'Bordeaux', 'Limoges', 'Clermont-Ferrand', 'Troyes', 'Le Mans', 'Angers', 'Niort',] # 'Rouen']
# available_cities_IDs = ['Angers', 'Niort',] # 'Rouen']


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
        rgb_array = np.array(img)

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


def copy_and_rename_file(source_path, destination_dir, new_name):
    # Copy the file to the destination directory
    destination_path = os.path.join(destination_dir, new_name)
    shutil.copy2(source_path, destination_path)


if __name__ == "__main__":
    input_PNG_folder = 'C:/DATASETs/UC2Synthesis/assembled.cloud700/'
    output_folder = 'C:/DATASETs/UC2Synthesis/assembled.Zenodov2/'
    max_scanneable_file_variation = 30
    # band dynamic range was estimated with band_histogram.py
    band_dynamic_range = {'B08':(2560.0, 12580.0), 'B02': (400.0, 4440.0), 'B03': (740.0, 5200.0), 'B04': (460.0, 6300.0)}


    for selected_city in available_cities_IDs:
        counter_RGB = 0
        counter_NIR = 0
        counter_final = 0
        no_more_files_flag = False
        while no_more_files_flag == False:
            # Update filenames
            input_RGB_filename = f'assembled_{selected_city}_RGB_v{counter_RGB}.png'
            input_NIR_filename = f'assembled_{selected_city}_IR_v{counter_NIR}.png'
            while file_exists(input_PNG_folder + input_RGB_filename) == False:
                counter_RGB = counter_RGB + 1
                input_RGB_filename = f'assembled_{selected_city}_RGB_v{counter_RGB}.png'
                if counter_RGB > max_scanneable_file_variation:
                    no_more_files_flag = True
                    break
            while file_exists(input_PNG_folder + input_NIR_filename) == False:
                counter_NIR = counter_NIR + 1
                input_NIR_filename = f'assembled_{selected_city}_IR_v{counter_NIR}.png'        
                if counter_NIR > max_scanneable_file_variation:
                    no_more_files_flag = True
                    break
            if no_more_files_flag:
                # We should change to another city
                break

            # Get corresponding numpy arrays
            original_RGB_city = load_png_as_rgb_array(input_PNG_folder + input_RGB_filename)
            original_NIR_city = load_png_as_rgb_array(input_PNG_folder + input_NIR_filename)
            # Let's recover the original spectral range 
            original_B04_city = scale_array(np.squeeze(original_RGB_city[:,:,0]), band_dynamic_range['B04'][0], band_dynamic_range['B04'][1])
            original_B03_city = scale_array(np.squeeze(original_RGB_city[:,:,1]), band_dynamic_range['B03'][0], band_dynamic_range['B03'][1])
            original_B02_city = scale_array(np.squeeze(original_RGB_city[:,:,2]), band_dynamic_range['B02'][0], band_dynamic_range['B02'][1])
            original_B08_city = scale_array(np.squeeze(original_NIR_city[:,:,1]), band_dynamic_range['B08'][0], band_dynamic_range['B08'][1])
            # Get all re-sized channels together
            res_factor = 3
            new_resolution = [int(original_RGB_city.shape[0]/3), int(original_RGB_city.shape[1]/3)]
            resized_fullband_city = np.zeros((new_resolution[0], new_resolution[1], 4))
            resized_fullband_city[:,:,0] = resize_array_with_antialiasing(original_B04_city, new_resolution[0], new_resolution[1])
            resized_fullband_city[:,:,1] = resize_array_with_antialiasing(original_B03_city, new_resolution[0], new_resolution[1])
            resized_fullband_city[:,:,2] = resize_array_with_antialiasing(original_B02_city, new_resolution[0], new_resolution[1])
            resized_fullband_city[:,:,3] = resize_array_with_antialiasing(original_B08_city, new_resolution[0], new_resolution[1])
            # Build the xarray
            xr_array = xr.DataArray(resized_fullband_city, dims=['y', 'x', 'band'], coords={'band': ['B04', 'B03', 'B02', 'B08']})
            # Store the netCDF file 
            xr_array.to_netcdf(output_folder + f'assembled_{selected_city}_v{counter_final:02d}.nc')
            # Copy RGB and NIR with new number in new location 
            copy_and_rename_file(input_PNG_folder + input_RGB_filename, output_folder, f'assembled_{selected_city}_RGB_v{counter_final:02d}.png')
            copy_and_rename_file(input_PNG_folder + input_NIR_filename, output_folder, f'assembled_{selected_city}_NIR_v{counter_final:02d}.png')
            print(f'variante # {counter_final} completada : {counter_RGB} + {counter_NIR} ')

            # Final scaling --> we performed it externally
            counter_RGB = counter_RGB + 1
            counter_NIR = counter_NIR + 1
            counter_final = counter_final + 1
        no_more_files_flag = False # reset while condition




























