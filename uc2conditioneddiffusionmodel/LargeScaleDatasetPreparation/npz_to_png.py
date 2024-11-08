# File Name:     npz_to_png.py
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
# This script has the functionality to transform NPZ files into common PNG
# Original dynamic range are estimated 


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.transform import resize

from common_tile_utils import available_Tiles_IDs

# /mnt/SD4EO/02_UC2-Settlements/02_SyntheticData/01_GMV-SES/UC2Training512/source/
# /mnt/SD4EO/02_UC2-Settlements/02_SyntheticData/01_GMV-SES/UC2Training512/targetRGB/
# /mnt/SD4EO/02_UC2-Settlements/02_SyntheticData/01_GMV-SES/UC2Training512/targetIR/

def resample_and_save_image(image: np.ndarray, output_path: str, size: int = 512, compression_level: int = 1):
    """
    >>> DEPRECATED!!
    Resamples a 4D numpy array (image in BGR+Alpha format) to a square image of specified size
    and saves it in PNG format with specified compression level.

    Args:
        image (np.ndarray): Input image array with shape (rows, columns, channels), 
                            where channels are in BGR+Alpha format.
        output_path (str): Path where the output image will be saved.
        size (int): Desired size for the output square image. Default is 512.
        compression_level (int): Compression level for PNG format. Must be between 0 and 9.
                                 Default is 1.
    """
    # Ensure compression level is within the valid range
    if not (0 <= compression_level <= 9):
        raise ValueError("Compression level must be between 0 and 9.")
    
    # Check if the image has 4 channels
    if image.shape[2] != 4:
        raise ValueError("Input image must have 4 channels (BGR + IR).")

    # Separate BGR and Alpha channels
    bgr = image[:, :, :3]
    infrared = image[:, :, 3]

    # Convert BGR image to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Resize the RGB and Alpha images to the desired square size using high-quality interpolation
    resized_rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    resized_infrared = cv2.resize(infrared, (size, size), interpolation=cv2.INTER_CUBIC)

    # # Merge the resized RGB and Alpha channels
    # resized_rgba = np.dstack((resized_rgb, resized_infrared))

    # Save the image as PNG with the specified compression level
    cv2.imwrite(output_path, resized_rgb, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])



def upscale_and_adjust_dynamic_range(array, min_value, max_value, squared_pixel_size=512):
    """
    Upscales a 2D NumPy array to a new 512x512 2D array with dynamic range adjustment.

    Args:
        array (np.array): The input 2D NumPy array to be upscaled.
        min_value (float): The minimum value for dynamic range adjustment.
        max_value (float): The maximum value for dynamic range adjustment.
        squared_pixel_size (int): Usuually 512 to get a 512x512 upscaled array

    Returns:
        The new upscaled 512x512 2D array.

    Note: 
        We DO NOT perform antialiasing, because we are UPscaling the original array
    """

    if array.ndim != 2:
        raise ValueError("Input array must be 2D")
    # Upscale the array using skimage's resize function
    upscaled_array = resize(array, (squared_pixel_size, squared_pixel_size), mode='symmetric')
    # Adjust dynamic range to specified values
    adjusted_array = np.clip(upscaled_array, min_value, max_value)
    # Normalize the adjusted array to a range of [0, 255]
    normalized_array = (adjusted_array - min_value) / (max_value - min_value) * 255
    return normalized_array

    # # Convert the normalized array to an appropriate image data type
    # image_array = normalized_array.astype(np.uint8)

    # # Save the PNG image with adjusted dynamic range
    # imwrite(output_path, image_array)


def create_png(blue_array, green_array, red_array, filename):
    """
    Creates a PNG image from 2D NumPy arrays representing RGB channels.

    Args:
        blue_array: 2D NumPy array containing blue channel values (between 0 and 255).
        green_array: 2D NumPy array containing green channel values (between 0 and 255).
        red_array: 2D NumPy array containing red channel values (between 0 and 255).

        filename: Name of the PNG file to be created.

    Raises:
        ValueError: If the arrays do not have the same size or if the values are not between 0 and 255.

    note: OpenCV uses BGR natively
    """

    if not (blue_array.shape == green_array.shape == red_array.shape):
        raise ValueError("Arrays must have the same shape.")

    # if not (np.min(red_array) >= 0 and np.max(red_array) <= 255 and
    #         np.min(green_array) >= 0 and np.max(green_array) <= 255 and
    #         np.min(blue_array) >= 0 and np.max(blue_array) <= 255):
    #     raise ValueError("Array values must be between 0 and 255.")

    # Combine arrays into an RGB image
    image = np.dstack([blue_array, green_array, red_array])
    # Convert image from uint8 to BGR format for OpenCV to read
    image = image.astype(np.uint8)
    # Save the image as a PNG
    cv2.imwrite(filename, image)


def list_npz_files(directory):
    """
    Lists all files with .npz extension in a given directory.
    
    Args:
        directory (str): The path to the directory to search for .npz files.
        
    Returns:
        list: A list of filenames with .npz extension found in the directory.
    """
    # Initialize an empty list to hold the names of .npz files
    npz_filenames = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .npz extension
        if filename.endswith('.npz'):
            # If it does, add it to the list
            npz_filenames.append(filename)
    
    return npz_filenames


if __name__ == "__main__":
    # band dynamic range was estimated with band_histogram.py
    band_dynamic_range = {'B08':(2560.0, 12580.0), 'B02': (400.0, 4440.0), 'B03': (740.0, 5200.0), 'B04': (460.0, 6300.0)}
    target_size = 512

    patch_output_RGB_base_path =f'C:/DATASETs/UC2Training512/targetRGB/'
    patch_output_IR_base_path =f'C:/DATASETs/UC2Training512/targetIR/'
    for tile_id in available_Tiles_IDs:
        patch_input_base_path =f'C:/DATASETs/UC2Training/npS2Patches/{tile_id}/' 
        npz_filenames = list_npz_files(patch_input_base_path)
        num_files = len(npz_filenames)

        for i, np_filename in enumerate(npz_filenames):
        # np_filename = patch_input_base_path + f'patch_{tile_id}_{col_i:03}_{row_i:03}_S2.npz'
            with np.load(patch_input_base_path + np_filename) as np_file:
                array3D = np_file['array3D']
            if i%50 == 0:
                print(f'{tile_id} : {i} / {num_files}')

            patch_B02 = upscale_and_adjust_dynamic_range(np.squeeze(array3D[:,:,0]), band_dynamic_range['B02'][0], band_dynamic_range['B02'][1], target_size)
            patch_B03 = upscale_and_adjust_dynamic_range(np.squeeze(array3D[:,:,1]), band_dynamic_range['B03'][0], band_dynamic_range['B03'][1], target_size)
            patch_B04 = upscale_and_adjust_dynamic_range(np.squeeze(array3D[:,:,2]), band_dynamic_range['B04'][0], band_dynamic_range['B04'][1], target_size)
            patch_B08 = upscale_and_adjust_dynamic_range(np.squeeze(array3D[:,:,3]), band_dynamic_range['B08'][0], band_dynamic_range['B08'][1], target_size)
            
            rgb_filename = patch_output_RGB_base_path + np_filename[:-3] + 'png'
            create_png(patch_B02, patch_B03, patch_B04, rgb_filename)
            ir_filename = patch_output_IR_base_path + np_filename[:-3] + 'png'
            create_png(patch_B08, patch_B08, patch_B08, ir_filename)

    pass



