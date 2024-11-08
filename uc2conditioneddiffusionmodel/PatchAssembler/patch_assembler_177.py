# File Name:     patch_assembler_177.py
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
# This is a very simple assembly script to get the fisrt version of synthetic Poitiers

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import sys

 

def get_poitiers_patch(tile_id, col, row, target_res, variation_i):
    path = 'C:/DMTA/projects/SD4EO/uc2conditioneddiffusionmodel/T30TYS/'
    filename = f'synthS2_{tile_id}_RGBm034_c{col:03}_r{row:03}_i050_v{variation_i:02}.jpg'

    rgb_patch = cv2.imread(path + filename)
    # Do not forget that OpenCV read images in BGR order.
    rgb_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_BGR2RGB)
    # Careful!!! Surce image was normalized to possitive values: [0, 1].
    # While target S2 RGB images were normalized in a different way [-1, 1].
    rgb_patch = rgb_patch.astype(np.float32)
    resized_rgb_patch = resize_image_with_filter(rgb_patch, (target_res,target_res))
    return resized_rgb_patch

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


if __name__ == "__main__":
    poitiers_tile = 'T30TYS'
    S2_resolution = 177
    ini_col = 26
    end_col = 32
    ini_row = 39+1
    end_row = 45+1
    total_cols = end_col - ini_col
    total_rows = end_row - ini_row

    for variation_i in range(8):
        output_filename = f'./synthetic_Poitiers_v{variation_i:02}.jpg'
        rgb_canvas = np.zeros((total_rows*S2_resolution, total_cols*S2_resolution,3))
        print(rgb_canvas.shape)
        col_i = 0
        for col in range(ini_col, end_col):
            row_i = total_rows - 1
            for row in range(ini_row, end_row):
                rgb_patch = get_poitiers_patch(poitiers_tile, col, row, S2_resolution, variation_i)
                # rgb_canvas[col_i*S2_resolution:(col_i+1)*S2_resolution, row_i*S2_resolution:(row_i+1)*S2_resolution,:] = rgb_patch[:,:,:]
                rgb_canvas[row_i*S2_resolution:(row_i+1)*S2_resolution, col_i*S2_resolution:(col_i+1)*S2_resolution,:] = rgb_patch[:,:,:]
                row_i = row_i - 1
            col_i = col_i + 1
        save_rgb_image(rgb_canvas, output_filename)
    


