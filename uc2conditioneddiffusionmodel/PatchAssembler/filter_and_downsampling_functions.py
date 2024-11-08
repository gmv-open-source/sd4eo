# File Name:     filter_and_downsampling_functions.py
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

import scipy.ndimage
from scipy.ndimage import gaussian_filter  # Not used
from skimage.transform import resize

# This funtion has been used for downsampling the band patches
# Explicit gaussian filter was discarded, instead we use the anti_aliasing flag
# As it can be checked here: 
#   https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
# it implicitly uses the gaussian filter size according to the re-scale factor
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



# This funtion has been used for downsampling the building covering masks
# So they were transformed from "binary" to floating point values
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