# File Name:     detect_clouds.py
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
# Just a small script to understand better cloud presence in our patches



import numpy as np
import os
from PIL import Image
import shutil


def list_png_files(directory):
    """
    Lists all files with .npz extension in a given directory.
    
    Args:
        directory (str): The path to the directory to search for .npz files.
        
    Returns:
        list: A list of filenames with .npz extension found in the directory.
    """
    # Initialize an empty list to hold the names of .npz files
    png_filenames = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .npz extension
        if filename.endswith('.png'):
            # If it does, add it to the list
            png_filenames.append(filename)
    
    return png_filenames



def count_near_white_pixels(image_path, threshold=240):
    """
    Count the number of pixels in a PNG image that are close to white.

    Args:
        image_path (str): The path to the PNG image.
        threshold (int): The threshold value to consider a pixel close to white (default is 240).

    Returns:
        int: The number of pixels close to white.
    """
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Convert the image to a NumPy array
    np_image = np.array(image)
    
    # Define the condition for near white pixels
    near_white_condition = (np_image[:, :, 0] > threshold) & (np_image[:, :, 1] > threshold) & (np_image[:, :, 2] > threshold)
    
    # Count the number of near white pixels
    near_white_count = np.sum(near_white_condition)
    
    return near_white_count


def copy_file(source, destination):
    """
    Copies a file from the source folder to the destination folder.

    Parameters:
    source (str): Path of the file to copy.
    destination (str): Path of the destination folder.
    """
    try:
        # Check if the source file exists
        if not os.path.isfile(source):
            print(f"The source file '{source}' does not exist.")
            return

        # Check if the destination folder exists, if not, create it
        if not os.path.exists(destination):
            os.makedirs(destination)

        # Build the full path for the destination file
        destination_file = os.path.join(destination, os.path.basename(source))

        # Copy the file
        shutil.copy(source, destination_file)
        # print(f"File successfully copied to '{destination_file}'.")

    except Exception as e:
        print(f"Error copying the file: {e}")



if __name__ == "__main__":
    threshold = 1500
    patch_RGB_base_path =f'C:/DATASETs/UC2Training512/targetRGB/'
    patch_cloud_base_path =f'C:/DATASETs/UC2Training512/targetRGB2/'
    png_filenames = list_png_files(patch_RGB_base_path)
    for i, png_filename in enumerate(png_filenames):
        n_pixels = count_near_white_pixels(patch_RGB_base_path + png_filename)
        if n_pixels > threshold:
            print(f'Cloud pixels = {n_pixels}  in file {png_filename}')
            copy_file(patch_RGB_base_path + png_filename, patch_cloud_base_path)

        