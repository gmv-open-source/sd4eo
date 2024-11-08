# File Name:     mask_OSM_gen.py
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
# The following source code was implemented based on the guide given by ControlNet documentation in
# https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md. All code taken from that guide is lincesed
# under the terms of Apache 2.0 License 

# This is a script that takes a 512x512 Open Street Map for:
# * generating a binary mask that shows the buildings
# * trying to guess non-residential buildings
# * later scale to original resolution and floating point masks
# 
# note: perhaps it should be better to work with the original 
#       566x566 OSM patches instead...
#
# Poitiers:    col = range(26,32+1)
#              row = range(39,45+1)
#
# NOTE: Not forget that the final resolution image will scale 3x3 pixels into ONE
#       So, small details will not be noticed when we have the BIG pixels 

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.morphology import square
from scipy.ndimage import label, find_objects, binary_opening, binary_dilation
from skimage.transform import resize
import sys


# Open Street Maps color pallete
urban_building_colors = {
    "Significant building" : (196,182,171),
    "Generic building" : (217,208,201),
    "Military building" : (220,198,192),
    "Church": (185,170,157),
    "Church border": (179,159,136),
    }

urban_building_border_colors = {
    "Border generic building" : (201,188,178),
    }

urban_surrounding_colors = {  #
    "Residential area": (224,223,223),
    "Retail area": (255,214,209),
    "Commercial area": (242,218,217),
    "Industrial area": (235,219,232),
    "Farm": (245,220,186),
    "Brownfield site": (199,199,180),
    "Cemetery": (170,203,175),
    "Allotments": (201,225,191),
    "Sports pitch": (136,224,190),
    "Sports centre": (223,252,226),
    "Airport apron": (218,218,224),
    "Airport runaway": (187,187,204),
    "Airport area": (233,231,226),
    "Military area (solid)": (255,242,242),
    "Military area (stripes)": (255,200,200),
    "School · University · Hospital": (255,255,229),
    "Railway station": (121,129,176),
    }

urban_area_colors = {
    "Park": (200,250,204),
    "Parking": (238,238,238),
    }

openland_surrounding_colors = {
    "Forest": (173,209,158),
    "Orchard": (174,223,163),
    "Grass": (205,235,163),
    "Farmland": (238,240,213),
    "Heathland": (214,217,159),
    "Scrubland": (200,215,171),
    "Bare rock": (238,229,220),
    "Sand": (245,233,198),
    "Golf course": (222,246,192),
    }


def visualize_array(numpy_array, title=''):
    """Visualize one RGB arrays 
    
    Args:
        array_rgb (np.ndarray): A 3D numpy array representing an RGB image (height, width, 3).
        title (str): optional title for the figure
    """
    plt.figure()
    plt.imshow(numpy_array/255.0)
    plt.title(title)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def visualize_boolean_array(array_2d):
    """Visualizes a 2D boolean array using a heatmap.
    
    Args:
        array_2d (np.ndarray): A 2D numpy array of boolean values.
        
    Raises:
        ValueError: If the input array is not 2D or not boolean.
    """
    if not isinstance(array_2d, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    if array_2d.ndim != 2:
        raise ValueError("Input array must be 2D.")
    
    if array_2d.dtype != bool:
        raise ValueError("Input array must contain boolean values.")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(array_2d, cmap='Greys', interpolation='none')
    plt.colorbar(label='Boolean Value (True/False)')
    plt.title('2D Boolean Array Visualization')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()



def visualize_2_arrays_v2(array_rgb, array_2d):
    """Visualize two arrays side by side: a 3D RGB array and a 2D float array.
    
    Args:
        array_rgb (np.ndarray): A 3D numpy array representing an RGB image (height, width, 3).
        array_2d (np.ndarray): A 2D numpy array of float values.
    """
    # Create a figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the RGB image
    axes[0].imshow(array_rgb/255.0)
    axes[0].axis('off')  # Hide axis
    axes[0].set_title('RGB Image')
    
    # Display the 2D array with a grayscale colormap and color bar
    cax = axes[1].imshow(array_2d, cmap='gray')
    fig.colorbar(cax, ax=axes[1])
    axes[1].axis('off')  # Hide axis
    axes[1].set_title('2D Grayscale Image')
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def mark_building_pixels(numpy_imageRGB, tolerance=0.02, all_buildings_flag=True):
    """
    Identifies building pixels in a 3D RGB image array from Open Street Maps.
    
    Args:
        image (np.ndarray): 3D numpy array representing the RGB image.
        tolerance (float): Maximum allowed variation in color values (default is 2%).
        all_buildings_flag (bool): It considers all buildings without exception, otherwise we only consider the significant buildings (like cathedral, churches, hospitals, military buildings...)

    Returns:
        np.ndarray: 2D numpy array with the same number of rows and columns as the input image,
                    where 1 indicates a building pixel and 0 indicates a non-building pixel.
    """
    # Define the RGB values for the known building types
    building_colors = []
    if all_buildings_flag:
        dict_to_explore = {**urban_building_border_colors, **urban_building_colors}
        for key in dict_to_explore:
            building_colors.append(np.array(dict_to_explore[key]))
    else: # only significant buildings
        for key in urban_building_colors:
            if key != 'Generic building':
                building_colors.append(np.array(urban_building_colors[key]))
    # Calculate the tolerance in absolute RGB values
    tolerance_value = tolerance * 255.0

    # Initialize the output 2D array
    raw_building_mask = np.zeros((numpy_imageRGB.shape[0], numpy_imageRGB.shape[1]), dtype=bool)
    raw_area_mask = np.zeros((numpy_imageRGB.shape[0], numpy_imageRGB.shape[1]), dtype=bool)

    for color in building_colors:
        # Create a boolean mask for pixels within the tolerance range of the current building color
        mask = np.all(np.abs(numpy_imageRGB - color) <= tolerance_value, axis=-1)
        # Update the building mask
        raw_building_mask = np.logical_or(raw_building_mask, mask)
    
    dict_area_colors = {**urban_surrounding_colors, **urban_area_colors, **openland_surrounding_colors}
    area_colors = []
    for key in dict_area_colors:
        area_colors.append(np.array(dict_area_colors[key]))
    for color in area_colors:
        mask = np.all(np.abs(numpy_imageRGB - color) <= tolerance_value/2.0, axis=-1)
        raw_area_mask = np.logical_or(raw_area_mask, mask)
    # Now we compose both boolean masks, in order to get a cleaner map with pixels where we detect 
    # buildings and they have colors which are different enough from surrounding areas
    final_mask = np.logical_and(raw_building_mask, np.logical_not(raw_area_mask))

    return final_mask


def mark_nonresidential_surrounding_areas(numpy_imageRGB, tolerance=0.02):
    """
    Identifies non residential surrounding areas in a 3D RGB image array from Open Street Maps.
    
    Args:
        image (np.ndarray): 3D numpy array representing the RGB image.
        tolerance (float): Maximum allowed variation in color values (default is 2%).

    Returns:
        np.ndarray: 2D numpy array with the same number of rows and columns as the input image,
                    where 1 indicates a non residential surrounding area and 0 any other stuff.
    """
    # Define the RGB values for the interest surrounding areas


    surrounding_colors = []
    for key in urban_surrounding_colors:
        if key != "Residential area":
            surrounding_colors.append(np.array(urban_surrounding_colors[key]))

    # Calculate the tolerance in absolute RGB values
    tolerance_value = tolerance * 255.0

    # Initialize the output 2D array
    raw_surrounding_mask = np.zeros((numpy_imageRGB.shape[0], numpy_imageRGB.shape[1]), dtype=bool)
    # raw_area_mask = np.zeros((numpy_imageRGB.shape[0], numpy_imageRGB.shape[1]), dtype=bool)

    for color in surrounding_colors:
        # Create a boolean mask for pixels within the tolerance range of the current building color
        mask = np.all(np.abs(numpy_imageRGB - color) <= tolerance_value, axis=-1)
        # Update the building mask
        raw_surrounding_mask = np.logical_or(raw_surrounding_mask, mask)
    
    return raw_surrounding_mask






def visualize_2_arrays(array_rgb, array_2d):
    """Visualize two arrays side by side: a 3D RGB array and a 2D float array.
    
    Args:
        array_rgb (np.ndarray): A 3D numpy array representing an RGB image (height, width, 3).
        array_2d (np.ndarray): A 2D numpy array of float values.
    """
    # Determine the shape of the images
    height_rgb, width_rgb, _ = array_rgb.shape
    height_2d, width_2d = array_2d.shape
    
    # Create a figure with a specific size to maintain relative sizes
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the RGB image
    axes[0].imshow(array_rgb/255.0)
    axes[0].axis('off')  # Hide axis
    axes[0].set_title('RGB Image')
    
    # Display the 2D array with a grayscale colormap and color bar
    cax = axes[1].imshow(array_2d, cmap='gray')
    # fig.colorbar(cax, ax=axes[1])
    axes[1].axis('off')  # Hide axis
    axes[1].set_title('2D Grayscale Image')
    
    # Adjust aspect ratio to match the images' sizes
    axes[0].set_aspect(height_rgb / width_rgb)
    axes[1].set_aspect(height_2d / width_2d)
    
    # Show the plot
    plt.tight_layout()
    plt.show()





def get_osm_image(city_id, col, row):
    path = f'C:/DATASETs/UC2Synthesis/OSM/{city_id}/'
    filename = f'patch_{city_id}_{col:03}_{row:03}_OSM.png'
    full_filename = path + filename
    # print(full_filename)

    condition_map = cv2.imread(full_filename)
    # Do not forget that OpenCV read images in BGR order.
    condition_map = cv2.cvtColor(condition_map, cv2.COLOR_BGR2RGB)
    # Careful!!! Surce image was normalized to possitive values: [0, 1].
    # While target S2 RGB images were normalized in a different way [-1, 1].
    condition_map = condition_map.astype(np.float32)
    return condition_map


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


def resize_boolean_image_with_filter(image: np.ndarray, new_size: tuple) -> np.ndarray:
    """
    Resizes a 2D numpy array representing a boolean image to a new specified size after applying a Gaussian filter.
    
    Args:
        image (np.ndarray): 2D numpy array of shape (height, width) representing the input boolean image.
        new_size (tuple): Tuple (new_height, new_width) specifying the desired output image size.
        
    Returns:
        np.ndarray: 2D numpy array of shape (new_height, new_width) representing the resized image as floats.
    """
    # Ensure the input is a boolean array
    if image.dtype != bool:
        raise ValueError("Input image must be a boolean array")   
    # Convert boolean to float for filtering
    image_float = image.astype(float)*255
    image_RGB = np.zeros((image_float.shape[0],image_float.shape[1],3))
    for i in range(3):
        image_RGB[:,:,i] = image_float
    print(image_RGB.shape)
    return resize_image_with_filter(image_RGB, new_size)




def remove_small_and_narrow_regions(array, size_threshold, narrow_threshold):
    """
    Removes small and narrow active regions (value 1) from a boolean 2D numpy array based on size and narrowness thresholds.
    
    Args:
        array (np.ndarray): 2D boolean numpy array.
        size_threshold (int): Minimum number of pixels for a region to be retained.
        narrow_threshold (int): Minimum width of the structuring element to remove narrow regions.
    
    Returns:
        np.ndarray: 2D boolean numpy array with small and narrow regions removed.
    """
    # Label connected regions in the array
    labeled_array, num_features = label(array)
    
    # Create an output array initialized to False
    output_array = np.zeros_like(array, dtype=bool)
    
    # Iterate through each region found
    for region_idx in range(1, num_features + 1):
        # Find the coordinates of the current region
        region_slice = find_objects(labeled_array == region_idx)[0]
        
        # Get the region from the labeled array
        region = labeled_array[region_slice] == region_idx
        
        # Check if the region size is above the threshold
        if np.sum(region) > size_threshold:
            # Apply binary opening to remove narrow regions
            opened_region = binary_opening(region, structure=square(narrow_threshold))
            
            # If the opened region still has enough pixels, retain it
            if np.sum(opened_region) > size_threshold:
                output_array[region_slice][region] = True  # Keep the original region if it is large enough

    return output_array



def remove_small_regions(array, threshold):
    """
    Removes small active regions (value 1) from a boolean 2D numpy array based on a size threshold.
    
    Args:
        array (np.ndarray): 2D boolean numpy array.
        threshold (int): Minimum number of pixels for a region to be retained.
    
    Returns:
        np.ndarray: 2D boolean numpy array with small regions removed.
    """
    # Label connected regions in the array
    labeled_array, num_features = label(array)
    
    # Create an output array initialized to False
    output_array = np.zeros_like(array, dtype=bool)
    
    # Iterate through each region found
    for region_idx in range(1, num_features + 1):
        # Find the coordinates of the current region
        region_slice = find_objects(labeled_array == region_idx)[0]
        
        # Get the region from the labeled array
        region = labeled_array[region_slice] == region_idx
        
        # Check if the region size is above the threshold
        if np.sum(region) > threshold:
            # Set the corresponding area in the output array to True
            output_array[region_slice][region] = True
    
    return output_array


def dilate_boolean_array(input_array, size=1):
    """Dilates a 2D boolean array by a specified number of pixels.

    Args:
        input_array (np.ndarray): 2D numpy array to be dilated. Will be converted to boolean if not already.
        size (int, optional): The size of the dilation. Defaults to 1 pixel.

    Returns:
        np.ndarray: A new 2D boolean numpy array that has been dilated.
    """
    if not isinstance(input_array, np.ndarray):
        raise ValueError("Input must be a 2D numpy array.")
    if input_array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    
    # Convert the input array to boolean type if it is not already
    boolean_array = input_array.astype(bool)
    
    # Define the structuring element for dilation
    structuring_element = np.ones((2 * size + 1, 2 * size + 1), dtype=bool)
    
    # Perform binary dilation
    dilated_array = binary_dilation(boolean_array, structure=structuring_element)
    
    return dilated_array


def highlight_nearby_buildings(building_map, interest_regions, proximity):
    """
    Highlights entire buildings that are in contact or near the regions of interest.

    Args:
        building_map (np.ndarray): 2D boolean array representing the building map (with noise).
        interest_regions (np.ndarray): 2D boolean array representing regions of interest.
        proximity (int): Number of pixels to consider for proximity.

    Returns:
        np.ndarray: 2D array with values 1 where entire buildings are near regions of interest.
    """
    # Ensure the input arrays are boolean
    building_map = building_map.astype(bool)
    interest_regions = interest_regions.astype(bool)

    # Dilate the regions of interest to include the specified proximity
    structuring_element = np.ones((2 * proximity + 1, 2 * proximity + 1), dtype=bool)
    dilated_interest_regions = binary_dilation(interest_regions, structure=structuring_element)

    # Label connected components in the building map
    labeled_buildings, num_features = label(building_map)

    # Initialize an array to store the highlighted buildings
    highlighted_buildings = np.zeros_like(building_map, dtype=int)

    # Iterate over each labeled building
    for building_label in range(1, num_features + 1):
        # Create a mask for the current building
        building_mask = (labeled_buildings == building_label)
        
        # Check if this building is near the dilated interest regions
        if np.any(building_mask & dilated_interest_regions):
            # If it is, include the entire building in the output
            highlighted_buildings[building_mask] = 1

    return highlighted_buildings

def check_extent_file(city_id, base_path):
    filename = f'np_extent_{city_id}_EPSG_3857.npz'
    extent_file = np.load(base_path + filename)
    extent_reg = extent_file['extent_reg']
    return extent_reg.shape[0], extent_reg.shape[1]

available_cities_IDs = ['Madrid', 'Paris', 'Toulouse', 'Bordeaux', 'Limoges', 'Clermont-Ferrand', 'Troyes', 'Le Mans', 'Angers', 'Poitiers', 'Niort',] # 'Rouen']

if __name__ == "__main__":
    patch_S2_size = 177
    size_threshold_noise = 8
    narrow_threshold_noise = 3
    proximity_px = 3
    size_threshold_s_buildings = 3
    narrow_threshold_s_buildings = 2
    extent_npz_folder = 'C:/DATASETs/UC2Synthesis/OSM/'

    # city_id = 'Poitiers'
    for city_id in available_cities_IDs:
        max_col, max_row = check_extent_file(city_id, extent_npz_folder)  
        for col in range(max_col):
            for row in range(max_row):
                # city_id = 'Le Mans'
                # col = 3
                # row = 4
                print(f'{city_id} :   col {col}  row {row}')
                numpy_img_RGB = get_osm_image(city_id, col, row)
                # First, the mask for all buildings
                all_building_binary_mask = mark_building_pixels(numpy_img_RGB, tolerance=0.02)
                # Second, the complex mask for non residential buildings
                surrounding_areas_bmask = mark_nonresidential_surrounding_areas(numpy_img_RGB, tolerance=0.02)
                filtered_surrounding_areas = remove_small_and_narrow_regions(surrounding_areas_bmask, size_threshold_noise, narrow_threshold_noise)
                nonresidential_surrounding_map = highlight_nearby_buildings(all_building_binary_mask, filtered_surrounding_areas, proximity_px)
                significant_building_binary_mask = mark_building_pixels(numpy_img_RGB, tolerance=0.01, all_buildings_flag=False)
                filtered_s_building_binary_mask = remove_small_and_narrow_regions(significant_building_binary_mask, size_threshold_s_buildings, narrow_threshold_s_buildings)
                increased_s_building_binary_mask = dilate_boolean_array(filtered_s_building_binary_mask, size=1)
                increased_nonresidential_surrounding_map = dilate_boolean_array(nonresidential_surrounding_map, size=1)
                non_residential_combined_mask = np.logical_or(increased_s_building_binary_mask, increased_nonresidential_surrounding_map)
                all_building_binary_mask = np.logical_or(all_building_binary_mask, non_residential_combined_mask)

                # visualize_2_arrays(numpy_img_RGB, non_residential_combined_mask*255.0)
                # visualize_2_arrays(numpy_img_RGB, all_building_binary_mask)
                # resized_mask_all_buildings = resize_boolean_image_with_filter(all_building_binary_mask, (patch_S2_size,patch_S2_size))
                # resized_mask_nonresidential_buildings = resize_boolean_image_with_filter(all_building_binary_mask, (patch_S2_size,patch_S2_size))
                
                # save_rgb_image(resized_mask_RGB, f"./test_mask.png")
                path = f'C:/DATASETs/UC2Synthesis/Mask/{city_id}/'
                npz_filename = f'mask_{city_id}_c{col:03}_r{row:03}.npz'
                np.savez(path + npz_filename, all_building_binary_mask=all_building_binary_mask, non_residential_combined_mask=non_residential_combined_mask)

           




