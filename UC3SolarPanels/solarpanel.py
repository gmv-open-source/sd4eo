# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     solarpanel.py
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

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_dilation

def rectify_solar_panel(rgb, mask):
    """Rectify the regions in the RGB image corresponding to white regions in the mask image.

    Args:
        img_rgb (str): Filename of the RGB image.
        img_mask (str): Filename of the mask image with white rectangles.

    Returns:
        numpy.ndarray: A new image with rectified solar panel regions aligned to the axes.
    """
    # Load the images
    # rgb = cv2.imread(img_rgb)
    # mask = cv2.imread(img_mask, cv2.IMREAD_GRAYSCALE)

    # Create an empty image with an alpha channel (RGBA) to store the rectified panels
    rectified_panels = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)

    # Find contours in the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            # Obtain the bounding box of the approximated polygon
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Get width and height of the bounding box
            width = int(rect[1][0])
            height = int(rect[1][1])

            # Define the destination points for the perspective transform
            dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
            src_pts = np.array(box, dtype="float32")

            # Compute the perspective transform matrix and apply it
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(rgb, M, (width, height))

            # Define the region in the rectified_panels image where the warped image will be placed
            x, y, w, h = cv2.boundingRect(contour)
            rectified_region = cv2.resize(warped, (w, h))

            # Place the rectified panel into the new image with transparency
            alpha_channel = np.ones((h, w), dtype=np.uint8) * 255  # Create an alpha channel (fully opaque)
            rectified_region = cv2.merge((rectified_region, alpha_channel))
            rectified_panels[y:y+h, x:x+w] = rectified_region

    return rectified_panels


def check_geometric_roof_criteria(region, min_size: int, min_solidity: float, max_perimeter_ratio: float):
    final_result = True
    if region.area < min_size:
        final_result = False
    if region.solidity < min_solidity:
        final_result = False
    perimeter_ratio = region.perimeter / np.sqrt(region.area)
    if perimeter_ratio > max_perimeter_ratio:
        final_result = False
    return final_result

def detect_white_regions(image: np.ndarray, tolerance: int, min_size: int, min_solidity: float, max_perimeter_ratio: float) -> (np.ndarray, list):
    """
    Detect regions in the image that are close to white within the given tolerance and larger than the specified size.
    
    Args:
        image (np.ndarray): Input RGB image.
        tolerance (int): Tolerance for the color difference from white (255, 255, 255).
        min_size (int): Minimum size in pixels for regions to be considered.
        min_solidity (float): Minimum ratio between the number of pixels in the region and the number of pixels in the convex hull
        max_perimeter_ratio (float): Maximum ratio between actual 4-connected perimeter and the square root of the number of pixels
    
    Returns:
        np.ndarray: Boolean array where pixels close to white are marked with 1, others with 0.
        list: List of oriented bounding boxes for each detected region larger than the specified size.
    """
    # Define the white color and tolerance bounds
    white_color = np.array([255, 255, 255])
    lower_bound = np.clip(white_color - tolerance, 0, 255)
    upper_bound = np.clip(white_color + tolerance, 0, 255)
    
    # Create a mask where white regions are within the tolerance
    mask = cv2.inRange(image, lower_bound, upper_bound)
    
    # Convert mask to boolean array
    boolean_mask = mask.astype(bool)
    boolean_mask = fill_holes(boolean_mask)

    # Label connected regions
    labeled_image, num_labels = label(boolean_mask, return_num=True)
    
    # Find properties of labeled regions
    regions = regionprops(labeled_image)
    
    # Extract oriented and axis-aligned bounding boxes for regions larger than min_size
    oriented_bounding_boxes = []
    axis_aligned_bounding_boxes = []
    for region in regions:
        if check_geometric_roof_criteria(region, min_size, min_solidity, max_perimeter_ratio):
            minr, minc, maxr, maxc = region.bbox
            # Get the coordinates of the region
            region_coords = np.column_stack((region.coords[:, 0], region.coords[:, 1]))
            # Compute the oriented bounding box
            rect = cv2.minAreaRect(region_coords.astype(np.float32))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            oriented_bounding_boxes.append(box)
            
            # Compute the axis-aligned bounding box in YOLOv8 format
            x_center = (minc + maxc) / 2
            y_center = (minr + maxr) / 2
            width = maxc - minc
            height = maxr - minr
            img_height, img_width = image.shape[:2]
            yolo_bbox = [x_center / img_width, y_center / img_height, width / img_width, height / img_height]
            axis_aligned_bounding_boxes.append(yolo_bbox)
    
    # Create a boolean mask with only regions larger than min_size
    filtered_mask = np.zeros_like(boolean_mask)
    for region in regions:
        if check_geometric_roof_criteria(region, min_size, min_solidity, max_perimeter_ratio):
            filtered_mask[labeled_image == region.label] = True
    
    return filtered_mask, oriented_bounding_boxes, axis_aligned_bounding_boxes


def write_yolo_bounding_boxes(filename: str, bboxes: list) -> None:
    """
    Write the axis-aligned bounding boxes to a file in YOLOv8 format.

    Args:
        filename (str): The name of the output file.
        bboxes (list): List of axis-aligned bounding boxes in YOLOv8 format.
                       Each bounding box should be a list or tuple of the form [x_center, y_center, width, height].

    Returns:
        None
    """
    # Check if the list of bounding boxes is not empty
    if bboxes:
        # Open the file for writing
        with open(filename, 'w') as file:
            # Iterate over each bounding box and write to the file
            for bbox in bboxes:
                # YOLOv8 format: <class> <x_center> <y_center> <width> <height>
                # Assuming class is 0 for all boxes
                line = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                file.write(line)


def get_image(full_path_filename):
    rgb_image = cv2.imread(full_path_filename)
    # Do not forget that OpenCV read images in BGR order.
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    # Careful!!! Surce image was normalized to possitive values: [0, 1].
    # While target S2 RGB images were normalized in a different way [-1, 1].
    # rgb_image = rgb_image.astype(np.float32)
    return rgb_image


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



def fill_holes(input_array: np.ndarray) -> np.ndarray:
    """Fill holes inside the regions of 1s in a 2D boolean numpy array.

    Args:
        input_array (np.ndarray): A 2D boolean numpy array.

    Returns:
        np.ndarray: A 2D boolean numpy array with holes filled.
    """
    # Ensure the input array is boolean
    if input_array.dtype != bool:
        raise ValueError("Input array must be of boolean type")

    # Fill holes inside the regions of 1s
    filled_array = binary_fill_holes(input_array)

    return filled_array


def visualize_with_bounding_boxes(boolean_array: np.ndarray, bounding_boxes: list) -> None:
    """
    Visualize a 2D boolean array and draw oriented bounding boxes with red lines.
    
    Args:
        boolean_array (np.ndarray): Input 2D boolean array.
        bounding_boxes (list): List of oriented bounding boxes.
    """
    # Convert the boolean array to a 3-channel (RGB) image for visualization
    visualization_image = np.stack([boolean_array] * 3, axis=-1).astype(np.uint8) * 255
    
    # Draw each bounding box on the visualization image
    for box in bounding_boxes:
        box = np.array(box, dtype=np.int32)
        # We swap rows & columns
        new_box = np.zeros(box.shape)
        new_box[:,0] = box[:,1]
        new_box[:,1] = box[:,0]
        new_box = np.array(new_box, dtype=np.int32)
        try:
            cv2.polylines(visualization_image, [new_box], isClosed=True, color=(255, 0, 0), thickness=2)
        except:
            pass
    
    # Display the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(visualization_image)
    plt.axis('off')
    plt.show()


def boolean_array_to_rgb_array(boolean_array):
    """
    Converts a 2D boolean NumPy array into a grayscale RGB image array.

    Args:
    boolean_array (np.ndarray): 2D NumPy array of boolean values (True/False).

    Returns:
    np.ndarray: RGB image array (height x width x 3) as a NumPy array of dtype np.uint8.
    """
    # Ensure input array is boolean and 2D
    # assert isinstance(boolean_array, np.ndarray), "Input must be a NumPy array"
    # assert boolean_array.dtype == np.bool, "Input array must be boolean"
    # assert boolean_array.ndim == 2, "Input array must be 2-dimensional"

    # Convert boolean array to uint8 with values 0 (False) and 255 (True)
    grayscale_array = np.where(boolean_array, 255, 0).astype(np.uint8)

    # Create a grayscale image from the array
    grayscale_image = Image.fromarray(grayscale_array, mode='L')

    # Convert grayscale image to RGB array
    rgb_array = np.array(grayscale_image.convert('RGB'))

    return rgb_array

def boolean_array_to_opencv_image(boolean_array):
    """
    Converts a 2D boolean NumPy array into an OpenCV grayscale image.

    Args:
    boolean_array (np.ndarray): 2D NumPy array of boolean values (True/False).

    Returns:
    np.ndarray: OpenCV grayscale image array (height x width) as a NumPy array of dtype np.uint8.
    """
    # Ensure input array is boolean and 2D
    # assert isinstance(boolean_array, np.ndarray), "Input must be a NumPy array"
    # assert boolean_array.dtype == np.bool, "Input array must be boolean"
    # assert boolean_array.ndim == 2, "Input array must be 2-dimensional"

    # Convert boolean array to uint8 with values 0 (False) and 255 (True)
    grayscale_array = np.where(boolean_array, 255, 0).astype(np.uint8)

    return grayscale_array


def grow_regions_by_one_pixel(input_array: np.ndarray) -> np.ndarray:
    """
    Grows regions with value 1 in a 2D boolean array by one pixel.
    
    Args:
        input_array (np.ndarray): A 2D boolean numpy array.
        
    Returns:
        np.ndarray: A 2D boolean numpy array with regions grown by one pixel.
    """
    # Check if the input array is a 2D boolean array
    if not (isinstance(input_array, np.ndarray) and input_array.ndim == 2 and input_array.dtype == bool):
        raise ValueError("Input must be a 2D boolean numpy array.")
    
    # Define the structuring element for dilation (a 3x3 matrix of ones)
    structuring_element = np.ones((3, 3), dtype=bool)
    
    # Perform binary dilation to grow regions by one pixel
    grown_array = binary_dilation(input_array, structure=structuring_element)
    
    return grown_array

def solar_panel_selector(bb_height, bb_width):
    min_px_length = 1  # virtually deactivates the process
    num_possible_panels = 96 # len(spanel_filenames)
    selected_spanel_id = random.randint(1, num_possible_panels)
    # print(f'random : {selected_spanel_id}')
    solar_panel_image = get_image(asset_path + f'panel_{selected_spanel_id:03d}.jpg') # spanel_filenames[selected_spanel_id])
    if bb_height < min_px_length or bb_height < min_px_length:
        solar_panel_mipmap = create_mipmaps(solar_panel_image)
        solar_panel_texture = select_mipmap_level(solar_panel_mipmap, bb_width, bb_height)
        print(f'mipmap texture: {solar_panel_texture.shape}')
    else:
        solar_panel_texture = solar_panel_image # Later we take a AOB portion
    return solar_panel_texture


def apply_solar_panel_texture(aerial_image, bounding_box_list, aerial_mask, aliased_border_flag=True):
    """
    Apply a solar panel texture to a specified bounding box area on an aerial image.

    Args:
        aerial_image (numpy.ndarray): The input aerial image in RGB format.
        bounding_box_list (list): A list of (4,2) array containing the corners of the arbitrarily oriented bounding boxes.
        solar_panel_texture (numpy.ndarray): The solar panel texture image in RGB format.
        aerial_mask (numpy.ndarray): A boolean mask with the regions wher we can map the solar panel
        aliased_border_flag (bool): modifies border management

    Returns:
        numpy.ndarray: The modified aerial image with the solar panel texture applied.
    """
    
    # Masks initialization
    black_image = np.zeros_like(aerial_image)
    if aliased_border_flag:
        # aerial_mask = grow_regions_by_one_pixel(aerial_mask)
        aerial_mask_grow = boolean_array_to_opencv_image(grow_regions_by_one_pixel(aerial_mask))
    aerial_mask = boolean_array_to_opencv_image(aerial_mask)
    masked_image = cv2.bitwise_or(aerial_image, black_image, mask=cv2.bitwise_not(aerial_mask))
    
    for bounding_box in bounding_box_list:
        # swap rows and cols 
        new_bounding_box = np.zeros_like(bounding_box)
        new_bounding_box[:,0] = bounding_box[:,1]
        new_bounding_box[:,1] = bounding_box[:,0]
        bounding_box = new_bounding_box

        # Calculate the approximate boundig box dimensions
        bb_width = int(np.linalg.norm(bounding_box[0,:] - bounding_box[1,:]))
        bb_height = int(np.linalg.norm(bounding_box[1,:] - bounding_box[2,:]))

        # print(bb_width)
        # print(bb_height)
        solar_panel_texture = solar_panel_selector(bb_height, bb_width)
        # Resize the solar panel texture to the size of the bounding box       
        solar_panel_resized = cv2.resize(solar_panel_texture, (bb_width, bb_height))
        # Define the destination points for the affine transformation
        src_pts = np.array([
            [1, 1],
            [bb_width - 1, 1],
            [bb_width - 1, bb_height - 1],
            [1, bb_height - 1]
        ], dtype='float32')

        # Define the source points from the bounding box
        dst_pts = np.array(bounding_box, dtype='float32')

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Warp the resized solar panel texture to fit the bounding box on the aerial image
        warped_panel = cv2.warpPerspective(solar_panel_resized, M, (aerial_image.shape[1], aerial_image.shape[0]))
        if aliased_border_flag:
            masked_panel = cv2.bitwise_and(warped_panel, aerial_image, mask=aerial_mask_grow)
        else:
            masked_panel = cv2.bitwise_and(warped_panel, aerial_image, mask=aerial_mask)
        # TODO: Update aerial_mask
        masked_image = cv2.add(masked_panel, masked_image)
        # aerial_image = masked_image

        # plt.figure()
        # if aliased_border_flag:
        #     plt.imshow(aerial_mask_grow)
        # else:
        #     plt.imshow(aerial_mask)
        # plt.show()
        # plt.figure()
        # plt.imshow(masked_panel)
        # plt.show()
        # plt.figure()
        # plt.imshow(masked_image)
        # plt.show()

    return masked_image

def list_png_files(directory):
    """
    Returns a list of filenames with the '.png' extension in the given directory.
    
    Args:
        directory (str): The name of the directory to search for .png files.
        
    Returns:
        List[str]: A list of filenames with the .png extension.
    """
    # Initialize an empty list to store the names of .png files
    png_files = []
    
    # Iterate over the files in the given directory
    for filename in os.listdir(directory):
        # Check if the file has a .png extension
        if filename.endswith('.png'):
            # Add the .png file to the list
            png_files.append(filename)
    
    return png_files

def create_mipmaps(image):
    mipmaps = [image]
    while image.shape[0] > 1 and image.shape[1] > 1:
        image = cv2.pyrDown(image)
        mipmaps.append(image)
    return mipmaps


def select_mipmap_level(mipmaps, target_width, target_height):
    for mip in mipmaps:
        if mip.shape[1] <= target_width and mip.shape[0] <= target_height:
            return mip
    return mipmaps[-1]


def save_boolean_array_as_image(boolean_array, folder_name, file_name):
    """Saves a 2D boolean numpy array as an RGB image in PNG format.
    
    Args:
        boolean_array (numpy.ndarray): 2D boolean numpy array to be saved as an image.
        folder_name (str): Name of the folder where the file will be saved.
        file_name (str): Full name of the file to be saved (including .png extension).
    """
    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)
    # Create a 3D numpy array with RGB channels, initialized to white
    height, width = boolean_array.shape
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    # Set black color for True values in the boolean array
    image_array[boolean_array] = [255, 255, 255]
    # Create an Image object from the numpy array
    image = Image.fromarray(image_array, 'RGB')
    # Construct the full file path
    file_path = os.path.join(folder_name, file_name)
    # Save the image as a PNG file
    image.save(file_path)


def save_rgb_image(array: np.ndarray, folder_name: str, file_name: str) -> None:
    """Saves an RGB image from a numpy array to a PNG file.

    Args:
        array (np.ndarray): A numpy array containing the RGB image data.
        folder_name (str): The name of the folder where the file will be saved.
        file_name (str): The complete name of the file to be saved (including the .png extension).

    Raises:
        ValueError: If the input array is not a valid RGB image.
    """
    # Ensure the input array is a valid RGB image
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Input array must be an RGB image with shape (height, width, 3).")
    # Create the folder if it does not exist
    os.makedirs(folder_name, exist_ok=True)
    # Full path to save the image
    file_path = os.path.join(folder_name, file_name)
    # Convert the numpy array to an Image object
    image = Image.fromarray(array.astype('uint8'), 'RGB')
    # Save the image to the specified file path
    image.save(file_path)


if __name__ == '__main__':
    # input_path = 'C:/DATASETs/UC3Synthesis/syntheticDM/'
    input_path = 'C:/DATASETs/UC3Synthesis/UC3DM/'
    asset_path = './textures/'
    aerial_filenames = list_png_files(input_path)
    num_total_images = len(aerial_filenames)
    str_to_cut = 4 # 11
    tolerance = 22+3
    min_roof_size = 100
    min_solidity = 0.93
    max_perimeter_ratio = 6.5

    for count, input_filename in enumerate(aerial_filenames): # 200
        subset_i = int(count/2604) + 1
        output_path = f'C:/DATASETs/UC3Synthesis/GMV-SD4EO-AI-generated EO Dataset-SolarPanels-{subset_i:02d}/'
        print(f' {count} / {num_total_images} : {input_filename}')
        initial_image = get_image(input_path + input_filename)  

        boolean_mask, oriented_bounding_boxes, axis_aligned_bboxes = detect_white_regions(initial_image, tolerance, min_roof_size, min_solidity, max_perimeter_ratio)
        # composed_image_b = apply_solar_panel_texture(initial_image, oriented_bounding_boxes, boolean_mask, aliased_border_flag=True)
        composed_image_w = apply_solar_panel_texture(initial_image, oriented_bounding_boxes, boolean_mask, aliased_border_flag=False)
        # plt.figure()
        # plt.imshow(boolean_mask)
        # plt.show()
        # plt.figure()
        # plt.imshow(composed_image)
        # plt.show()
        mask_filename = input_filename[:-str_to_cut] + '_mask.png'
        bb_filename = input_filename[:-str_to_cut] + '_boundingboxes.npz'
        # output_filename_b = input_filename[:-str_to_cut] + '_b_solarpanel.png'
        output_filename_w = input_filename[:-str_to_cut] + '_w_solarpanel.png'
        # yolo_filename_b = output_filename_b[:-4] + '.txt'
        yolo_filename_w = output_filename_w[:-4] + '.txt'
        save_boolean_array_as_image(boolean_mask, output_path, mask_filename)
        np.savez(output_path+ bb_filename, oriented_bounding_boxes=oriented_bounding_boxes)
        # save_rgb_image(composed_image_b, output_path, output_filename_b)
        save_rgb_image(composed_image_w, output_path, output_filename_w)
        # write_yolo_bounding_boxes(output_path + yolo_filename_b, axis_aligned_bboxes)
        write_yolo_bounding_boxes(output_path + yolo_filename_w, axis_aligned_bboxes)





