# File Name:     band_histogram.py
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

# This script help us to estimate/study the optimal original dynamic range 
# for teh selected band in the set of JP2 files inside a folder (and subfolders)

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image



def analyze_jp2_files(folder: str, key: str, histogram_n_bins: int = 1024):
    """
    Searches for JP2 files in the given folder and its subfolders that contain the key in their filename,
    and analyzes the content of the matching JP2 files in blocks to save memory.
    
    Args:
        folder (str): The folder to search for JP2 files.
        key (str): The key to search for in JP2 filenames.
        histogram_bins (int): Number of bins of the histogram
        
    Returns:
        tuple: A tuple containing the minimum and maximum values found across all the JP2 files.
        histogram: a numpy vector with the accumulated histogram
    """
    default_min = 0             # Default large range to explore
    default_max = 1024.0 * 20   # Default large range to explore
    # List to store file paths of matching JP2 files
    matching_files = []
    
    # Walk through the folder and subfolders to find matching JP2 files
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.jp2') and key in file:
                matching_files.append(os.path.join(root, file))
    
    # If no matching files are found, return None for min and max
    if not matching_files:
        print("No matching JP2 files found.")
        return None, None
       
    # Initialize histogram
    acc_histogram = np.zeros(histogram_n_bins, dtype=int)
    
    # Iterate over matching files and collect pixel values
    print('Processing...')
    for count, file_path in enumerate(matching_files):
        print(f'{count}/{len(matching_files)} :  {file_path}')
        # Open the JP2 file
        with Image.open(file_path) as img:
            img_array = np.array(img)
            new_histogram = np_calculate_histogram(img_array, default_min, default_max, histogram_n_bins)
            acc_histogram = acc_histogram + new_histogram
    
    return default_min, default_max, acc_histogram


def np_calculate_histogram(array_2d, min_val, max_val, num_bins=1024):
    """
    Calculate a histogram for a 2D numpy array with 1024 bins.
    
    Args:
        array_2d (numpy.ndarray): 2D array of numerical values.
        min_val (float): Minimum value of the range for the histogram.
        max_val (float): Maximum value of the range for the histogram.
        num_bins (int): Number of bins

    Returns:
        numpy.ndarray: A 1D array of length 1024 with the histogram counts.
    """   
    # Calculate the bin edges
    # bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Flatten the 2D array to a 1D array
    flat_array = array_2d.flatten()
    
    # Use numpy's histogram function to calculate the histogram
    histogram, _ = np.histogram(flat_array, bins=num_bins, range=(min_val, max_val))
    # plt.figure()
    # plt.hist(flat_array, bins=100)
    # plt.show()
    
    return histogram


def fast_calculate_histogram(array_2d, min_val, max_val, num_bins=1024):
    """
    Calculate a histogram for a 2D numpy array with 1024 bins.
    
    Args:
        array_2d (numpy.ndarray): 2D array of numerical values.
        min_val (float): Minimum value of the range for the histogram.
        max_val (float): Maximum value of the range for the histogram.
        num_bins (int): Number of bins

    Returns:
        numpy.ndarray: A 1D array of length 1024 with the histogram counts.
    """   
    # Flatten the 2D array to a 1D array
    flat_array = array_2d.flatten()
    
    # Filter out values outside the min and max range
    mask = (flat_array >= min_val) & (flat_array < max_val)
    filtered_array = flat_array[mask]
    
    # Calculate the bin indices for each value
    bin_indices = ((filtered_array - min_val) / (max_val - min_val) * num_bins).astype(np.int32)
    
    # Use numpy's bincount to count the occurrences of each bin index
    histogram = np.bincount(bin_indices, minlength=num_bins)
    
    return histogram


def slow_calculate_histogram(array_2d, min_val, max_val, num_bins=1024):
    """
    Calculate a histogram for a 2D numpy array with 1024 bins.
    
    Args:
        array_2d (numpy.ndarray): 2D array of numerical values.
        min_val (float): Minimum value of the range for the histogram.
        max_val (float): Maximum value of the range for the histogram.
        num_bins (int): Number of bins
        
    Returns:
        numpy.ndarray: A 1D array of length 1024 with the histogram counts.
    """
    # Initialize the histogram vector with zeros
    histogram = np.zeros(num_bins)
    
    # Calculate the bin width
    bin_width = (max_val - min_val) / num_bins
    
    # Flatten the 2D array to a 1D array
    flat_array = array_2d.flatten()
    
    # Iterate through each value in the flattened array
    for value in flat_array:
        if min_val <= value < max_val:
            # Calculate the appropriate bin index for the current value
            bin_index = int((value - min_val) / bin_width)
            # Increment the corresponding bin in the histogram
            histogram[bin_index] += 1
    
    return histogram


def plot_histogram(histogram, min_val, max_val, range_min, range_max, band_id):
    """
    Plot a histogram using matplotlib based on the provided histogram data.
    
    Args:
        histogram (numpy.ndarray): A 1D array with histogram counts.
        min_val (float): Minimum value of the range for the histogram.
        max_val (float): Maximum value of the range for the histogram.
        range_min (float): Minimum value of the suggested range for dynamic range.
        range_max (float): Maximum value of the suggested range for dynamic range.
        band_id (str): string with the band ID
    """
    # Define the number of bins
    num_bins = len(histogram)
    
    # Calculate the bin edges
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], histogram, width=(max_val - min_val) / num_bins, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of accumulated {band_id}')
    plt.xlim(min_val, max_val)
    plt.grid(True)
    plt.axvline(x=range_min, color='r', linestyle='--', label=f'range min = {range_min}')
    plt.axvline(x=range_max, color='b', linestyle='--', label=f'range max = {range_max}')
    plt.legend()
    plt.show()



def large_mem_analyze_jp2_files(folder: str, key: str):
    """
    Searches for JP2 files in the given folder and its subfolders that contain the key in their filename,
    and analyzes the content of the matching JP2 files.
    
    Args:
        folder (str): The folder to search for JP2 files.
        key (str): The key to search for in JP2 filenames.
        
    Returns:
        tuple: A tuple containing the minimum and maximum values found across all the JP2 files.
    """
    
    # List to store file paths of matching JP2 files
    matching_files = []
    
    # Walk through the folder and subfolders to find matching JP2 files
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.jp2') and key in file:
                matching_files.append(os.path.join(root, file))
    
    # If no matching files are found, return None for min and max
    if not matching_files:
        print("No matching JP2 files found.")
        return None, None
    
    # List to store pixel values from all matching JP2 files
    pixel_values = []
    
    # Iterate over matching files and collect pixel values
    for count, file_path in enumerate(matching_files):
        print(f'{count}/{len(matching_files)} :  {file_path}')    
        # Open the JP2 file and convert it to a numpy array
        with Image.open(file_path) as img:
            img_array = np.array(img)
            # Flatten the array and extend it to the pixel values list
            pixel_values.extend(img_array.flatten())
    
    # Convert pixel values to a numpy array for easier analysis
    pixel_values = np.array(pixel_values)
    
    # Calculate the minimum and maximum pixel values
    min_value = pixel_values.min()
    max_value = pixel_values.max()
    
    # Plot the histogram of the pixel values
    plt.hist(pixel_values, bins=256, color='blue', alpha=0.7)
    plt.title('Histogram of Pixel Values')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    
    return min_value, max_value


def plot_histogram_2D(array_2d):
    """
    Plots a histogram of the values in a 2D NumPy array.
    
    Args:
        array_2d (np.ndarray): A 2D numpy array whose values are to be plotted in a histogram.
        
    Returns:
        None
    """
    # Flatten the 2D array to a 1D array
    flattened_array = array_2d.flatten()
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flattened_array, bins=30, alpha=0.75, color='blue', edgecolor='black')
    
    # Adding titles and labels
    plt.title('Histogram of 2D Array Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Show the plot
    plt.show()


def find_range_for_98_percent(histogram, percentage=99.0):
    """
    Finds the minimum and maximum index range that contains the given percentage of occurrences.
    
    Args:
        histogram (list): A list of integers representing the histogram.
        percentage (float, optional): The percentage of occurrences to consider. Default is 98.
        
    Returns:
        tuple: A tuple containing the minimum and maximum index that contains the percentage of occurrences.
    """  
    total_occurrences = sum(histogram)
    target_occurrences = total_occurrences * (percentage / 100.0)

    cumulative_sum = 0
    min_index = 0
    max_index = len(histogram) - 1

    # Find the minimum index that starts the range
    for i, count in enumerate(histogram):
        cumulative_sum += count
        if cumulative_sum >= target_occurrences:
            max_index = i
            break
    
    cumulative_sum = 0
    
    # Find the maximum index that ends the range
    for i in range(len(histogram)-1, -1, -1):
        cumulative_sum += histogram[i]
        if cumulative_sum >= target_occurrences:
            min_index = i
            break
    
    return min_index, max_index


if __name__ == "__main__":
    flag_recalculate = False
    key_word = '_B08_'
    folder_path = 'C:/DATASETs/UC2Training/Sentinel2/'
    band_id = key_word[1:-1]

    if flag_recalculate:
        min_val, max_val, histogram = analyze_jp2_files(folder_path, key_word)
        np.savez(f'./histograms/histogram{band_id}', min_val=min_val, max_val=max_val, histogram=histogram)

    # Cargar el archivo .npz
    hist_file = np.load(f'./histograms/histogram{band_id}.npz')
    # Extraer las variables del archivo
    min_val = hist_file['min_val']
    max_val = hist_file['max_val']
    histogram = hist_file['histogram']

    print(f"Default minimum value: {min_val}, Maximum value: {max_val}")
    new_min_index, new_max_index = find_range_for_98_percent(histogram, percentage=98.5)
    num_elems = len(histogram)
    delta_range = (max_val - min_val)*1.0/num_elems
    new_min_val = new_min_index * delta_range
    new_max_val = new_max_index * delta_range
    print(f"Optimal minimum value: {new_min_val}, Maximum value: {new_max_val} for 98% of occurences")
    plot_histogram(histogram, min_val, max_val, new_min_val, new_max_val, band_id)


