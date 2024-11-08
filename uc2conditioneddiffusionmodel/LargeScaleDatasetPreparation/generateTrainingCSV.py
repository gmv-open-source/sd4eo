# File Name:     generateTrainingCSV.py
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
# We need a CSV to specify the correspondence between the pais of images, 
# although they share the same filename


# /mnt/SD4EO/02_UC2-Settlements/02_SyntheticData/01_GMV-SES/UC2Training512/source/
# /mnt/SD4EO/02_UC2-Settlements/02_SyntheticData/01_GMV-SES/UC2Training512/targetRGB/
# /mnt/SD4EO/02_UC2-Settlements/02_SyntheticData/01_GMV-SES/UC2Training512/targetIR/

import copy
import json
import numpy as np
import os
from PIL import Image
import shutil
from common_tile_utils import available_Tiles_IDs

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

import csv


def check_substrings_in_filename(filename, substrings):
  """
  Checks if any of the given substrings are present in a filename.

  Args:
    filename: The filename to check.
    substrings: A list of substrings to search for.

  Returns:
    True if any of the substrings are present in the filename, False otherwise.
  """

  for substring in substrings:
    if substring in filename:
      return True
  return False


def write_csv(filenames, source_folder, target_folder, csv_path, excluded_substrings, flag_RGB):
    """Writes a CSV file with three fields for each filename.

    Args:
        filenames (list): A list of filenames.
        source_folder (str): The name of the source folder.
        target_folder (str): The name of the target folder.
        csv_path (str): The path to the CSV file.
        excluded_substrings (str): if the string appears in the filename, that entry is not writen in the CSV

    Raises:
        IOError: If an error occurs while writing the CSV file.
    """
    if flag_RGB:
       prompt = "'visible spectrum image from satellite view'"
    else:
       prompt = "'infrared spectrum image from satellite view'"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(["source", "target", "prompt"])
        for filename in filenames:
            if check_substrings_in_filename(filename, excluded_substrings) == False:
                source_path = os.path.join(source_folder, filename)
                target_path = os.path.join(target_folder, filename)
                writer.writerow([source_path, target_path, prompt])



def write_json(filenames, source_folder, target_folder, json_path, excluded_substrings, flag_RGB):
    """Writes a CSV file with three fields for each filename.

    Args:
        filenames (list): A list of filenames.
        source_folder (str): The name of the source folder.
        target_folder (str): The name of the target folder.
        csv_path (str): The path to the CSV file.
        excluded_substrings (str): if the string appears in the filename, that entry is not writen in the CSV
    """
    if flag_RGB:
       text_prompt = "visible spectrum image from satellite view"
    else:
       text_prompt = "infrared spectrum image from satellite view"

    # data = [
    #     {"source": "source/0.png", "target": "target/0.png", "prompt": "pale golden rod circle with old lace background"},
    #     {"source": "source/1.png", "target": "target/1.png", "prompt": "light coral circle with white background"},
    #     {"source": "source/2.png", "target": "target/2.png", "prompt": "aqua circle with light pink background"},
    #     {"source": "source/3.png", "target": "target/3.png", "prompt": "cornflower blue circle with light golden rod yellow background"}
    # ]
    
    with open(json_path, 'w') as f:
        for filename in filenames:
            if check_substrings_in_filename(filename, excluded_substrings) == False:
                source_path = os.path.join(source_folder, filename[:-6]+'OSM.png')
                target_path = os.path.join(target_folder, filename)
                # entry = {"file_name" : source_path, "image": source_path, "conditioning_image": target_path, "text": text_prompt}
                # entry2 = {"image": source_path, "conditioning_image": target_path, "text": text_prompt}
                entry = {"file_name" : source_path, "source": source_path, "target": target_path, "prompt": text_prompt}
                # entry2 = {"image": source_path, "conditioning_image": target_path, "text": text_prompt}
                f.write(json.dumps(entry) + '\n')
                # f.write(json.dumps(entry2) + '\n')


if __name__ == "__main__":
    origin_base_path =f'C:/DATASETs/UC2Training512/targetRGB/'
    server_source_path = 'source/' #'/workspace/UC2Training512/datasetRGB/source/'
    server_target1_path = 'target/' #'/workspace/UC2Training512/datasetRGB/target/'
    server_target2_path = 'target/' #'/workspace/UC2Training512/datasetIR/target/'
    png_filenames = list_png_files(origin_base_path)
    not_valid_tile = 'T31TEM'
    validation_tile = 'T31UFQ'
    excluded_tiles = [not_valid_tile, validation_tile]
    write_json(png_filenames, server_source_path, server_target1_path, 'metadata.jsonl', excluded_tiles, True)
    # write_csv(png_filenames, server_source_path, server_target2_path, 'train2.csv', excluded_tiles, False)
    non_validation_tiles = copy.deepcopy(available_Tiles_IDs) + [not_valid_tile]
    non_validation_tiles.remove(validation_tile)
    write_json(png_filenames, server_source_path, server_target1_path, 'prompt_v.json', non_validation_tiles, True)
    # write_csv(png_filenames, server_source_path, server_target2_path, 'validation2.csv', non_validation_tiles, False)


