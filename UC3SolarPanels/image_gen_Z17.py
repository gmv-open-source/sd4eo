# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     image_gen_Z17.py
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

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import os
from PIL import Image
import shutil
import torch


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


if __name__ == '__main__':
    ukserver2_flag = False # True
    
    num_new_gen_variants = 1
    img_n_shift = 0

    if ukserver2_flag:  # folders related to solar panels or UC3...
        input_path = '/workspace/MISC/UC3OSM/'
        move_input_path = '/workspace/MISC/UC3OSMprocessed/'
        output_path = '/workspace/MISC/UC3DM/'
    else:
        input_path = 'C:/DATASETs/UC3Synthesis/curatedOSM/current_batch/'
        move_input_path = 'C:/DATASETs/UC3Synthesis/curatedOSM/already_processed/'
        output_path = 'C:/DATASETs/UC3Synthesis/syntheticDM/'        

    # Pipeline Setup
    controlnet = ControlNetModel.from_pretrained("./syntheticS2model/controlearth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "./runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
            safety_checker = None, requires_safety_checker = False
        )
        
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Synthetize the conditioned images at zoom level 17 
    filename_list = list_png_files(input_path)
    total_files = len(filename_list)
    for count, filename in enumerate(filename_list):
        print(f'* {count} / {total_files} : {filename}')
        original_image = Image.open(input_path + filename).convert("RGB")
        resized_image = original_image.resize((256, 256), Image.LANCZOS)
        prompt = "convert this openstreetmap into its satellite view"
        for var_i in range(num_new_gen_variants):
            output_filename = filename[:-7] + f'_DM_{var_i:02d}.png'
            image = pipe(prompt, num_inference_steps=50, image=resized_image).images[0].save(output_path + output_filename)
        shutil.move(input_path + filename, move_input_path + filename)


    # control_images = []
    # for i in range(num_original_imgs):
    #     control_images.append(Image.open(f'./images/example{i+img_n_shift}.png').convert("RGB"))
    #     prompt = "convert this openstreetmap into its satellite view"
    #     for j in range(num_gen_images):
    #         image = pipe(prompt, num_inference_steps=50, image=control_images[i]).images[0].save(f'./output5/img{(i+img_n_shift):02d}-generated-{j:02d}.png')



