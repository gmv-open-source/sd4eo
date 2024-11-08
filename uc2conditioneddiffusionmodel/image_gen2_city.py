# File Name:     image_gen2_city.py
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

# Image inference for new diffusion model (v2)
#

# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from PIL import Image
# import torch
# controlnet = ControlNetModel.from_pretrained("./uk_model_good/controlearth", torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#         "./runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
#         safety_checker = None, requires_safety_checker = False
#     )
    
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()

# num_original_imgs = 3
# num_gen_images = 6
# img_n_shift = 0

# control_images = []
# for i in range(num_original_imgs):
#     control_images.append(Image.open(f'./images/example{i+img_n_shift}.png').convert("RGB"))
#     prompt = "convert this openstreetmap into its satellite view"
#     for j in range(num_gen_images):
#         image = pipe(prompt, num_inference_steps=50, image=control_images[i]).images[0].save(f'./output5/img{(i+img_n_shift):02d}-generated-{j:02d}.png')

"""
Tools for running inference of a pretrained ControlNet model.
Adapted from gradio_scribble2image.py from the original authors and Gabe Grand example.
"""


import sys

# sys.path.append("..")
from share import *

# import cv2
import einops  # Horrible library to do transpositions in Einstein notation
# import gradio as gr
import cv2
import numpy as np
import torch
import random
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from pytorch_lightning import seed_everything
# from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from dataset import *
import matplotlib.pyplot as plt

available_cities_IDs = ['Angers', 'Toulouse', 'Bordeaux', 'Limoges', 'Clermont-Ferrand', 'Troyes', 'Le Mans', 'Madrid', 'Paris', 'Poitiers', 'Niort'] # 'Rouen']

A_PROMPT_DEFAULT = "well iluminated, visible spectrum image from satellite view"
N_PROMPT_DEFAULT = "cropped, worst quality, low quality"
# A_PROMPT_DEFAULT = "best quality, extremely detailed"
# N_PROMPT_DEFAULT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"


def run_sampler(
    model,
    input_image: np.ndarray,
    prompt: str,
    num_samples: int = 1,
    # image_resolution: int = 512, # <-- IT MUST BE 512x512x3
    seed: int = -1,
    a_prompt: str = A_PROMPT_DEFAULT,
    n_prompt: str = N_PROMPT_DEFAULT,
    guess_mode=False,
    strength=1.0,
    ddim_steps=50,
    eta=0.0,
    scale=9.0,
    show_progress: bool = True,
):
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()

        ddim_sampler = DDIMSampler(model)

        # img = resize_image(HWC3(input_image), image_resolution)
        img = input_image # IT MUST BE 512x512x3
        H, W, C = img.shape

        if num_samples > 1:
            multiple_img = np.zeros((num_samples, img.shape[0], img.shape[1], img.shape[2]))
            for rep_i in range(num_samples):
                multiple_img[rep_i, :, :, :] = img[:,:,:]

        old_code_flag = False
        if old_code_flag:
            detected_map = np.zeros_like(img, dtype=np.uint8)
            detected_map[np.min(img, axis=2) < 127] = 255 # Está generando una imagen binaria, tal que si el valor mínimo de los 3 canales del pixel está por debajo de 127 pone el valor 255 y si tiene valores mayores, pone simplemente 0
            # NOTA: Nosotros no tenemos ese tipo de imagen para condicionar nuestro modelo
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0 # normalize in GPU
            control = torch.stack([control for _ in range(num_samples)], dim=0)  # We generate num_samples from this guidance image at once
            control = einops.rearrange(control, "b h w c -> b c h w").clone()  # b = number of samples to generate 
        else:
            control = torch.from_numpy(img.copy()).float().cuda()
            control = torch.stack([control for _ in range(num_samples)], dim=0)  # We generate num_samples from this guidance image at once
            control = einops.rearrange(control, "b h w c -> b c h w").clone()  # b = number of samples to generate           

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            show_progress=show_progress,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)

        # Se restaura el rango dinámico de la imagen (aquí se supone que son estilo "PNGs")
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]

        return results


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


def get_val_image(base_path, city_id, col, row):
    path = base_path + city_id + '/'
    filename = f'patch_{city_id}_{col:03}_{row:03}_OSM.png'
    # path = ''
    # filename = f'./T31UFQ/patch_T31UFQ_060_054_OSM.png' # CHANGE!

    condition_map = cv2.imread(path + filename)
    # Do not forget that OpenCV read images in BGR order.
    condition_map = cv2.cvtColor(condition_map, cv2.COLOR_BGR2RGB)
    # Careful!!! Surce image was normalized to possitive values: [0, 1].
    # While target S2 RGB images were normalized in a different way [-1, 1].
    condition_map = condition_map.astype(np.float32)
    condition_map = resize_image_with_filter(condition_map, (512,512))
    condition_map = condition_map / 255.0
    return condition_map

def check_extent_file(city_id, base_path):
    filename = f'np_extent_{city_id}_EPSG_3857.npz'
    extent_file = np.load(base_path + filename)
    extent_reg = extent_file['extent_reg']
    return extent_reg.shape[0], extent_reg.shape[1]


if __name__ == "__main__":
    ukserver2_flag = False
    IR_flag = False
    multiple_image_support = False
    debug_show_result = False
    model = create_model('./models/cldm_v15.yaml').cpu()
    rgb_models = {3: "./RGB_Cond_Models/epoch=3-step=16636.ckpt", 9: "./RGB_Cond_Models/epoch=9-step=41590.ckpt", 27: "./RGB_Cond_Models/epoch=27-step=116452.ckpt", 30: "./RGB_Cond_Models/epoch=30-step=128929.ckpt", 33: "./RGB_Cond_Models/epoch=33-step=141406.ckpt", 34: "./RGB_Cond_Models/epoch=34-step=145565.ckpt", 69: "./RGB_Cond_Models/control_sd15_RGB80.ckpt", 80: "./RGB_Cond_Models/control_sd15_RGB80.ckpt"}
    ir_models = {31: "./IR_Cond_Models/epoch=31-step=133088.ckpt", 34: "./IR_Cond_Models/epoch=34-step=145565.ckpt"}

    epoch_i = 34
    if IR_flag:
        model.load_state_dict(load_state_dict(ir_models[epoch_i], location='cuda'))
        spec_str = 'IR'
    else:
        model.load_state_dict(load_state_dict(rgb_models[epoch_i], location='cuda'))
        spec_str = 'RGB'
    prompt = "visible spectrum image from satellite view"
    
    diff_steps = 50
    seed_ini = 142+100+100 
    num_variations_same_map = 15
    init_num_variations = 50 # TODO: swap to ZERO later!
    model = model.cuda()

    # This line must become a for loop !
    for city_ID in available_cities_IDs:
        print(city_ID)
        if ukserver2_flag:
            input_base_path = '/workspace/MISC/OSM/'
            output_path = f'/workspace/MISC/Synth{spec_str}/{city_ID}/'
        else:
            input_base_path = 'C:/DATASETs/UC2Synthesis/OSM/'
            output_path = f'C:/DATASETs/UC2Synthesis/Synth{spec_str}/{city_ID}/'
            # output_path = f'./T31UFQ/'
        max_col, max_row = check_extent_file(city_ID, input_base_path) 

        for col in range(max_col):
            for row in range(max_row):
                print(f'*** {city_ID} : col {col}  row {row}')
                guidance_map_image = get_val_image(input_base_path, city_ID, col, row)

                if multiple_image_support:
                    results = run_sampler(model, guidance_map_image, prompt, num_samples=num_variations_same_map, ddim_steps=diff_steps, seed=seed_ini)
                    for version_i in range(init_num_variations, init_num_variations+num_variations_same_map):
                        out_image = Image.fromarray(results[version_i], "RGB")
                        out_image.save(output_path + f"synthS2_{city_ID}_{spec_str}m{epoch_i:03}_c{col:03}_r{row:03}_i{diff_steps:03}_v{version_i:02}.png")
                        if debug_show_result:
                            plt.figure()
                            plt.imshow(results[version_i])
                            plt.show()
                else:
                    for version_i in range(init_num_variations, init_num_variations+num_variations_same_map):
                        results = run_sampler(model, guidance_map_image, prompt, num_samples=1, ddim_steps=diff_steps, seed=seed_ini + version_i)
                        out_image = Image.fromarray(results[0], "RGB")
                        out_image.save(output_path + f"synthS2_{city_ID}_{spec_str}m{epoch_i:03}_c{col:03}_r{row:03}_i{diff_steps:03}_v{version_i:02}.png")
                        if debug_show_result:
                            plt.figure()
                            plt.imshow(results[0])
                            plt.show()         





