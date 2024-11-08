# File Name:     image_gen_Z17.py
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

# import ssl
# import requests

# if hasattr(ssl, '_create_unverified_context'):
#     ssl._create_default_https_context = ssl._create_unverified_context
# # Configura una sesión de requests
# session = requests.Session()
# session.verify = False  # Desactiva la verificación de SSL


# from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
# import torch

# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
# )


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import torch
controlnet = ControlNetModel.from_pretrained("./uk_model_good/controlearth", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "./runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
        safety_checker = None, requires_safety_checker = False
    )
    
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

num_original_imgs = 3
num_gen_images = 6
img_n_shift = 0

control_images = []
for i in range(num_original_imgs):
    control_images.append(Image.open(f'./images/example{i+img_n_shift}.png').convert("RGB"))
    prompt = "convert this openstreetmap into its satellite view"
    for j in range(num_gen_images):
        image = pipe(prompt, num_inference_steps=50, image=control_images[i]).images[0].save(f'./output5/img{(i+img_n_shift):02d}-generated-{j:02d}.png')


# """
# accelerate launch train_controlnet.py \
#  --pretrained_model_name_or_path="./runwayml/stable-diffusion-v1-5/" \
#  --controlnet_model_name_or_path="./uk_model_good/controlearth/" \
#  --output_dir="./model_out/" \
#  --dataset_name="/mnt/SD4EO/02_UC2-Settlements/02_SyntheticData/01_GMV-SES/UC2Training512/" \
#  --conditioning_image_column=source \
#  --image_column=target \
#  --caption_column=prompt \
#  --resolution=512 \
#  --learning_rate=1e-5 \
#  --validation_image "/mnt/SD4EO/02_UC2-Settlements/02_SyntheticData/01_GMV-SES/UC2Training512/source/patch_T31UFQ_000_000_S2.png" "/mnt/SD4EO/02_UC2-Settlements/02_SyntheticData/01_GMV-SES/UC2Training512/source/patch_T31UFQ_000_001_S2.png" "/mnt/SD4EO/02_UC2-Settlements/02_SyntheticData/01_GMV-SES/UC2Training512/source/patch_T31UFQ_000_003_S2.png" \
#  --validation_prompt "visible spectrum image from satellite view", "visible spectrum image from satellite view", "visible spectrum image from satellite view" \
#  --train_batch_size=4 \
#  --num_train_epochs=250 \
#  --tracker_project_name="controlnet" \
#  --enable_xformers_memory_efficient_attention \
#  --checkpointing_steps=5000 \
#  --checkpoints_total_limit=12\
#  --validation_steps=5000 \
#  --dataloader_num_workers=8 \
#  --report_to wandb 
# """



# """
# accelerate launch train_controlnet.py \
#  --pretrained_model_name_or_path="./runwayml/stable-diffusion-v1-5/" \
#  --controlnet_model_name_or_path="./uk_model_good/controlearth/" \
#  --output_dir="./model_out/" \
#  --dataset_name="/workspace/UC2Training512/datasetRGB/train/" \
#  --conditioning_image_column=source \
#  --image_column=target \
#  --caption_column=prompt \
#  --resolution=512 \
#  --learning_rate=1e-5 \
#  --validation_image "/workspace/UC2Training512/source/patch_T31UFQ_000_000_S2.png" "/workspace/UC2Training512/source/patch_T31UFQ_000_001_S2.png" "/workspace/UC2Training512/source/patch_T31UFQ_000_003_S2.png" \
#  --validation_prompt "visible spectrum image from satellite view", "visible spectrum image from satellite view", "visible spectrum image from satellite view" \
#  --train_batch_size=4 \
#  --num_train_epochs=250 \
#  --tracker_project_name="controlnet" \
#  --enable_xformers_memory_efficient_attention \
#  --checkpointing_steps=5000 \
#  --checkpoints_total_limit=12\
#  --validation_steps=5000 \
#  --dataloader_num_workers=8 \
#  --report_to wandb 
# """



# accelerate launch train_controlnet.py \
#  --pretrained_model_name_or_path="./runwayml/stable-diffusion-v1-5/" \
#  --controlnet_model_name_or_path="./uk_model_good/controlearth/" \
#  --output_dir="./model_out/" \
#  --train_data_dir="/workspace/UC2Training512/datasetRGB/train/" \
#  --conditioning_image_column=source \
#  --image_column=target \
#  --caption_column=prompt \
#  --resolution=512 \
#  --learning_rate=1e-5 \
#  --validation_image "/workspace/UC2Training512/source/patch_T31UFQ_000_000_S2.png" "/workspace/UC2Training512/source/patch_T31UFQ_000_001_S2.png" "/workspace/UC2Training512/source/patch_T31UFQ_000_003_S2.png" \
#  --validation_prompt "visible spectrum image from satellite view", "visible spectrum image from satellite view", "visible spectrum image from satellite view" \
#  --train_batch_size=4 \
#  --num_train_epochs=250 \
#  --tracker_project_name="controlnet" \
#  --enable_xformers_memory_efficient_attention \
#  --checkpointing_steps=5000 \
#  --checkpoints_total_limit=12\
#  --validation_steps=5000 \
#  --dataloader_num_workers=8 \
#  --report_to wandb 




# accelerate launch train_controlnet.py \
#  --pretrained_model_name_or_path="./runwayml/stable-diffusion-v1-5/" \
#  --controlnet_model_name_or_path="./uk_model_good/controlearth/" \
#  --output_dir="./model_out/" \
#  --train_data_dir="/workspace/UC2Training512/datasetRGB/train/" \
#  --conditioning_image_column="source" \
#  --image_column="target" \
#  --caption_column="prompt" \
#  --resolution=512 \
#  --learning_rate=1e-5 \
#  --validation_image "/workspace/UC2Training512/source/patch_T31UFQ_000_000_S2.png" "/workspace/UC2Training512/source/patch_T31UFQ_000_001_S2.png" "/workspace/UC2Training512/source/patch_T31UFQ_000_003_S2.png" \
#  --validation_prompt "visible spectrum image from satellite view", "visible spectrum image from satellite view", "visible spectrum image from satellite view" \
#  --train_batch_size=4 \
#  --num_train_epochs=250 \
#  --tracker_project_name="controlnet" \
#  --enable_xformers_memory_efficient_attention \
#  --checkpointing_steps=5000 \
#  --checkpoints_total_limit=12\
#  --validation_steps=5000 \
#  --dataloader_num_workers=8 \
#  --report_to wandb 



# pip install einops
# pip install open-clip-torch
# pip install omegaconf
# pip install pytorch-lightning==1.9.1
# pip install 




# python tool_add_control.py ./runwayml/stable-diffusion-v1-5/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt

# nohup python ./gmv_train.py > training_RGB_batch6.screen.log 2>&1 &


# root@4125626b391c:/workspace/mainDM# nohup python ./gmv_train.py > training_RGB_batch6.screen.log 2>&1 &
# [1] 384179

