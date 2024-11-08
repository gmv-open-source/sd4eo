# File Name:     gmv_train.py
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

# https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from loadmydatasetIR import MyDataset
from loadmydatasetRGB import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/control_sd15_RGB34.ckpt'
# resume_path = './models/control_sd15_ini.ckpt'
batch_size = 16
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=32, batch_size=batch_size, shuffle=True, prefetch_factor=2)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, accelerator='gpu', devices=1, precision=32, callbacks=[logger], max_epochs=250)
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
