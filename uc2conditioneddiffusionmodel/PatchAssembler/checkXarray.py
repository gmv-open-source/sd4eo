# File Name:     checkXarray.py
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
import xarray as xr


if __name__ == "__main__":
    folder = 'C:/DATASETs/UC2Synthesis/assembled.Zenodov2/'
    filename = 'assembled_Paris_v05.nc'
    data_array = xr.open_dataset(folder+filename)
    print(data_array)

