# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     fromZIPtoGeoTIFF.py
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

import rasterio
import geopandas as gpd
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from multichannelFileFormatMS1 import bands_IDs

T11_ID_per_month = ['T11SPS_20220105T182741', 'T11SPS_20220204T182541', 'T11SPS_20220301T182259', 'T11SPS_20220405T181921', 'T11SPS_20220505T181921', 'T11SPS_20220604T181921', 'T11SPS_20220704T181931', 'T11SPS_20220823T181931', 'T11SPS_20220917T182019', 'T11SPS_20221002T182201', 'T11SPS_20221106T182539', 'T11SPS_20221206T182739']



def dump_rastered_polygons(unsorted_polygons, crs_shapefile, geotiff_path, flag_sort_by_area = False, flag_save_tif_files=False):
    # Step 2: Open the GeoTIFF file with rasterio
    rastered_polygons = []
    with rasterio.open(geotiff_path) as src:
        # Let's check the reference system
        crs_raster = src.crs
        print(f"CRS del GeoTIFF: {crs_raster}")
        # If it's different, we transform it just in the polygons
        if crs_shapefile != crs_raster:
            projected_polygons = unsorted_polygons.to_crs(crs_raster)
        else:
            projected_polygons = unsorted_polygons

        if flag_sort_by_area:
            # Calculate the area of each polygon and store it in a new column
            projected_polygons['area'] = projected_polygons.area
            # Sort the GeoDataFrame by the area column from largest to smallest
            polygons = projected_polygons.sort_values(by='area', ascending=False)
        else:
            polygons = projected_polygons

        # For each polygon in the shapefile
        for _, polygon in polygons.iterrows():
            # Step 3: Extract the portion of the image corresponding to the polygon
            out_image, out_transform = mask(src, [polygon['geometry']], crop=True)
            # Convert the image portion to a numpy array
            out_image_array = out_image.data
            
            # # Is it a numpy array ?
            # print(out_image_array.shape)
            # print(np.max(np.max(out_image_array)))

            # We could save the image portion as a new GeoTIFF file
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image_array.shape[1],
                "width": out_image_array.shape[2],
                "transform": out_transform
            })

            # Let's see each image portion before saving it into a file
            np_out_image_array = np.array(out_image_array)
            if flag_save_tif_files:
                plt.imshow(np.squeeze(np_out_image_array), interpolation='none')
                plt.colorbar()
                plt.show()
                with rasterio.open(f'./pieces/piece_{_}.tif', 'w', **out_meta) as dest:
                    dest.write(np_out_image_array)
            rastered_polygons.append(np_out_image_array)
    return rastered_polygons


if __name__ == "__main__":
    # Get the polygons of the selected crops
    # Step 1: Read the shapefile with geopandas
    # crop_type = 'alfalfa'
    # shapefile_path = 'C:/DMTA/projects/SD4EO/QGIS.ImperialValley/AlfalfaRectangles2.shp'
    # crop_type = 'durumwheat'
    # shapefile_path = 'C:/DMTA/projects/SD4EO/QGIS.ImperialValley/DurumWheatRectangles5.shp'
    # crop_type = 'corn'
    # shapefile_path = 'C:/DMTA/projects/SD4EO/QGIS.ImperialValley/CornRectangles5.shp'
    crop_type = 'sugarbeets'
    shapefile_path = 'C:/DMTA/projects/SD4EO/QGIS.ImperialValley/SugarbeetsRectangles5.shp'
    unsorted_polygons = gpd.read_file(shapefile_path)
    # Let's check the reference system
    crs_shapefile = unsorted_polygons.crs
    print(f"CRS del shapefile: {crs_shapefile}")
    for month_i in range(12):
        rastered_polygons_per_band = []
        for band_ID in bands_IDs:
            print(f"processing bands # {band_ID}")
            # This coding may requiere a large amount of RAM
            geotiff_path = f"C:/DATASETs/SD4EO.ImperialValley/GeoTIFFs/{T11_ID_per_month[month_i]}_{band_ID}_10m.tif"
            rastered_polygons = dump_rastered_polygons(unsorted_polygons, crs_shapefile, geotiff_path)
            rastered_polygons_per_band.append(rastered_polygons)
        num_bands = len(bands_IDs)
        num_polygons = len(rastered_polygons_per_band[0])
        # Now we assembly the xarrays for each polygon
        for polygon_index in range(num_polygons):
            res_x = rastered_polygons_per_band[0][polygon_index].shape[1]
            res_y = rastered_polygons_per_band[0][polygon_index].shape[2]
            print(f"processing polygon # {polygon_index} :  size {res_x} x {res_y}")
            base_3D_array = np.zeros((res_x, res_y, num_bands))
            for band_index in range(num_bands):
                base_3D_array[:,:,band_index] = rastered_polygons_per_band[band_index][polygon_index][0,:,:]
            # Transform as a xarray
            xr_array = xr.DataArray(base_3D_array, dims=['y', 'x', 'band'], coords={'band': bands_IDs})
            xr_array.to_netcdf(f"./original_parcels/orig_{crop_type}_{polygon_index}_{month_i}.nc")





