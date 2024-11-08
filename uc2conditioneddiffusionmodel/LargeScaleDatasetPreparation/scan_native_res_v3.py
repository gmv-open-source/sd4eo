# File Name:     scan_native_res_v3.py
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
# This Python script contains functions to be executed within QGIS to scan 
# an area indicated by a set of coordinates (a pair of corners is sufficient 
# to delineate each region): it subdivides into patches of the corresponding 
# zoom level and extracts the OpenStreetMaps image along with the corresponding
# images for the Sentinel-2 spectral bands B4+B3+B2+B8. 
# It hasn't been easy because I'm not familiar with the pyQGIS API, and 
# it hasn't been possible to obtain the same set of Python 3.9 packages 
# to make QGIS work outside its console.

# from qgis.core import (
#     QgsProject,
#     QgsRasterLayer,
#     QgsCoordinateReferenceSystem,
#     QgsRectangle,
#     QgsCoordinateTransformContext,
#     QgsCoordinateTransform,
#     QgsLayout,
#     QgsLayoutItemMap,
#     QgsLayoutExporter,
#     QgsLayoutSize
# )
# from qgis.gui import QgsLayerTreeMapCanvasBridge
# from qgis.utils import iface
# from PyQt5.QtCore import QSize

import copy
import math
import numpy as np
import time
from common_tile_utils import available_Tiles_IDs, provide_inner_tile_coords


# def export_osm_image(longitude, latitude, zoom, output_file, remove_layer_flag=True):
def get_initial_extent(longitude, latitude, zoom, output_file, remove_layer_flag=True):
    # Configuración del entorno de trabajo
    project = QgsProject.instance()
    canvas = iface.mapCanvas()
    
    # URL del servicio de teselas de OpenStreetMap
    osm_url = "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    
    # Crea una capa raster usando la URL del servicio de teselas
    layer_name = "OpenStreetMap"
    raster_layer = QgsRasterLayer(osm_url, layer_name, "wms")
    
    if not raster_layer.isValid():
        QgsMessageLog.logMessage("La capa raster no es válida")
        return
    
    # Añade la capa al proyecto
    project.addMapLayer(raster_layer)
    
    # Configura el CRS al CRS EPSG:3857 (Web Mercator)
    crs = QgsCoordinateReferenceSystem("EPSG:3857")
    canvas.setDestinationCrs(crs)
    
    # Convertir longitud y latitud a coordenadas en EPSG:3857
    source_crs = QgsCoordinateReferenceSystem("EPSG:4326")
    dest_crs = QgsCoordinateReferenceSystem("EPSG:3857")
    transform_context = QgsCoordinateTransformContext()
    transform = QgsCoordinateTransform(source_crs, dest_crs, transform_context)
    point = transform.transform(longitude, latitude)
    
    # Calcular la extensión del mapa en función del nivel de zoom
    tile_size = 256/8 *1.5
    initial_resolution = 2 * 3.14159265358979323846 * 6378137 / tile_size
    resolution = initial_resolution / (2**zoom)
    
    extent_size = tile_size * resolution
    extent = QgsRectangle(point.x() - extent_size / 2, point.y() - extent_size / 2, point.x() + extent_size / 2, point.y() + extent_size / 2)
    
    # Ajustar la extensión del canvas
    canvas.setExtent(extent)
    canvas.refresh()
    
    # Crear un diseño de impresión
    layout = QgsLayout(project)
    layout.initializeDefaults()
    
    
    map_item = QgsLayoutItemMap(layout)
    map_item.setRect(0, 0, tile_size, tile_size)
    map_item.setExtent(extent)
    layout.addLayoutItem(map_item)
    
    # Exportar el diseño a imagen
    exporter = QgsLayoutExporter(layout)

    im_size = QgsLayoutExporter.ImageExportSettings().imageSize
    QgsMessageLog.logMessage(f'image size: {im_size}', level=Qgis.Info)
    im_dpi = QgsLayoutExporter.ImageExportSettings().dpi
    QgsMessageLog.logMessage(f'image dpi: {im_dpi}', level=Qgis.Info)
    im_cropToContents = QgsLayoutExporter.ImageExportSettings().cropToContents
    QgsMessageLog.logMessage(f'image crop contents: {im_cropToContents}', level=Qgis.Info)
    
    im_settings = QgsLayoutExporter.ImageExportSettings()
    im_settings.cropToContents = True
    im_settings.size = QSize(256,256)
    exporter.exportToImage(output_file, im_settings)
    # im_settings.dpi = 300
    # exporter.exportToImage(output_file[:-4] + '.tif', im_settings) 
    time.sleep(0.3)
    if remove_layer_flag:
        project.removeMapLayer(raster_layer)
        raster_layer = None
    return extent, dest_crs, raster_layer




def export_osm_image_by_extent(modified_extent, output_file, remove_layer_flag=True):
    # Configuración del entorno de trabajo
    project = QgsProject.instance()
    canvas = iface.mapCanvas()
    
    # URL del servicio de teselas de OpenStreetMap
    osm_url = "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    
    # Crea una capa raster usando la URL del servicio de teselas
    layer_name = "OpenStreetMap"
    raster_layer = QgsRasterLayer(osm_url, layer_name, "wms")
    
    if not raster_layer.isValid():
        QgsMessageLog.logMessage("La capa raster no es válida")
        return
    
    # Añade la capa al proyecto
    project.addMapLayer(raster_layer)
    
    # # Configura el CRS al CRS EPSG:3857 (Web Mercator)
    # crs = QgsCoordinateReferenceSystem("EPSG:3857")
    # canvas.setDestinationCrs(crs)
    
    # # Convertir longitud y latitud a coordenadas en EPSG:3857
    # source_crs = QgsCoordinateReferenceSystem("EPSG:4326")
    # dest_crs = QgsCoordinateReferenceSystem("EPSG:3857")
    # transform_context = QgsCoordinateTransformContext()
    # transform = QgsCoordinateTransform(source_crs, dest_crs, transform_context)
    # point = transform.transform(longitude, latitude)
    
    # # Calcular la extensión del mapa en función del nivel de zoom
    tile_size = 256/8 *1.5
    # initial_resolution = 2 * 3.14159265358979323846 * 6378137 / tile_size
    # resolution = initial_resolution / (2**zoom)
    
    # extent_size = tile_size * resolution
    # extent = QgsRectangle(point.x() - extent_size / 2, point.y() - extent_size / 2, point.x() + extent_size / 2, point.y() + extent_size / 2)
    
    # Ajustar la extensión del canvas
    canvas.setExtent(modified_extent)
    canvas.refresh()
    
    # Crear un diseño de impresión
    layout = QgsLayout(project)
    layout.initializeDefaults()
    
    
    map_item = QgsLayoutItemMap(layout)
    map_item.setRect(0, 0, tile_size, tile_size)
    map_item.setExtent(modified_extent)
    layout.addLayoutItem(map_item)
    
    # Exportar el diseño a imagen
    exporter = QgsLayoutExporter(layout)

    im_size = QgsLayoutExporter.ImageExportSettings().imageSize
    # QgsMessageLog.logMessage(f'image size: {im_size}', level=Qgis.Info)
    im_dpi = QgsLayoutExporter.ImageExportSettings().dpi
    # QgsMessageLog.logMessage(f'image dpi: {im_dpi}', level=Qgis.Info)
    im_cropToContents = QgsLayoutExporter.ImageExportSettings().cropToContents
    # QgsMessageLog.logMessage(f'image crop contents: {im_cropToContents}', level=Qgis.Info)
    
    im_settings = QgsLayoutExporter.ImageExportSettings()
    im_settings.cropToContents = True
    im_settings.size = QSize(256,256)
    exporter.exportToImage(output_file, im_settings)
    # im_settings.dpi = 300
    # exporter.exportToImage(output_file[:-4] + '.tif', im_settings) 
    time.sleep(0.3)
    if remove_layer_flag:
        project.removeMapLayer(raster_layer)
        raster_layer = None


# def move_extent(extent, long_unit, lat_unit):
def move_extent(extent):
    xmin = extent.xMinimum()
    ymin = extent.yMinimum()
    xmax = extent.xMaximum()
    ymax = extent.yMaximum()
    width = extent.width()
    height = extent.height()
    new_ymin = ymax
    new_ymax = ymax + height
    new_extent = QgsRectangle(xmin, new_ymin, xmax, new_ymax, normalize=False)
    return new_extent



def layoutbased_extract_raster_portion(input_path, output_path, extent, extent_crs_authid):
    # Crear un proyecto QGIS
    project = QgsProject.instance()

    # Cargar la capa raster
    raster_layer = QgsRasterLayer(input_path, "Sentinel-2 Band JP2")
    if not raster_layer.isValid():
        raise Exception(f"Failed to load raster layer: {input_path}")
    project.addMapLayer(raster_layer)

    # Definir el CRS del raster
    raster_crs = raster_layer.crs()

    # Verificar si el CRS de la extensión coincide con el CRS del raster
    extent_crs = QgsCoordinateReferenceSystem(extent_crs_authid)
    if extent_crs != raster_crs:
        # Transformar la extensión al CRS del raster
        coord_transform = QgsCoordinateTransform(extent_crs, raster_crs, project)
        min_x, min_y = coord_transform.transform(extent.xMinimum(), extent.yMinimum())
        max_x, max_y = coord_transform.transform(extent.xMaximum(), extent.yMaximum())
        transformed_extent = QgsRectangle(min_x, min_y, max_x, max_y)
    else:
        transformed_extent = extent

    # Crear un layout (diseño)
    layout = QgsLayout(project)
    layout.initializeDefaults()

    # Crear un elemento de mapa
    map_item = QgsLayoutItemMap(layout)
    map_item.setRect(QRectF(0, 0, 200, 200))
    map_item.setExtent(transformed_extent)
    map_item.setCrs(raster_crs)
    map_item.setFrameEnabled(True)

    # Añadir el elemento de mapa al layout
    layout.addLayoutItem(map_item)

    # Ajustar el tamaño del elemento de mapa para que coincida con la extensión deseada
    map_item.attemptResize(QgsLayoutSize(200, 200, QgsUnitTypes.LayoutMillimeters))

    # # Exportar el layout a un archivo de imagen
    # exporter = QgsLayoutExporter(layout)
    # image = QImage(QSize(800, 600), QImage.Format_ARGB32)
    # image.fill(1)
    # exporter.renderPage(image, 0)
    # image.save(output_path)

    # Exportar el diseño a imagen
    exporter = QgsLayoutExporter(layout)

    im_size = QgsLayoutExporter.ImageExportSettings().imageSize
    # QgsMessageLog.logMessage(f'image size: {im_size}', level=Qgis.Info)
    im_dpi = QgsLayoutExporter.ImageExportSettings().dpi
    # QgsMessageLog.logMessage(f'image dpi: {im_dpi}', level=Qgis.Info)
    im_cropToContents = QgsLayoutExporter.ImageExportSettings().cropToContents
    # QgsMessageLog.logMessage(f'image crop contents: {im_cropToContents}', level=Qgis.Info)
    
    im_settings = QgsLayoutExporter.ImageExportSettings()
    im_settings.cropToContents = True
    im_settings.size = QSize(256,256)
    exporter.exportToImage(output_path, im_settings)
    # im_settings.dpi = 300
    # exporter.exportToImage(output_file[:-4] + '.tif', im_settings) 

    project.removeMapLayer(raster_layer)


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def get_number_of_tiles_in_area(upper_left_coords, bottom_right_coords, zoom):
    ul_lat_deg, ul_lon_deg = upper_left_coords
    br_lat_deg, br_lon_deg = bottom_right_coords
    ul_xtile, ul_ytile = deg2num(ul_lat_deg, ul_lon_deg, zoom)
    br_xtile, br_ytile = deg2num(br_lat_deg, br_lon_deg, zoom)
    return int(np.abs(ul_xtile - br_xtile)), int(np.abs(ul_ytile - br_ytile))


#def tile_size_in_degrees(lat: float, lon: float, zoom: int) -> (float, float):
def tile_size_in_degrees(lat, lon, zoom):
    """
    Calculate the size in degrees of latitude and longitude for a given tile
    at a specified zoom level in Open Street Maps.
    
    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        zoom (int): Zoom level of the tile.
    
    Returns:
        tuple: Size in degrees of latitude and longitude of the tile (lat_deg, lon_deg).
    """
    # Number of tiles at the given zoom level
    n = 2 ** zoom
    # Size in degrees of one tile in longitude
    lon_deg = 360 / n
    # Convert latitude to radians
    lat_rad = math.radians(lat)
    # Calculate the size in degrees of one tile in latitude
    lat_deg = (360 / n) / math.cosh(lat_rad)
    return lat_deg, lon_deg



def transform_qgsrectangle(rect, src_crs, dst_crs):
    """
    Transform a QgsRectangle from the source CRS to the destination CRS.

    Args:
        rect (QgsRectangle): The rectangle to transform.
        src_crs (str): Source CRS in EPSG code (e.g., 'EPSG:3857').
        dst_crs (str): Destination CRS in EPSG code (e.g., 'EPSG:32630').

    Returns:
        QgsRectangle: The transformed rectangle.
    """
    # Create QgsCoordinateReferenceSystem objects for source and destination CRS
    source_crs = QgsCoordinateReferenceSystem(src_crs)
    destination_crs = QgsCoordinateReferenceSystem(dst_crs)

    # Create QgsCoordinateTransform object for transforming coordinates
    coordinate_transform = QgsCoordinateTransform(source_crs, destination_crs, QgsProject.instance())

    # Transform the rectangle coordinates
    transformed_rect = coordinate_transform.transformBoundingBox(rect)

    return transformed_rect




def estimate_num_tiles(patch_extent, tile_extent):
    # This function is independent of CRS, but both extents must be 
    # espressed in the same CRS/units
    num_width = int(tile_extent.width()/patch_extent.width())
    num_height = int(tile_extent.height()/patch_extent.height())
    print(f'num patches : # width: {num_width}  /  # height: {num_height}')



class SentinelTile:
    def __init__(self, id_str, folder, base_filename, valid_extent, extent_crs):
        self.id = id_str
        self.folder = folder
        self.base_filename = base_filename
        self.valid_extent = valid_extent
        self.tile_crs = extent_crs

    def get_num_patches_inside(self, patch_extent, patch_extent_crs):
        if self.tile_crs != patch_extent_crs:
            conv_patch_extent = transform_qgsrectangle(patch_extent, patch_extent_crs, self.tile_crs)
        else:
            conv_patch_extent = patch_extent
        num_width = int(self.valid_extent.width()/conv_patch_extent.width())
        num_height = int(self.valid_extent.height()/conv_patch_extent.height())
        print(f'num patches inside Tile {self.id} : # width: {num_width}  /  # height: {num_height}')
        return num_width, num_height
    
    def OSM_scan_area_inside_tile(self, example_patch_extent, patch_extent_crs, output_folder):
        # We sample all the tile surface and dump OpenStreetMap patches as small PNG files
        num_width, num_height = self.get_num_patches_inside(example_patch_extent, patch_extent_crs)
        if self.tile_crs != patch_extent_crs:
            conv_patch_extent = transform_qgsrectangle(example_patch_extent, patch_extent_crs, self.tile_crs)
        else:
            conv_patch_extent = example_patch_extent
        min_x_1st_patch = self.valid_extent.xMinimum()
        min_y_1st_patch = self.valid_extent.yMinimum()
        max_x_1st_patch = min_x_1st_patch + conv_patch_extent.width()
        max_y_1st_patch = min_y_1st_patch + conv_patch_extent.height()
        num_processed_patches = 0
        num_total_patches = num_width * num_height
        time.sleep(0.4)
        extent_reg = np.zeros((num_width, num_height,4))
        for x_i in range(num_width):
            for y_i in range(num_height):
                min_x_i_patch = min_x_1st_patch + conv_patch_extent.width() * x_i
                min_y_i_patch = min_y_1st_patch + conv_patch_extent.height() * y_i
                max_x_i_patch = max_x_1st_patch + conv_patch_extent.width() * x_i
                max_y_i_patch = max_y_1st_patch + conv_patch_extent.height() * y_i
                patch_i_extent = QgsRectangle(min_x_i_patch, min_y_i_patch, max_x_i_patch, max_y_i_patch)
                print(f' OSM patch x: {x_i}  y: {y_i}   -->   ({num_processed_patches} / {num_total_patches}) ')
                output_file_png = output_folder + f'patch_{self.id}_{x_i:03}_{y_i:03}_OSM.png'
                export_osm_image_by_extent(patch_i_extent, output_file_png)
                extent_reg[x_i, y_i, 0] = min_x_i_patch
                extent_reg[x_i, y_i, 1] = min_y_i_patch
                extent_reg[x_i, y_i, 2] = max_x_i_patch
                extent_reg[x_i, y_i, 3] = max_y_i_patch
                num_processed_patches = num_processed_patches + 1
                time.sleep(0.3)
        tile_crs_str = self.tile_crs.replace(':', '_')
        np.savez(f'./np_extent_{self.id}_{tile_crs_str}', extent_reg=extent_reg)

def warm_up(output_file_png):
    # Parámetros: longitud, latitud, zoom y archivo de salida
    longitude = -3.9303404 + 0.0003404
    latitude = 40.3070349
    zoom = 17 # en realidad es equivalente al nivel 15 en el navegador con nuestra corrección de /8 * 1.5 en tile_size
    QgsMessageLog.logMessage('--> Start, please wait...', level=Qgis.Info)

    # 1) Magical warm up
    if warmp_up_flag:
        raster_layer_list_to_clean = []
        for i in range(2):
            extent, extent_crs_authid, raster_layer = get_initial_extent(longitude, latitude, zoom, output_file_png, remove_layer_flag=False)
            raster_layer_list_to_clean.append(raster_layer)
            time.sleep(15)
        QgsMessageLog.logMessage('Warm up completed', level=Qgis.Info)
        # for raster_layer in raster_layer_list_to_clean:
        #     project = QgsProject.instance()
        #     project.removeMapLayer(raster_layer)
        # QgsMessageLog.logMessage('Cleaned OSM layers', level=Qgis.Info)
        # time.sleep(2)
    return extent, extent_crs_authid





# if __name__ == '__console__2': # or __name__ == '__main__'
#     warmp_up_flag = False # True
#     default_output_file_png = "C:/DMTA/projects/SD4EO/map-epiphany/output_file.png"
#     madrid_tile_EPSG32630 = QgsRectangle(400375, 4499727, 509067, 4391081)
#     madrid_tile_EPSG4326 = QgsRectangle(40.63831, -4.17545, 39.66683, -2.89456)
#     madrid_tile_EPSG3857 = transform_qgsrectangle(madrid_tile_EPSG32630, 'EPSG:32630', 'EPSG:3857')
    

#     if warmp_up_flag:
#         extent, extent_crs_authid  = warm_up(default_output_file_png)
#     else:
#         zoom = 14
#         # Mostoles coords
#         ul_coords = (40.3379,-3.8894)
#         br_coords = (40.3055,-3.8351)
#         # Additional initialization
#         new_lat = ul_coords[0]
#         new_lon = ul_coords[1]
#         extent, extent_crs_authid, _ = get_initial_extent(new_lon, new_lat, zoom, default_output_file_png) # Sólo para obtener el extent correcto
#         # Define a rectangle in EPSG:3857
#         rect_3857 = extent
#         # Transform the rectangle to EPSG:32630
#         rect_32630 = transform_qgsrectangle(rect_3857, 'EPSG:3857', 'EPSG:32630')
#         # estimate_num_tiles(rect_32630, madrid_tile_EPSG32630)
#         tile_madrid = SentinelTile('T30TVK', 'C:/DATASETs/_prueba/', 'T30TVK_20240417T105619_{band_name}_10m.jp2', madrid_tile_EPSG3857, 'EPSG:3857') #madrid_tile_EPSG32630, 'EPSG:32630')
#         tile_madrid.scan_area_inside_tile(rect_3857, 'EPSG:3857', 'C:/DMTA/projects/SD4EO/map-epiphany/check/')



if __name__ == '__console__': # or __name__ == '__main__'
    warmp_up_flag = False # True
    default_output_file_png = "C:/DMTA/projects/SD4EO/map-epiphany/output_file.png"
    # madrid_tile_EPSG32630 = QgsRectangle(400375, 4499727, 509067, 4391081)
    # madrid_tile_EPSG4326 = QgsRectangle(40.63831, -4.17545, 39.66683, -2.89456)
    # madrid_tile_EPSG3857 = transform_qgsrectangle(madrid_tile_EPSG32630, 'EPSG:32630', 'EPSG:3857')
    

    if warmp_up_flag:
        extent, extent_crs_authid  = warm_up(default_output_file_png)
    else:
        zoom = 14
        # Mostoles coords
        ul_coords = (40.3379,-3.8894)
        br_coords = (40.3055,-3.8351)
        # Additional initialization
        new_lat = ul_coords[0]
        new_lon = ul_coords[1]
        example_extent, extent_crs_authid, _ = get_initial_extent(new_lon, new_lat, zoom, default_output_file_png) # Sólo para obtener el extent correcto
        # Define a rectangle in EPSG:3857
        rect_3857 = example_extent
        # Transform the rectangle to EPSG:32630
        rect_32630 = transform_qgsrectangle(rect_3857, 'EPSG:3857', 'EPSG:32630')
        # estimate_num_tiles(rect_32630, madrid_tile_EPSG32630)
        for tile_id in available_Tiles_IDs[:1]:
            print(f'Processing tile : {tile_id}')
            output_OSM_png_folder = f'C:/DATASETs/UC2Training/OSM/{tile_id}/'
            icoords_EPGS3857, icoords_EPGS4326, base_filename = provide_inner_tile_coords(tile_id)
            tile_i = SentinelTile(tile_id, f'C:/DATASETs/UC2Training/Sentinel2/{tile_id}/', base_filename, QgsRectangle(*icoords_EPGS3857), 'EPSG:3857') #madrid_tile_EPSG32630, 'EPSG:32630')
            tile_i.OSM_scan_area_inside_tile(rect_3857, 'EPSG:3857', output_OSM_png_folder)


