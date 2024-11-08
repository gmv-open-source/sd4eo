# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     common_tile_utils.py
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
# Comon data for the S2 Tiles

available_Tiles_IDs = ['T30TVK', 'T30TXT', 'T30TYS', 'T30UWU', 'T30UXU', 'T30UYU', 'T31TCJ', 'T31TCL', 'T31TCN', 'T31TDL', 'T31TDM', 'T31TDN', 'T31TEN', 'T31TFM', 'T31TFN', 'T31UDP', 'T31UDQ', 'T31UEP', 'T31UEQ', 'T31UFP', 'T31UFQ'] # 'T31TEM'

available_cities_IDs = ['Madrid', 'Paris', 'Toulouse', 'Poitiers', 'Bordeaux', 'Limoges', 'Clermont-Ferrand', 'Troyes', 'Le Mans', 'Angers',] # 'Rouen']

def provide_inner_tile_coords(tile_id):
    # python 3.9 does not support match statements
    # so, I use chained "if-a-gogo"  :-/
    icoords_EPSG3857 = (0,0,0,0)
    icoords_EPSG4326 = (0.0, 0.0, 0.0, 0.0)
    base_filename = ''
    orig_tile_name = tile_id
    # Full Tiles
    if tile_id == 'T30TVK': # Madrid
        icoords_EPGS3857 = (-461082.0,4954666.0, -324885.8,4820817.3)
        icoords_EPGS4326 = (40.607702,-4.141970, 39.688782,-2.918383)
        base_filename = 'T30TVK_20240417T105619_{band_name}_10m.jp2'
    elif tile_id == 'T30TXT':
        icoords_EPGS3857 = (-174073.7,6070449.5, -32283.2,5923028.2)
        icoords_EPGS4326 = (46.883329,-0.289922, 47.780624,-1.563854)
        base_filename = 'T30TXT_20231010T105859_{band_name}_10m.jp2'
    elif tile_id == 'T30TYS':
        icoords_EPGS3857 = (-35098.6,5924209.3, 106330.3,5770471.7)
        icoords_EPGS4326 = (45.938271,0.955078, 46.890299,-0.315378)
        base_filename = 'T30TYS_20231010T105859_{band_name}_10m.jp2'
    elif tile_id == 'T30UWU':
        icoords_EPGS3857 = (-325008.5,6192979.8, -173940.9,6075140.7)
        icoords_EPGS4326 = (47.809086,-1.562723, 48.515120,-2.919622)
        base_filename = 'T30UWU_20231107T111251_{band_name}_10m.jp2'
    elif tile_id == 'T30UXU':
        icoords_EPGS3857 = (-175483.7,6220817.8, -29138.2,6070575.4)
        icoords_EPGS4326 = (47.7814669,-0.2617938, 48.680422,-1.575492)
        base_filename = 'T30UXU_20231010T105859_{band_name}_10m.jp2'
    elif tile_id == 'T30UYU':
        icoords_EPGS3857 = (-24825.8,6219362.1, 119308.6,6063851.7)
        icoords_EPGS4326 = (47.740726,1.071932, 48.671760,-0.222644)
        base_filename = 'T30UYU_20231010T105859_{band_name}_10m.jp2'
    elif tile_id == 'T31TCJ':  # Toulouse
        icoords_EPGS3857 = (61252.0,5497148.5, 205551.2,5356484.2)
        icoords_EPGS4326 = (43.291894,1.846744, 44.204501,0.547398)
        base_filename = 'T31TCJ_20210415T105021_{band_name}_10m.jp2'
    elif tile_id == 'T31TCL':
        icoords_EPGS3857 = (55131.3,5778552.8, 199737.3,5635240.1)
        icoords_EPGS4326 = (45.087061,1.794188, 45.988503,0.495171)
        base_filename = 'T31TCL_20240509T105031_{band_name}_10m.jp2'
    elif tile_id == 'T31TCN':
        icoords_EPGS3857 = (65021.8,6074829.4, 194523.1,5926396.4)
        icoords_EPGS4326 = (46.903949,1.747513, 47.807015,0.584019)
        base_filename = 'T31TCN_20210425T105021_{band_name}_10m.jp2'
    elif tile_id == 'T31TDL':
        icoords_EPGS3857 = (200517.8,5781220.5, 340124.6,5638860.1)
        icoords_EPGS4326 = (45.112294,3.049264, 46.005280,1.799143)
        base_filename = 'T31TDL_20240509T105031_{band_name}_10m.jp2'
    elif tile_id == 'T31TDM':
        icoords_EPGS3857 = (198315.8,5901497.3, 342898.8,5779341.3)
        icoords_EPGS4326 = (45.993853,3.080230, 46.750913,1.781213)
        base_filename = 'T31TDM_20210331T104619_{band_name}_10m.jp2'
    elif tile_id == 'T31TDN':
        icoords_EPGS3857 = (195129.6,6071018.3, 339773.3,5926623.0)
        icoords_EPGS4326 = (46.905045,3.052379, 47.784154,1.752715)
        base_filename = 'T31TDN_20210425T105021_{band_name}_10m.jp2'
    elif tile_id == 'T31TEM':
        icoords_EPGS3857 = (194883.6,6074678.3, 338949.3,5927172.4)
        icoords_EPGS4326 = (46.908655,3.044936, 47.806172,1.750751)
        base_filename = 'T31TEM_20210427T103619_{band_name}_10m.jp2'
    elif tile_id == 'T31TEN':
        icoords_EPGS3857 = (339030.6,6045961.9, 490911.4,5924292.9)
        icoords_EPGS4326 = (46.890939,4.410304, 47.632653,3.045407)
        base_filename = 'T31TEN_20210427T103619_{band_name}_10m.jp2'
    elif tile_id == 'T31TFM':
        icoords_EPGS3857 = (490580.0,5921317.2, 604662.9,5781901.3)
        icoords_EPGS4326 = (46.009617,5.431544, 46.872722,4.407034)
        base_filename = 'T31TFM_20210427T103619_{band_name}_10m.jp2'
    elif tile_id == 'T31TFN':
        icoords_EPGS3857 = (489939.1,6072027.4, 630929.0,5927174.7)
        icoords_EPGS4326 = (46.908641,5.667770, 47.790126,4.401041)
        base_filename = 'T31TFN_20240511T103629_{band_name}_10m.jp2'
    elif tile_id == 'T31UDP':
        icoords_EPGS3857 = (190908.0,6223561.6, 338716.7,6077418.4)
        icoords_EPGS4326 = (47.823125,3.042509, 48.696722,1.714917)
        base_filename = 'T31UDP_20210425T105021_{band_name}_10m.jp2'
    elif tile_id == 'T31UDQ':  # Paris
        icoords_EPGS3857 = (190642.0,6374725.2, 341165.8,6228680.1)
        icoords_EPGS4326 = (48.7271200,3.0646176, 49.585110,1.712449)
        base_filename = 'T31UDQ_20231007T104829_{band_name}_10m.jp2'
    elif tile_id == 'T31UEP':
        icoords_EPGS3857 = (342037.8,6223448.3, 491313.6,6076845.1)
        icoords_EPGS4326 = (47.819233,4.413604, 48.695972,3.072891)
        base_filename = 'T31UEP_20210425T105021_{band_name}_10m.jp2'
    elif tile_id == 'T31UEQ':
        icoords_EPGS3857 = (344877.2,6377603.8, 493538.2,6232951.7)
        icoords_EPGS4326 = (48.752316,4.433608, 49.601795,3.097928)
        base_filename = 'T31UEQ_20231007T104829_{band_name}_10m.jp2'
    elif tile_id == 'T31UFP':
        icoords_EPGS3857 = (492306.6,6223896.2, 638421.4,6070574.5)
        icoords_EPGS4326 = (47.781475,5.734920, 48.698654,4.422269)
        base_filename = 'T31UFP_20240511T103629_{band_name}_10m.jp2'
    elif tile_id == 'T31UFQ':
        icoords_EPGS3857 = (494244.5,6371336.5, 644302.9,6222178.4)
        icoords_EPGS4326 = (48.688559,5.787813, 49.565319,4.440031)
        base_filename = 'T31UFQ_20240511T103629_{band_name}_10m.jp2'
    # Cities:
    elif tile_id == 'Poitiers': # Poitiers
        icoords_EPGS3857 = (31242.22,5879868.90, 45680.11,5868271.74 )
        icoords_EPGS4326 = (46.6175362,0.2806514, 46.5459456,0.4103712)
        base_filename = 'T30TYS_20231010T105859_{band_name}_10m.jp2'
        orig_tile_name = 'T30TYS'
    elif tile_id == 'Toulouse':  # Toulouse
        icoords_EPGS3857 = (142103.48,5416061.56, 175921.16,5387986.96)
        icoords_EPGS4326 = (43.6800475,1.2765196, 43.4973656,1.5803443)
        base_filename = 'T31TCJ_20210415T105021_{band_name}_10m.jp2'
        orig_tile_name = 'T31TCJ'
    elif tile_id == 'Paris':  # Paris
        icoords_EPGS3857 = (240313.73,6277032.84, 293387.34,6221346.24)
        icoords_EPGS4326 = (49.0135827,2.1596661, 48.6835930,2.6356139)
        base_filename = 'T31UDQ_20231007T104829_{band_name}_10m.jp2'
        orig_tile_name = 'T31UDQ'
    elif tile_id == 'Madrid': # Madrid
        icoords_EPGS3857 = (-415074.6,4940758.7, -398942.40,4917240.28)
        icoords_EPGS4326 = (40.513335,-3.724938, 40.3520871,-3.5837782)
        base_filename = 'T30TVK_20240417T105619_{band_name}_10m.jp2'
        orig_tile_name = 'T30TVK'
    elif tile_id == 'Móstoles': # Móstoles (only for testing purposes)
        icoords_EPGS3857 = (-431779.2,4915038.3, -427896.01,4910448.26)
        icoords_EPGS4326 = (40.337004,-3.878668, -427896.01,4910448.26)
        base_filename = 'T30TVK_20240417T105619_{band_name}_10m.jp2'        
        orig_tile_name = 'T30TVK'
    elif tile_id == 'Bordeaux':
        icoords_EPGS3857 = (-84433.72,5611675.11, -46689.97,5585108.27)
        icoords_EPGS4326 = (44.9373920,-0.7584645, 44.7682464,-0.4194397)
        base_filename = ''        
        orig_tile_name = ''        
    elif tile_id == 'Limoges':
        icoords_EPGS3857 = (130700.6,5761148.9, 149222.18,5744217.71)
        icoords_EPGS4326 = (45.880079,1.174136, 45.7740488,1.3404320)
        base_filename = ''        
        orig_tile_name = '' 
    elif tile_id == 'Clermont-Ferrand':
        icoords_EPGS3857 = (335221.6,5763191.5, 361040.55,5724531.98)
        icoords_EPGS4326 = (45.892840,3.011429, 45.6505857,3.2432700)
        base_filename = ''        
        orig_tile_name = '' 
    elif tile_id == 'Troyes':
        icoords_EPGS3857 = (445690.2,6162661.7, 464852.1,6145274.3)
        icoords_EPGS4326 = (48.334371,4.003720, 48.230397,4.175903)
        base_filename = ''        
        orig_tile_name = '' 
    elif tile_id == 'Nancy':
        icoords_EPGS3857 = (679646.9,6228627.7, 693934.4,6214279.5)
        icoords_EPGS4326 = (48.726777,6.105240, 48.641605,6.233619)
        base_filename = ''        
        orig_tile_name = ''         
    elif tile_id == 'Le Mans':
        icoords_EPGS3857 = (14679.3,6114213.9, 31500.1,6097582.6)
        icoords_EPGS4326 = (48.044249,0.131817, 47.943925,0.278310)
        base_filename = ''        
        orig_tile_name = '' 
    elif tile_id == 'Angers':
        icoords_EPGS3857 = (-72076.6,6030395.1, -49771.5,6012328.8)
        icoords_EPGS4326 = (47.538360,-0.647641, 47.428671,-0.447171)
        base_filename = ''        
        orig_tile_name = '' 
    # elif tile_id == 'Rouen':
    #     icoords_EPGS3857 = (48.230397,4.175903, 132022.9,6332356.2)
    #     icoords_EPGS4326 = (49.490801,1.015178, 49.337714,1.184957)
    #     base_filename = ''        
    #     orig_tile_name = ''         
    else:
        print(f'ERROR: Tile not recognized: >>{tile_id}<<')
    return icoords_EPGS3857, icoords_EPGS4326, base_filename





