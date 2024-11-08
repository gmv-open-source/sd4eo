#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     image_quilting.py
# Author:        David Miraut
# License:       MIT License
# 
# The following script is the result of mixing and adapting several works and source code pieces, 
# some of them coded in matlab, to python code, in order to achieve the purpose described
# below. Here are the references to those works:
# https://github.com/rohitrango/Image-Quilting-for-Texture-Synthesis
# https://github.com/PJunhyuk/ImageQuilting
# https://github.com/TetsuyaOdaka/texture-synthesis-portilla-simoncelli
# https://github.com/ganguli-lab/textureSynth
# https://github.com/plenoptic-org/plenoptic

# All of them are licensed by the terms of the MIT License

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
# Image Quilting algorithm as described by Alexei Efros in [1-4].
# We add the multi-channel support for an arbitrary number of channels + large dynamic range
# As it is implemented in python, it takes several minutes to compute each new image.

# [1] Efros, A. A., & Freeman, W. T. (2023). Image quilting for texture synthesis and transfer. 
#     In Seminal Graphics Papers: Pushing the Boundaries, Volume 2 (pp. 571-576).
# [2] Efros, A. A., & Freeman, W. T. (2001). Image quilting for texture synthesis and transfer. 
#     SIGGRAPH '01: Proceedings of the 28th annual conference on Computer graphics and interactive
#     techniques. August 2001. Pages 341-346
#     https://doi.org/10.1145/383259.383296
# [3] https://www.youtube.com/watch?v=QMiCNJofJUk
# [4] https://people.eecs.berkeley.edu/~efros/research/quilting.html


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math
from skimage import io, util
import heapq
from multichannelFileFormatMS1 import show_mcArray, load_NC_patch, save_new_NC_patch, bands_IDs


def randomPatch(texture, patchLength):
    h, w, _ = texture.shape
    i = np.random.randint(h - patchLength)
    j = np.random.randint(w - patchLength)

    return texture[i:i+patchLength, j:j+patchLength]


def L2OverlapDiff(patch, patchLength, overlap, res, y, x):
    error = 0

    if x > 0:
        left = patch[:, :overlap] - res[y:y+patchLength, x:x+overlap]
        error += np.sum(left**2)

    if y > 0:
        up   = patch[:overlap, :] - res[y:y+overlap, x:x+patchLength]
        error += np.sum(up**2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
        error -= np.sum(corner**2)

    return error
 

def randomBestPatch(texture, patchLength, overlap, res, y, x):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i+patchLength, j:j+patchLength]
            # print(patch.shape)
            e = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            errors[i, j] = e

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+patchLength, j:j+patchLength]



def minCutPath(errors):
    # dijkstra's algorithm vertical
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)

    h, w = errors.shape
    seen = set()

    while pq:
        error, path = heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[-1]

        if curDepth == h:
            return path

        for delta in -1, 0, 1:
            nextIndex = curIndex + delta

            if 0 <= nextIndex < w:
                if (curDepth, nextIndex) not in seen:
                    cumError = error + errors[curDepth, nextIndex]
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))


def minCutPath2(errors):
    # dynamic programming, unused
    errors = np.pad(errors, [(0, 0), (1, 1)], 
                    mode='constant', 
                    constant_values=np.inf)

    cumError = errors[0].copy()
    paths = np.zeros_like(errors, dtype=int)    

    for i in range(1, len(errors)):
        M = cumError
        L = np.roll(M, 1)
        R = np.roll(M, -1)

        # optimize with np.choose?
        cumError = np.min((L, M, R), axis=0) + errors[i]
        paths[i] = np.argmin((L, M, R), axis=0)
    
    paths -= 1
    
    minCutPath = [np.argmin(cumError)]
    for i in reversed(range(1, len(errors))):
        minCutPath.append(minCutPath[-1] + paths[i][minCutPath[-1]])
    
    return map(lambda x: x - 1, reversed(minCutPath))


def minCutPatch(patch, patchLength, overlap, res, y, x):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
        leftL2 = np.sum(left**2, axis=2)
        for i, j in enumerate(minCutPath(leftL2)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
        upL2 = np.sum(up**2, axis=2)
        for j, i in enumerate(minCutPath(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)

    return patch


def quilt(texture, patchLength, numPatches, mode="cut", sequence=False):
    texture = util.img_as_float(texture)

    overlap = patchLength // 6
    numPatchesHigh, numPatchesWide = numPatches

    h = (numPatchesHigh * patchLength) - (numPatchesHigh - 1) * overlap
    w = (numPatchesWide * patchLength) - (numPatchesWide - 1) * overlap

    res = np.zeros((h, w, texture.shape[2]))  # Here, it takes all channels

    print(res.shape)
    print(texture.shape)
    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "random":
                patch = randomPatch(texture, patchLength)
            elif mode == "best":
                patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
            elif mode == "cut":
                patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
                patch = minCutPatch(patch, patchLength, overlap, res, y, x)
            
            res[y:y+patchLength, x:x+patchLength] = patch

            if sequence:
                io.imshow(res)
                io.show()
      
    return res


def quiltSize(texture, patchLength, shape, mode="cut"):
    overlap = patchLength // 6
    h, w = shape

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    res = quilt(texture, patchLength, (numPatchesHigh, numPatchesWide), mode)

    return res[:h, :w]



if __name__ == "__main__":
    crop_margin = 3
    # crop_type = 'alfalfa'
    crop_type = 'durumwheat'
    # crop_type = 'corn'
    # crop_type = 'sugarbeets'
    num_pieces = 5
    num_months = 12
    for piece_i in range(num_pieces):
        for month_i in range(num_months):
            print(f"parcel # {piece_i}   month # {month_i}")
            full_input_filename = f"./original_parcels/orig_{crop_type}_{piece_i}_{month_i}.nc"
            full_output_filename = f"./extended_parcels/quilted_{crop_type}_{piece_i}_{month_i}.nc"
            mc_array, max_value, max_visible_value = load_NC_patch(full_input_filename, crop_margin, flag_show=False)

            # I try to ensure we have a different random seed each tiem we call the algorithm
            np.random.seed(None)

            patch_size = 33 # 31
            num_patchs = (9,9) #(10,10) #(20,20) #(6, 6)
            # new_mc_texture_random = quilt(mc_array, patch_size, num_patchs, "random")
            # # show_mcArray(new_mc_texture_random, max_visible_value)

            # new_mc_texture_best = quilt(mc_array, patch_size, num_patchs, "best")
            # # show_mcArray(new_mc_texture_best, max_visible_value)

            new_mc_texture_cut = quilt(mc_array, patch_size, num_patchs, "cut")
            # show_mcArray(new_mc_texture_cut, max_visible_value)
            save_new_NC_patch(new_mc_texture_cut, full_output_filename, max_visible_value)
            pass

if __name__ == "__main__2":
    # Multichannel proof of concept
    test_img_filename1 = 'alfalfa_crop_Brawley_256.jpg'
    test_img_filename2 = 'crop5.jpg'

    texture1 = io.imread('./samples/' + test_img_filename1) 
    print(texture1.shape)
    texture2 = io.imread('./samples/' + test_img_filename2) 
    print(texture2.shape)
    
    texture_N = np.zeros((texture1.shape[0], texture1.shape[1], texture1.shape[2]*2))
    texture_N[:,:,:3] = texture1*1.0/256
    texture_N[:,:,3:] = texture2*1.0/256

    # I try to ensure we have a different random seed each tiem we call the algorithm
    np.random.seed(None)

    patch_size = 25
    num_patchs = (10,10) #(20,20) #(6, 6)
    new_texture_N_random = quilt(texture_N, patch_size, num_patchs, "random")
    io.imshow(new_texture_N_random[:,:,:3])
    io.show()

    new_texture_N_best = quilt(texture_N, patch_size, num_patchs, "best")
    io.imshow(new_texture_N_best[:,:,:3])
    io.show()

    new_texture_N_cut = quilt(texture_N, patch_size, num_patchs, "cut")
    io.imshow(new_texture_N_cut[:,:,:3])
    io.show()



if __name__ == "__main__2":
    test_img_filename = 'test_quilting.png'
    # test_img_filename = 'alfalfa_crop_Brawley_256.jpg'
    # test_img_filename = 'crop5.jpg'

    texture = io.imread('./samples/' + test_img_filename) 
    print(texture.shape)
    io.imshow(texture)
    io.show()

    # I try to ensure we have a different random seed each tiem we call the algorithm
    np.random.seed(None)

    patch_size = 25
    num_patchs = (20,20) #(6, 6)
    io.imshow(quilt(texture, patch_size, num_patchs, "random"))
    io.show()

    io.imshow(quilt(texture, patch_size, num_patchs, "best"))
    io.show()

    io.imshow(quilt(texture, patch_size, num_patchs, "cut"))
    io.show()