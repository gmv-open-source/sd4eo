#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     texture_transfer.py
# Author:        David Miraut
# License:       MIT License
# 
# The following script is the result of mixing and adapting several works, specially the one made by Tetsuya Odaka, 
# and source code pieces, some of them coded in matlab, to python code, in order to achieve the purpose described
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
# Texture transfer algorithm as described by Alexei Efros in [1-4].
# We add the multi-channel support for an arbitrary number of channels + large dynamic range

# [1] Efros, A. A., & Freeman, W. T. (2023). Image quilting for texture synthesis and transfer. 
#     In Seminal Graphics Papers: Pushing the Boundaries, Volume 2 (pp. 571-576).
# [2] Efros, A. A., & Freeman, W. T. (2001). Image quilting for texture synthesis and transfer. 
#     SIGGRAPH '01: Proceedings of the 28th annual conference on Computer graphics and interactive
#     techniques. August 2001. Pages 341-346
#     https://doi.org/10.1145/383259.383296
# [3] https://www.youtube.com/watch?v=QMiCNJofJUk
# [4] https://people.eecs.berkeley.edu/~efros/research/quilting.html


import numpy as np
from matplotlib import pyplot as plt
import math
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage import io, util
import heapq
from image_quilting import L2OverlapDiff, minCutPatch

def bestCorrPatch(texture, corrTexture, patchLength, corrTarget, y, x):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    corrTargetPatch = corrTarget[y:y+patchLength, x:x+patchLength]
    curPatchHeight, curPatchWidth = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            corrTexturePatch = corrTexture[i:i+curPatchHeight, j:j+curPatchWidth]
            e = corrTexturePatch - corrTargetPatch
            errors[i, j] = np.sum(e**2)

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+curPatchHeight, j:j+curPatchWidth]

def bestCorrOverlapPatch(texture, corrTexture, patchLength, overlap, 
                         corrTarget, res, y, x, alpha=0.1, level=0):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    corrTargetPatch = corrTarget[y:y+patchLength, x:x+patchLength]
    di, dj = corrTargetPatch.shape

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i+di, j:j+dj]
            l2error = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            overlapError = np.sum(l2error)

            corrTexturePatch = corrTexture[i:i+di, j:j+dj]
            corrError = np.sum((corrTexturePatch - corrTargetPatch)**2)

            prevError = 0
            if level > 0:
                prevError = patch[overlap:, overlap:] - res[y+overlap:y+patchLength, x+overlap:x+patchLength]
                prevError = np.sum(prevError**2)
            
            errors[i, j] = alpha * (overlapError + prevError) + (1 - alpha) * corrError

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i+di, j:j+dj]



def transfer(texture, target, patchLength, mode="cut", 
             alpha=0.1, level=0, prior=None, blur=False):
    corrTexture = rgb2gray(texture)
    corrTarget  = rgb2gray(target)

    if blur:
        corrTexture = gaussian(corrTexture, sigma=3)
        corrTarget  = gaussian(corrTarget,  sigma=3)

    io.imshow(corrTexture)
    io.show()
    io.imshow(corrTarget)
    io.show()

    # remove alpha channel
    texture = util.img_as_float(texture)[:,:,:3]
    target = util.img_as_float(target)[:,:,:3]

    h, w, _ = target.shape
    overlap = patchLength // 6

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    
    if level == 0:
        res = np.zeros_like(target)
    else:
        res = prior

    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "best":
                patch = bestCorrPatch(texture, corrTexture, patchLength, corrTarget, y, x)
            elif mode == "overlap":
                patch = bestCorrOverlapPatch(texture, corrTexture, patchLength, 
                                             overlap, corrTarget, res, y, x)
            elif mode == "cut":
                patch = bestCorrOverlapPatch(texture, corrTexture, patchLength, 
                                             overlap, corrTarget, res, y, x, 
                                             alpha, level)
                patch = minCutPatch(patch, patchLength, overlap, res, y, x)
            
            res[y:y+patchLength, x:x+patchLength] = patch
      
    return res

def transferIter(texture, target, patchLength, n):
    res = transfer(texture, target, patchLength)
    for i in range(1, n):
        print(f"Iter {i} / {n}")
        alpha = 0.1 + 0.8 * i / (n - 1)
        patchLength = patchLength * 2**i // 3**i
        print((alpha, patchLength))
        res = transfer(texture, target, patchLength, 
                       alpha=alpha, level=i, prior=res)
    
    return res


## Loss function

def Loss_function(original, syn):
  height, width, depth = original.shape
  for i in range(height):
      loss3 += np.sqrt(np.sum(np.square(original[i][:,0:3]/np.max(original) - syn[i]/np.max(syn))))


if __name__ == "__main__":
    bill = io.imread('./samples/bill-big_quilting.jpg')
    rice = io.imread('./samples/rice_quilting.gif')
    drawing = io.imread('./samples/drawing_quilting.png')
    man = io.imread('./samples/man_quilting.png')

    print(bill.shape)
    plt.imshow(bill)
    plt.show()

    print(rice.shape)
    plt.imshow(np.squeeze(rice))
    plt.show()

    res2 = transferIter(np.squeeze(rice), bill, 20, 2)
    io.imshow(res2)
    io.show()

    # io.imsave("ricebill2.png", res2) #It fails, probably because of dynamic range

    res3 = transfer(drawing, man, 20, blur=True)
    io.imshow(res3)
    io.show()
    io.imsave("drawingman.png", res3)

