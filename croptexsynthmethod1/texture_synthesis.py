# © GMV Soluciones Globales Internet, S.A.U. [2024]
# File Name:     texture_synthesis.py
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
# This is an adaptation to multichannel support based on 
#     textureSynth/textureColorSynthesis.m by J. Portilla and E. Simoncelli.
#     http://www.cns.nyu.edu/~lcv/texture/

#     Differences:
#     (1) Only real doain steerable pyramid support (not complex vsalues).
#     (2) I don't use filter masks of orientations in the process of coarse to fine.

#     TODO: Update *obsolete* usage via command line parameters:
#     python texture_synthesis.py -i pebbles.jpg -o tmp -n 5 -k 4 -m 7 --iter 100

#     -i : input image path
#     -o : path for output
#     -n : depth of steerable pyramid (default:5)
#     -k : num of orientations of steerable pyramid (default:4)
#     -n : pixel distance for calculating auto-correlations (default:7)
#     --iter : number of iterations (default:100)

import argparse, copy
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from PIL import Image
import sys, os
from scipy.stats import skew, kurtosis
import time

import sutils
import steerable_pyramid as steerable
import texture_analysis as ta
# from multichannelFileFormatMS1 import show_mcArray, load_NC_patch, save_new_NC_patch, bands_IDs
from multichannelFileFormat import load_from_netCDF, save_as_netCDF, show_mcArray, MultiChannelFileMetadata, year_month_list, Romania_Crop_Codes, ALL_BAND_CODE, S_BAND_INDX

SCRIPT_NAME = os.path.basename(__file__)

# logging
LOG_FMT = "[%(name)s] %(asctime)s %(levelname)s %(lineno)s %(message)s"
logging.basicConfig(level=logging.CRITICAL, format=LOG_FMT)
LOGGER = logging.getLogger(os.path.basename(__file__))


def apply_band_weights(image, band_weights, flag_reverse=False):
    epsilon = 0.001
    new_image = copy.deepcopy(image)
    for i, band_w_i in enumerate(band_weights):
        if band_w_i < (1.0+epsilon) and band_w_i > (1.0-epsilon):
            continue
        else:
            weight = band_w_i
            if flag_reverse:
                weight = 1.0/band_w_i
                print(f'{i} : {band_w_i}')
            new_image[:,:,i] = new_image[:,:,i]*weight
    return new_image

'''
    Texture Synthesis by Portilla-Simoncelli's algorithm heavily modified by DMTA

'''
def synthesis(image, resol_x, resol_y, num_channels, num_depth, num_ori, num_neighbor, iter, band_weights):
    # # Scale per bands
    # image = apply_band_weights(image, band_weights)
    # analyse original image
    orig_data = ta.TextureAnalysis(image, resol_x, resol_y, num_channels, num_depth, num_ori, num_neighbor)
    orig_data.analyse()

    # I try to ensure we have a different random seed each time we call the algorithm
    np.random.seed(None)

    # initialize random image
    im = np.random.normal(0, 1, resol_x * resol_y * num_channels).reshape(resol_y*resol_x, num_channels)
# test
#	im = np.loadtxt("init-im.csv",delimiter=",")
    ## adjust covariance among RGB channels
    _tmp = orig_data.COV_RGB
    im = sutils.adjust_corr1(im, _tmp)
    im = im + orig_data.MEAN_RGB
    im = im.reshape(resol_y, resol_x, num_channels)

    # eigen values and vectors of original image
    ocov_eval, ocov_evec = np.linalg.eig(orig_data.COV_RGB)
    _idx = np.argsort(ocov_eval)[::-1]
    ocov_ediag = np.diag(ocov_eval[_idx])
    ocov_evec = ocov_evec[:, _idx]

    ## this treatment is to get same result as Matlab
    for k in range(ocov_evec.shape[1]):
        if np.sum(ocov_evec[:,k] < 0) > np.sum(ocov_evec[:,k] >= 0):
            ocov_evec[:,k] = -1. * ocov_evec[:,k]
    ## [Attn.] Bellow (1/4 power) may be mistake of textureColorAnalysis.m/textureColorSynthesis.m.
    ## **(0.5) would be right. this obstructs color reproduction.
#	ocov_iediag = np.linalg.pinv(ocov_ediag**(0.25))
    # Moore-Penrose Pseudo Inverse.
    ocov_iediag = np.linalg.pinv(ocov_ediag**(0.5))
    
    # iteration
    prev_im = np.array([])
    prev_dst = 0.

    for it in range(0, iter):
        LOGGER.debug('iteration {}'.format(str(it)))

        # ------------------------------------
        # Normalized pca components
        _dim = im.shape
        im = im.reshape(_dim[0]*_dim[1], num_channels)
        _mean = np.mean(im, axis=0)
        im = im - _mean
        ## get principal components
        _pcscore = np.dot(im, ocov_evec)
        ## normalize principal components 
        im = np.dot(_pcscore, ocov_iediag)
        im = im.reshape(_dim[0], _dim[1], num_channels)
    
        pyr_l = []
        lr_l = []

        # ------------------------------------
        # Create pyramids of each PCA channel
        for clr in range(num_channels):
            # steerable pyramid
            _sp = steerable.SteerablePyramid(im[:, :, clr], resol_x, resol_y, num_depth, num_ori, '', '', 0)
            _sp.create_pyramids()

            # subtract means from lowpass residuals
            _sp.LR['s'] = _sp.LR['s'].real - np.mean(_sp.LR['s'].real.flatten())

            pyr_l.append(copy.deepcopy(_sp))
            lr_l.append(_sp.LR)

        # ------------------------------------
        # Adjust lowpass residual and get initial image for coarse to fine
        ## get auto-correlations (2 slide)
        ## this tric is according to textureSynthesis.m
        _mat = sutils.get_2slide(lr_l, num_channels)

        ## adjust auto correlation of lowpass residuals
        _mat = sutils.adjust_corr1(_mat, orig_data.COV_LR)

        ## back to lowpass residuals
        _dim = tuple(map(lambda x: x * 2, _sp.LR['s'].shape))
        for clr in range(num_channels):
            _tns = np.zeros((_dim[0], _dim[1], 5))
            _tns[:, :, 0] = _mat[:, 0 + 5*clr].reshape(_dim[0], _dim[1])
            _tns[:, :, 1] = np.roll(_mat[:, 1 + 5*clr].reshape(_dim[0], _dim[1]), -2, axis=1)
            _tns[:, :, 2] = np.roll(_mat[:, 2 + 5*clr].reshape(_dim[0], _dim[1]), 2, axis=1)
            _tns[:, :, 3] = np.roll(_mat[:, 3 + 5*clr].reshape(_dim[0], _dim[1]), -2, axis=0)
            _tns[:, :, 4] = np.roll(_mat[:, 4 + 5*clr].reshape(_dim[0], _dim[1]), 2, axis=0)
            _mean = np.mean(_tns, axis=2)
            _mean = sutils.shrink(_mean, 2) * 4.
            lr_l[clr]['s'] = _mean
            lr_l[clr]['f'] = np.fft.fftshift(np.fft.fft2(_mean))
            pyr_l[clr].LR['s'] = lr_l[clr]['s']
            pyr_l[clr].LR['f'] = lr_l[clr]['f']

        ## get initial data for coarse to fine
        rec_im = []
        for clr in range(num_channels):
            # get lowband
            _z = np.zeros_like(lr_l[clr]['f'])
            _s = steerable.SteerablePyramid(_z, _z.shape[1], _z.shape[0], 1, num_ori, '', '', 0)
            _lr_f = lr_l[clr]['f'] * _s.L0_FILT
            _lr_s = np.fft.ifft2(np.fft.ifftshift(_lr_f)).real
            # modify central auto correlation
            if(orig_data.LR_MAR[clr][1]/ocov_ediag[clr,clr] > 1.0e-3):
                try:
                    lr_l[clr]['s'] = sutils.mod_acorr(_lr_s, orig_data.LR_CA[clr], num_neighbor)
                except LinAlgError as e:
                    LOGGER.info('LinAlgError {}'.format(e))
            else:
                lr_l[clr]['s'] = lr_l[clr]['s'] * np.sqrt(orig_data.LR_MAR[clr][1] / np.var(lr_l[clr]['s']))

            lr_l[clr]['s'] = lr_l[clr]['s'].real
            # modify skewness of lowpass residual
            lr_l[clr]['s'] = sutils.mod_skew(lr_l[clr]['s'], orig_data.LR_MAR[clr][2])
            # modify kurtosis of lowpass residual
            lr_l[clr]['s'] = sutils.mod_kurt(lr_l[clr]['s'], orig_data.LR_MAR[clr][3])
            lr_l[clr]['f'] = np.fft.fftshift(np.fft.fft2(lr_l[clr]['s']))

             # initial coarse to fine
            rec_im.append(lr_l[clr]['s'])

        ## get original statistics of bandpass signals.
        bnd = []
        bnd_m = []
        bnd_p = []
        bnd_rp = []
        bnd_ip = []
        for clr in range(num_channels):
            # create parents
            bnd.append(copy.deepcopy(pyr_l[clr].BND))
            _b_m, _, _ = sutils.trans_b(pyr_l[clr].BND)
            for i in range(len(_b_m)):
                for k in range(len(_b_m[i])):
                    _b_m[i][k] -= np.mean(_b_m[i][k])
            ## magnitude
            bnd_m.append(_b_m)

            _b_p, _b_rp, _b_ip = sutils.get_parent(pyr_l[clr].BND, pyr_l[clr].LR)
            ## maginitude of parent bandpass  (this is 'parent' in textureColorAnalysis.m)
            bnd_p.append(_b_p)
            ## real values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
            bnd_rp.append(_b_rp)
            ## imaginary values of parent bandpass (this is half of 'rparent' in textureColorAnalysis.m)
            bnd_ip.append(_b_ip)

        # ------------------------------------
        # Coarse to fine adjustment
        for dp in range(num_depth-1, -1, -1):

            # combine colors
            cousins = sutils.cclr_b(bnd_m, dp, num_channels)
            rparents = sutils.cclr_rp(bnd_rp, bnd_ip, dp, num_channels)

            # adjust covariances
            _prev = cousins
            if dp < num_depth-1:
                parents = sutils.cclr_p(bnd_p, dp, num_channels)
                cousins = sutils.adjust_corr2(_prev, orig_data.CF_COUS[dp], parents, orig_data.CF_CPAR[dp])
                if np.isnan(cousins).any():
                    LOGGER.info('NaN in adjust_corr2')
                    cousins = sutils.adjust_corr1(_prev, orig_data.CF_COUS[dp])
            else:
                cousins = sutils.adjust_corr1(_prev, orig_data.CF_COUS[dp])

            # separate colors
            cousins = sutils.sclr_b(cousins, num_ori, num_channels)

            # adjust central auto corr. and update bandpass.
            bnd_r = []
            for clr in range(num_channels):
                _list = []
                for k in range(num_ori):
                    # adjust central auto-correlations
                    _tmp = sutils.mod_acorr(cousins[clr][k], orig_data.BND_MCOR[clr][dp][k], num_neighbor)
                    # update BND_N
                    bnd_m[clr][dp][k] = _tmp
                    _mean = orig_data.BND_MMAR[clr][dp][k][0]
                    _tmp = _tmp + _mean
                    _idx = np.where(_tmp < 0)
                    _tmp[_idx] = 0

                    _bnd = pyr_l[clr].BND[dp][k]['s']
                    _idx1 = np.where(np.abs(_bnd) < 10**(-12))
                    _idx2 = np.where(np.abs(_bnd) >= 10**(-12))
                    _bnd[_idx1] = _bnd[_idx1] * _tmp[_idx1]
                    _bnd[_idx2] = _bnd[_idx2] * _tmp[_idx2] / np.abs(_bnd[_idx2])

                    _list.append(_bnd.real)

                bnd_r.append(_list)

            # combine colors & make rcousins
            rcousins = sutils.cclr_bc(bnd_r, dp, num_channels)

            # adjust cross-correlation of real values of B and real/imaginary values of parents
            _prev = rcousins
            try:
                rcousins = sutils.adjust_corr2(_prev, orig_data.CF_RCOU[dp], rparents, orig_data.CF_RPAR[dp])
                if np.isnan(rcousins).any():
                    LOGGER.info('NaN in adjust_corr2')
                    rcousins = sutils.adjust_corr1(_prev, orig_data.CF_RCOU[dp])
                    if np.isnan(rcousins).any():
                        LOGGER.info('NaN in adjust_corr1')
                        rcousins = _prev
            except LinAlgError as e:
                LOGGER.info('LinAlgError {}'.format(e))
                rcousins = sutils.adjust_corr1(_prev, orig_data.CF_RCOU[dp])
                if np.isnan(rcousins).any():
                    LOGGER.info('NaN in adjust_corr1')
                    rcousins = _prev

            # separate colors
            rcousins = sutils.sclr_b(rcousins, num_ori, num_channels)
            for clr in range(num_channels):
                for k in range(num_ori):
                    ## update pyramid
                    pyr_l[clr].BND[dp][k]['s'] = rcousins[clr][k]
                    pyr_l[clr].BND[dp][k]['f'] = np.fft.fftshift(np.fft.fft2(rcousins[clr][k]))

            # combine bands
            _rc = copy.deepcopy(rcousins)
            for clr in range(num_channels):
                # same size
                _z = np.zeros_like(_rc[clr][0])
                _s = steerable.SteerablePyramid(_z, _z.shape[1], _z.shape[0], 1, num_ori, '', '', 0)

                _recon = np.zeros_like(_z)
                for k in range(num_ori):
                    ## modify angle: not good
#					amask = cr_mask(_s.AT[0], k, num_ori)
#					_recon = _recon + pyr_l[clr].BND[dp][k]['f'] * amask * _s.B_FILT[0][k]

                    _recon = _recon + pyr_l[clr].BND[dp][k]['f'] * _s.B_FILT[0][k]
                
                _recon = _recon * _s.L0_FILT
                _recon = np.fft.ifft2(np.fft.ifftshift(_recon)).real

                # expand image created before and sum up
                _im = rec_im[clr]
                _im = sutils.expand(_im, 2).real / 4.
                _im = _im.real + _recon

                # adjust auto-correlation
                try:
                    _im = sutils.mod_acorr(_im, orig_data.CF_CA[clr][dp], num_neighbor)
                except LinAlgError as e:
                    LOGGER.info('Pass. LinAlgError {}'.format(e))

                # modify skewness
                _im = sutils.mod_skew(_im, orig_data.CF_MAR[clr][dp][2])

                # modify kurtosis
                _im = sutils.mod_kurt(_im, orig_data.CF_MAR[clr][dp][3])

                rec_im[clr] = _im

        # end of coarse to fine

        # ------------------------------------
        # Adjustment variance in H0 and final adjustment of coarse to fine.
        for clr in range(num_channels):
            _tmp = pyr_l[clr].H0['s'].real
            _var = np.var(_tmp)
            _tmp = _tmp * np.sqrt(orig_data.H0_PRO[clr] / _var)

            # recon H0
            _recon = np.fft.fftshift(np.fft.fft2(_tmp))
            _recon = _recon * _s.H0_FILT
            _recon = np.fft.ifft2(np.fft.ifftshift(_recon)).real

            ## this is final data of coarse to fine.
            rec_im[clr] = rec_im[clr] + _recon

            # adjust auto correlations
            rec_im[clr] = sutils.mod_acorr(rec_im[clr], orig_data.PCA_CA[clr], num_neighbor)

            # adjust skewness and kurtosis to original.
            _mean = np.mean(rec_im[clr])
            _var = np.var(rec_im[clr])
            rec_im[clr] = ( rec_im[clr] - _mean ) / np.sqrt(_var)
            ## skewness
            rec_im[clr] = sutils.mod_skew(rec_im[clr], orig_data.MS_PCA[clr][2])
            ## kurtosis
            rec_im[clr] = sutils.mod_kurt(rec_im[clr], orig_data.MS_PCA[clr][3])


        # ------------------------------------
        # Back to RBG channels and impose desired statistics
        _dim = im.shape
        im = im.reshape(_dim[0]*_dim[1], num_channels)
        for clr in range(num_channels):
            im[:, clr] = rec_im[clr].flatten()

        _mean = np.mean(im, axis=0)

        im = im - _mean
        im = sutils.adjust_corr1(im, np.eye(num_channels))
        ## [Attn.] Bellow (1/4 power) may be mistake of textureColorAnalysis.m/textureColorSynthesis.m.
        ## **(0.5) would be right. this obstructs color reproduction.
#		im = np.dot(im, np.dot(ocov_ediag**(0.25), ocov_evec.T))
        im = np.dot(im, np.dot(ocov_ediag**(0.5), ocov_evec.T))
        
        _mean = np.zeros(num_channels)
        for i in range(num_channels):
            _mean[i] = orig_data.MS_RGB[i][0]
        print(_mean.shape)
        print(im.shape)
        
        im += _mean

        im = im.reshape(_dim[0], _dim[1], num_channels)

        # ------------------------------------
        # Adjust pixel statistic of RBG channels.
        for clr in range(num_channels):
            # modify mean and variance of image created
            _mean = np.mean(im[:, :, clr])
            _var = np.var(im[:, :, clr])
            im[:, :, clr] = im[:, :, clr] - _mean
            im[:, :, clr] = im[:, :, clr] * np.sqrt(orig_data.RGB_MAR[clr][1] / _var)
            im[:, :, clr] = im[:, :, clr] + orig_data.MS_RGB[clr][0]
            # modify skewness of image created
            im[:, :, clr] = sutils.mod_skew(im[:, :, clr], orig_data.RGB_MAR[clr][2])
            # modify kurtsis of image created
            im[:, :, clr] = sutils.mod_kurt(im[:, :, clr], orig_data.RGB_MAR[clr][3])
            # adjust range
            _tmp = im[:, :, clr].reshape(_dim[0], _dim[1])
            _idx  = np.where(_tmp > orig_data.RGB_MAR[clr][4])
            _tmp[_idx] = orig_data.RGB_MAR[clr][4]
            im[:, :, clr] = _tmp
            _idx  = np.where(_tmp < orig_data.RGB_MAR[clr][5])
            _tmp[_idx] = orig_data.RGB_MAR[clr][5]
            im[:, :, clr] = _tmp

        # # ------------------------------------
        # # Save image
        # # bugfix
        # _o_img = Image.fromarray(np.uint8(im[:,:,:3])) # We only dump the 3 fisrt channels
		# #_o_img = Image.fromarray(np.uint8(im)).convert('L')
        # # TODO: update output
        # _o_img.save(out_path + '/out-n{}-k{}-m{}-{}.png'.format(str(num_depth), str(num_ori), str(num_neighbor), str(it)))

        if it > 0:
            dst = np.sqrt(np.sum((prev_im - im)**2))
            rt = np.sqrt(np.sum((prev_im - im)**2)) / np.sqrt(np.sum(prev_im**2))
            LOGGER.debug('change {}, ratio {}'.format(str(dst), str(rt)))

            if it > 1:
                thr = np.abs(np.abs(prev_dst) - np.abs(dst)) / np.abs(prev_dst)
                LOGGER.debug('threshold {}'.format(str(thr)))
                if thr < 1e-6:
                    break

            prev_dst = dst

        prev_im = im

    # Revert scale per bands
    # im = apply_band_weights(im, band_weights, flag_reverse=True)
    return im

'''
    make mask

    this is not good.

'''
#def cr_mask(angle, k, num_ori):
#	at = angle
#	th1, th2 = at, at
#
#	amask = np.zeros_like(at)
#	th1[np.where(at - k*np.pi/num_ori < -np.pi)] += 2.*np.pi
#	th1[np.where(at - k*np.pi/num_ori > np.pi)] -= 2.*np.pi
#	_ind = np.where(np.absolute(th1 - k*np.pi/num_ori) < np.pi/2.)
#	amask[_ind] = 2.
#	_ind = np.where(np.absolute(th1 - k*np.pi/num_ori) == np.pi/2.)
#	amask[_ind] = 1.
#	th2[np.where(at + (num_ori-k)*np.pi/4. < -np.pi)] += 2.*np.pi
#	th2[np.where(at + (num_ori-k)*np.pi/4. > np.pi)] -= 2.*np.pi
#	_ind = np.where(np.absolute(th2 + (num_ori-k) * np.pi/num_ori) < np.pi/2.)
#	amask[_ind] = 2.
#	_ind = np.where(np.absolute(th2 + (num_ori-k) * np.pi/num_ori) == np.pi/2.)
#	amask[_ind] = 1.
#
#	amask[int(amask.shape[0]/2), int(amask.shape[1]/2)] = 1.
#	amask[0, 0] = 1.
#	amask[0, amask.shape[1]-1] = 1.
#	amask[amask.shape[0]-1, 0] = 1.
#	amask[amask.shape[0]-1, amask.shape[1]-1] = 1.
#
#	return amask



if __name__ == "__main__2":
    LOGGER.info('script start')
    
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Texture Synthesis (Color Version) by Portilla and Simoncelli')
    parser.add_argument('--orig_img', '-i', default='pebbles.jpg',
                    help='Original image')
    parser.add_argument('--out_dir', '-o', default='tmp',
                    help='Output directory')
    parser.add_argument('--num_depth', '-n', default=5, type=int,
                    help='depth of steerable pyramid')
    parser.add_argument('--num_ori', '-k', default=4, type=int,
                    help='orientation of steerable pyramid')
    parser.add_argument('--num_neighbor', '-m', default=7, type=int,
                    help='local neighborhood')
    parser.add_argument('--iter', default=100, type=int,
                    help='number of iterations')

    args = parser.parse_args()

    ## validation of num. of neighbours.
    ms = [3, 5, 7, 9, 11, 13]
    if not args.num_neighbor in ms:
            LOGGER.error('illegal number of orientation: {}'.format(str(args.num_neighbor)))
            raise ValueError('illegal number of orientation: {}'.format(str(args.num_neighbor)))


    im = np.array(Image.open(args.orig_img))  # 3 channel imagen
    num_channels = 3
    synthesis(im, im.shape[1], im.shape[0], num_channels, args.num_depth, args.num_ori, args.num_neighbor, args.iter, args.out_dir)


if __name__ == "__main__3":
    LOGGER.info('script start')
    
    start_time = time.time()
    num_channels = 3
    num_depth = 5
    num_ori  = 4
    num_neighbor = 7
    iter = 10
    out_dir = 'out'
    input_filename = './samples/alfalfa_crop_Brawley_256.jpg'
    im = np.array(Image.open(input_filename))  # 3 channel imagen
    synthesis(im, im.shape[1], im.shape[0], num_channels, num_depth, num_ori, num_neighbor, iter, out_dir)


if __name__ == "__main__4":
    LOGGER.info('script start')
    LOGGER.propagate = False  # deactivate the logging output in stdout
    LOGGER.disabled = True    # deactivate the logging output in stdout

    start_time = time.time()
    num_channels = 7
    num_depth = 5
    num_ori  = 4
    num_neighbor = 7
    iter = 10
    out_path = 'out'
    composed_input_im = np.zeros((256,256,num_channels))
    coord_ini = 1295
    for i in range(num_channels):
        input_filename = './samples/SD4EO_Spain_Crops_20171009_B'+str(i+2).zfill(2)+'_10m.tif'
        monochannel_im = np.array(Image.open(input_filename))  # only  8 bits-depth
        cropped_monochannel_im = monochannel_im[coord_ini:coord_ini+256,coord_ini:coord_ini+256]
        min_value = np.min(np.min(cropped_monochannel_im))
        max_value = np.max(np.max(cropped_monochannel_im))
        print(f"layer {i}    min: {min_value}   max: {max_value} ")
        composed_input_im[:,:,i] = cropped_monochannel_im * 1.0/25 # TODO: Hay un problema con el rango dinámico en las imágenes TIFF
    #plt.imshow(composed_input_im)
    #plt.show()
    check_img = Image.fromarray(np.uint8(composed_input_im[:,:,:3])) # We only dump the 3 first channels
    check_img.save('./' + out_path + '/base.png')

    synthesis(composed_input_im, composed_input_im.shape[1], composed_input_im.shape[0], num_channels, num_depth, num_ori, num_neighbor, iter, out_path)



# def old_script_for_MS1():
#     LOGGER.info('script start')
#     LOGGER.propagate = False  # deactivate the logging output in stdout
#     LOGGER.disabled = True    # deactivate the logging output in stdout

#     crop_type = 'alfalfa'
#     crop_type = 'durumwheat'
#     crop_type = 'corn'
#     crop_type = 'sugarbeets'
#     start_time = time.time() 
#     num_channels = 7
#     num_depth = 5
#     num_ori  = 4
#     num_neighbor = 7
#     iter = 12
#     input_path = './extended_parcels/'
#     output_path = './synthetic_parcels/'
#     for polygon_i in range(5):
#         for month_i in range (12):
#             input_full_filename = input_path + f"quilted_{crop_type}_{polygon_i}_{month_i}.nc"
#             extended_mc_array, max_value, max_visible_value = load_NC_patch(input_full_filename, 0, flag_show=False)
#             mc_array = extended_mc_array[:256,:256,:9] #* 1.0/max_value
#             num_channels = mc_array.shape[2]
#             for var_i in range(8):
#                 output_full_filename = output_path + f"synthetic_{crop_type}_{polygon_i}_{month_i}_{var_i}.nc"
#                 try:
#                     mc_synthetic = synthesis(mc_array, mc_array.shape[1], mc_array.shape[0], num_channels, num_depth, num_ori, num_neighbor, iter, output_path)
#                     # show_mcArray(mc_synthetic, max_visible_value)
#                     save_new_NC_patch(mc_synthetic, output_full_filename, max_visible_value)
#                 except BaseException as e:
#                     print(f"Se ha capturado una excepción de nivel base: {e}")

def find_incomplete_file_path(input_path):
    """
    Finds and returns the full path of a file within a specified directory, based on an input string
    that combines the directory path and the starting characters of the file name.

    Args:
        input_path (str): The combined directory path and file start characters.

    Returns:
        str: The full path of the matching file. If no file is found, returns an empty string.

    """
    print('Inicio')
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    # Determine the last occurrence of the path separator to split directory and file start
    separator_index = input_path.rfind(os.path.sep)
    
    # If the separator is not found, return an empty string (invalid input)
    if separator_index == -1:
        separator_index = input_path.rfind('/')
        if separator_index == -1:
            return ''

    # Extract the directory and file start from the input path
    directory = input_path[:separator_index]
    file_start = input_path[separator_index + 1:]
    
    # Construct the sea1rch pattern to include the directory, the starting characters, and any extension
    search_pattern = os.path.join(directory, file_start + '*')

    # Use glob to find files that match the search pattern
    matching_files = glob.glob(search_pattern)

    # Return the first matching file's full path if any, else return an empty string
    return matching_files[0] if matching_files else ''


if __name__ == "__main__":
    LOGGER.info('script start')
    LOGGER.propagate = False  # deactivate the logging output in stdout
    LOGGER.disabled = True    # deactivate the logging output in stdout
    start_time = time.time() 
    normalization_mode = True
    num_depth = 4
    num_ori  = 6 #4
    num_neighbor = 14 #7
    iter = 12   # 12
    crop_type_indx = 8

    # for crop_type_indx in (4,):
    crop_type_name = Romania_Crop_Codes[crop_type_indx][0]
    for month_i in range(12):
        date_str = year_month_list[month_i]

        input_incomplete_filename = f'C:/DMTA/projects/SD4EO/chaotic-aggregation/CyL_and_Cat/{date_str}_{crop_type_name}_'
        output_incomplete_filename = f'C:/DATASETs/OUT_HighOrder/HO_{crop_type_name}_{date_str}_'
        # input_incomplete_filename = f'../assembledpuzzles.CyLplusCat/{date_str}_{crop_type_name}_'
        # output_incomplete_filename = f'../OUT_HighOrder/HO_{crop_type_name}_{date_str}_'
        # # output_incomplete_filename = f'../MISC/HO_{crop_type_name}_{date_str}_'

        input_full_filename = find_incomplete_file_path(input_incomplete_filename)

        extended_mc_array, max_value, max_visible_value, metadata = load_from_netCDF(input_full_filename, 0, flag_normalization=normalization_mode, flag_show=False)
        # we suppose the input reference image is power-of-2 decomposable size at least at 2**num_depth
        side_size = extended_mc_array.shape[0]
        num_channels = extended_mc_array.shape[2]
        if side_size > 1024: # For now we'll consider only a max of 1024x1024xXi texture
            new_size = int(side_size/2)
            print(f"We reduce the size of the sample from {side_size} to {new_size}")
            new_mc_array = np.zeros((new_size,new_size,num_channels))
            new_mc_array[:new_size,:new_size,:] = extended_mc_array[:new_size,:new_size,:] + 0.0 # Forced deep copy 
            extended_mc_array = new_mc_array # to free memory
            side_size = new_size
        for var_i in range(6):  # 6 is a trade-off
            print(f"******* NEW IMAGE {crop_type_name} - {date_str} - {var_i}")
            output_full_filename = output_incomplete_filename + f"{side_size}_{var_i}.nc"
            try:
                # band_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0) # 11 bands
                band_weights = (1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 3.0) # 11 bands
                # Scale per bands
                extended_mc_array_m = apply_band_weights(extended_mc_array, band_weights)
                mc_synthetic = synthesis(extended_mc_array_m, extended_mc_array_m.shape[1],extended_mc_array_m.shape[0], num_channels, num_depth, num_ori, num_neighbor, iter, band_weights)
                mc_synthetic = apply_band_weights(mc_synthetic, band_weights, flag_reverse=True)
                # show_mcArray(mc_synthetic, max_visible_value)
                save_as_netCDF(mc_synthetic, output_full_filename, crop_type_name, date_str, metadata.dataset, 'HighOrder', max_visible_value, metadata, flag_revert_normalization=normalization_mode)
            except BaseException as e:
                print(f"Se ha capturado una excepción de nivel base: {e}")


    
