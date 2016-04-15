'''
Neil Zimmerman
Space Telescope Science Institute

Core toolkit for post-processing coronagraph testbed data

created spring 2016
'''

import numpy as np
import time as time
import astropy.io.fits as pyfits
import warnings
from scipy.ndimage.interpolation import *
from scipy.interpolate import *
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import nanmean, nanmedian
from scipy.io.idl import readsav
from shutil import copyfile
import multiprocessing
import sys
import os
import pdb
import cPickle as pickle


def get_radius_sqrd(shape, c=None):
    if c is None:
        c = (0.5*float(shape[1] - 1),  0.5*float(shape[0] - 1))
    y, x = np.indices(shape)
    rsqrd = (x - c[0])**2 + (y - c[1])**2
    return rsqrd

def get_angle(s, c=None):
    if c is None:
        c = (0.5*float(s[1] - 1),  0.5*float(s[0] - 1))
    y, x = np.indices(s)
    theta = np.arctan2(y - c[1], x - c[0])
    # Change the angle range from [-pi, pi] to [0, 360]
    theta_360 = np.where(np.greater_equal(theta, 0), np.rad2deg(theta), np.rad2deg(theta + 2*np.pi))
    return theta_360

#def get_KL_basis(R, K):
#    w, V = np.linalg.eig(np.dot(R, np.transpose(R)))
#    sort_ind = np.argsort(w)[::-1] #indices of eigenvals sorted in descending order
#    sv = np.sqrt(w[sort_ind][w[sort_ind] > 0]).reshape(-1,1) #column of ranked singular values
#    Z = np.dot(1./sv*np.transpose(V[:, sort_ind[w[sort_ind] > 0]]), R)
#    N_modes = min([K, Z.shape[0]])
#    return Z[0:N_modes, :], sv, N_modes

def get_trunc_KL_basis(R, K):
    U, sv, Vt = np.linalg.svd(R, full_matrices=False)
    N_modes = min([K, Vt.shape[0]])
    return Vt[:N_modes, :], sv, N_modes

def get_full_KL_basis(R):
    U, sv, Vt = np.linalg.svd(R, full_matrices=False)
    N_modes = Vt.shape[0]
    return Vt, sv, N_modes

def superpose_srcmodel(data_img, srcmodel_img, srcmodel_destxy, srcmodel_centxy = None, rolloff_rad = None):
    assert(len(data_img.shape) == 2)
    assert(len(srcmodel_img.shape) == 2)
    assert( srcmodel_destxy[0] < data_img.shape[1] and srcmodel_destxy[1] < data_img.shape[0]\
            and min(srcmodel_destxy) >= 0 )
    if srcmodel_centxy == None:
        srcmodel_centxy = ((srcmodel_img.shape[0] - 1.)/2., (srcmodel_img.shape[1] - 1.)/2.)
    subpix_xyoffset = np.array( [(srcmodel_destxy[0] - srcmodel_centxy[0])%1.,\
                                 (srcmodel_destxy[1] - srcmodel_centxy[1])%1.] )
    if abs(round(srcmodel_centxy[0] + subpix_xyoffset[0]) - round(srcmodel_centxy[0])) == 1: # jump in pixel containing x center
        subpix_xyoffset[0] -= round(srcmodel_centxy[0] + subpix_xyoffset[0]) - round(srcmodel_centxy[0])
    if abs(round(srcmodel_centxy[1] + subpix_xyoffset[1]) - round(srcmodel_centxy[1])) == 1: # jump in pixel containing y center
        subpix_xyoffset[1] -= round(srcmodel_centxy[1] + subpix_xyoffset[1]) - round(srcmodel_centxy[1])
    #print "subpix_offset: ", subpix_xyoffset

    if rolloff_rad:
        Y, X = np.indices(srcmodel_img.shape)
        Rsqrd = (X - srcmodel_centxy[0])**2 + (Y - srcmodel_centxy[1])**2
        rolloff_arr = np.exp( -(Rsqrd / rolloff_rad**2)**2 )
        srcmodel_img *= rolloff_arr
    shifted_srcmodel_img = shift(input = srcmodel_img, shift = subpix_xyoffset[::-1], order=3)

    srcmodel_BLcorneryx = np.array( [round(srcmodel_destxy[1]) - round(srcmodel_centxy[1]),
                                     round(srcmodel_destxy[0]) - round(srcmodel_centxy[0])], dtype=np.int)
    srcmodel_TRcorneryx = srcmodel_BLcorneryx + np.array(srcmodel_img.shape)
    super_BLcorneryx = np.amax(np.vstack((srcmodel_BLcorneryx, np.zeros(2))), axis=0)
    super_TRcorneryx = np.amin(np.vstack((srcmodel_TRcorneryx, np.array(data_img.shape))), axis=0)
    BLcropyx = super_BLcorneryx - srcmodel_BLcorneryx
    TRcropyx = srcmodel_TRcorneryx - super_TRcorneryx
    super_img = data_img.copy()
    super_img[super_BLcorneryx[0]:super_TRcorneryx[0],\
              super_BLcorneryx[1]:super_TRcorneryx[1]] +=\
            shifted_srcmodel_img[BLcropyx[0]:srcmodel_img.shape[0]-TRcropyx[0],\
                                 BLcropyx[1]:srcmodel_img.shape[1]-TRcropyx[1]]
    return super_img 

def rdi_klipsub(R, T, Kcut): 
    # perform KLIP RDI subtraction with Kcut modes
    # R is the reference vector array; N_ref_frame rows and N_pix columns
    # T is the target vector array; 1 row and N_pix columns

    # Subtract spatial mean from each image vector
    R_smean = np.tile(np.reshape(np.mean(R, axis=1),(R.shape[0],1)), (1, R.shape[1]))
    Rs = R - R_smean
    T_smean = np.tile(np.mean(T, axis=1), (1, T.shape[1]))
    Ts = T - T_smean

    # Get K-L basis
    Z, sv, K = get_trunc_KL_basis(Rs, Kcut)
    # Project target and subtract
    S = Ts - Ts.dot(Z.T).dot(Z)

    return S

def rdi_klipsub_thrupt(R, T, P, Kcut):
    # perform KLIP RDI subtraction with Kcut modes
    # R is the reference vector array; N_ref_frame rows and N_pix columns
    # T is the target vector array; 1 row and N_pix columns
    # Determine the point source throughput based on the projection of an off-axis
    # PSF model onto the K-L basis

    # Subtract spatial mean from each image vector
    R_smean = np.tile(np.reshape(np.mean(R, axis=1),(R.shape[0],1)), (1, R.shape[1]))
    Rs = R - R_smean
    T_smean = np.tile(np.mean(T, axis=1), (1, T.shape[1]))
    Ts = T - T_smean

    # Get K-L basis
    Z, sv, K = get_trunc_KL_basis(Rs, Kcut)
    # Project target and subtract
    S = Ts - Ts.dot(Z.T).dot(Z)

    klip_thrupt = np.empty(P.shape[:-1])
    if len(klip_thrupt.shape) == 2:
        for si in range(klip_thrupt.shape[0]):
            for ti in range(klip_thrupt.shape[1]):
                P_smean = np.tile(np.mean(P[si,ti]), (1, P[si,ti].shape[0]))
                Ps = P[si,ti] - P_smean
                klip_thrupt[si, ti] = 1. - Ps.dot(Z.T).dot(Z).dot(Ps.T) / Ps.dot(Ps.T)
    elif len(klip_thrupt.shape) == 1:
        for ti in range(klip_thrupt.shape[0]):
            P_smean = np.tile(np.mean(P[ti]), (1, P[ti].shape[0]))
            Ps = P[ti] - P_smean
            klip_thrupt[ti] = 1. - (Ps - Ps.dot(Z.T).dot(Z)).dot(Ps.T) / Ps.dot(Ps.T)

    return S, klip_thrupt

def rel_coron_thrupt(Pmod, ref_pos):
    # Given 2-d off-axis PSF model over a set of positions,
    # comput the FWHM throughput relative to the off-axis PSF
    # at a reference (presumably peak throughput) location.
    coron_thrupt = np.empty(Pmod.shape[:-2])

    Pref = Pmod[ref_pos]
    ref_peak = np.max(Pref)
    ref_fwhm_ind = np.greater_equal(Pref, ref_peak/2)
    ref_fwhm_sum = np.sum(Pref[ref_fwhm_ind])

    if len(coron_thrupt.shape) == 2:
        for si in range(coron_thrupt.shape[0]):
            for ti in range(coron_thrupt.shape[1]):
                P = Pmod[si,ti]
                fwhm_ind = np.greater_equal(P, np.max(P)/2)
                fwhm_sum = np.sum(P[fwhm_ind])
                coron_thrupt[si,ti] = fwhm_sum/ref_fwhm_sum

    elif len(coron_thrupt.shape) == 1:
        for ti in range(coron_thrupt.shape[0]):
            P = Pmod[ti]
            fwhm_ind = np.greater_equal(P, np.max(P)/2)
            fwhm_sum = np.sum(P[fwhm_ind])
            coron_thrupt[ti] = fwhm_sum/ref_fwhm_sum

    return coron_thrupt

def radial_contrast_flr(image, xc, yc, seps, zw, coron_thrupt, klip_thrupt=None):
    rad_flr_ctc = np.empty((len(seps)))
    assert(len(seps) == len(coron_thrupt))
    if klip_thrupt is not None:
        assert(len(seps) == len(klip_thrupt))
        rad_flr_ctc_ktc = np.empty((len(seps)))
    else:
        rad_flr_ctc_ktc = None

    imh = image.shape[0]
    imw = image.shape[1]

    xs = np.arange(imw) - xc
    ys = np.arange(imh) - yc
    XXs, YYs = np.meshgrid(xs, ys)
    RRs = np.sqrt(XXs**2 + YYs**2)

    for si, sep in enumerate(seps):
        r_in = np.max([seps[0], sep-zw/2.])
        r_out = np.min([seps[-1], sep+zw/2.])
        meas_ann_mask = np.logical_and(np.greater_equal(RRs, r_in),
                                          np.less_equal(RRs, r_out))
        meas_ann_ind = np.nonzero(np.logical_and(np.greater_equal(RRs, r_in).ravel(),
                                                    np.less_equal(RRs, r_out).ravel()))[0]
        meas_ann = np.ravel(image)[meas_ann_ind]
        rad_flr_ctc[si] = np.nanstd(meas_ann)/coron_thrupt[si]
        if rad_flr_ctc_ktc is not None:
            rad_flr_ctc_ktc[si] = np.nanstd(meas_ann)/coron_thrupt[si]/klip_thrupt[si]

    #pdb.set_trace()
    return rad_flr_ctc, rad_flr_ctc_ktc
