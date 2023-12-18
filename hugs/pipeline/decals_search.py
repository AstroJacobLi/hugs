"""
Modified based on `next_gen_search.py` to run on DECaLS data.
"""

from __future__ import division, print_function

import os
import numpy as np
from astropy.table import Table, hstack
from lsst.pipe.base import Struct
from ..sep_stepper import SepLsstStepper, sep_ellipse_mask
from ..stats import get_clipped_sig_task
from ..utils import pixscale, zpt
from .. import utils
from .. import imtools
from .. import primitives as prim
from ..cattools import xmatch, xmatch_re
from ..star_mask import scott_star_mask

import copy
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.math as afwMath
import lsst.afw.display as afwDisplay
import lsst.geom as geom
import sep

__all__ = ['run']


def run(cfg, reset_mask_planes=False):
    """
    Run hugs pipeline using SExtractor for the final detection 
    and photometry.

    Parameters
    ----------
    cfg : hugs_pipe.Config 
        Configuration object which stores all params 
        as well as the exposure object. 

    Returns
    -------
    results : lsst.pipe.base.Struct
        Object containing results:
        results.all_detections : catalog of all detections
        results.sources : catalog of sources we are keeping
        results.exp : exposure object for this run
        results.exp_clean : cleaned exposure object for this run
        results.success : boolean flag of run status 
    """

    assert cfg.brick, 'No brick is given!'
    cfg.timer  # start timer

    try:
        ############################################################
        # Generate star mask
        ############################################################
        cfg.logger.info('generating star mask')
        star_mask = scott_star_mask(exposure=cfg.exp[cfg.band_detect], 
                                    p1=0.8, p2=0.8, bright_thresh=13)
        mi = cfg.exp[cfg.band_detect].getMaskedImage()
        mask = mi.getMask()
        # Set star mask to the MASK
        binary_mask_image = afwImage.ImageF(mi.getDimensions())
        binary_mask_image.array[:] = star_mask.astype(float)
        threshold = afwDet.Threshold(0.1)
        footprint_set = afwDet.FootprintSet(binary_mask_image, threshold)
        mask.addMaskPlane('BRIGHT_OBJECT')
        footprint_set.setMask(mask, 'BRIGHT_OBJECT')
        mask.removeAndClearMaskPlane('DETECTED')
        
        ############################################################
        # Get masked image and check if we have enough good data
        ############################################################
        if cfg.band_mask != cfg.band_detect:
            cfg.logger.info('making mask using {}-band'.format(cfg.band_mask))
            mi_band_mask = cfg.exp[cfg.band_mask].getMaskedImage().clone()
        else:
            mi_band_mask = mi
        stat_task = get_clipped_sig_task()

        cfg.exp.patch_meta.good_data_frac = cfg.exp.good_data_fraction(cfg.band_detect)
        
        cfg.logger.info('good data fraction = {:.2f}'.
                        format(cfg.exp.patch_meta.good_data_frac))

        if cfg.exp.patch_meta.good_data_frac < cfg.min_good_data_frac:
            msg = '***** not enough data in {} {} {}-band!!! ****'
            cfg.logger.warning(msg.format(cfg.tract, cfg.patch, cfg.band_mask))
            results = _null_return(cfg)
            return results

        if cfg.band_mask != cfg.band_detect:
            min_good = cfg.min_good_data_frac
            if cfg.exp.good_data_fraction(cfg.band_mask) < min_good:
                msg = '***** not enough data in {} {} {}-band!!! ****'
                cfg.logger.warning(
                    msg.format(cfg.tract, cfg.patch, cfg.band_mask))
                results = _null_return(cfg)
                return results

        ############################################################
        # Check data quality in both bands. If the r-band is deeper,
        # we increase the smoothing scale.
        ############################################################
        depth_ratio = stat_task.run(cfg.exp['g'].getMaskedImage()).stdev / stat_task.run(cfg.exp['r'].getMaskedImage()).stdev
        cfg.logger.info(f'depth ratio r / g = {depth_ratio}')
        if depth_ratio > 1: # when r-band is deeper
            cfg.psf_sigma['r'] *= 1.5
            cfg.psf_sigma['g'] *= 1.5

        ############################################################
        # Image thesholding at low and high thresholds. In both
        # cases, the image is smoothed at the psf scale.
        ############################################################
        # mi_smooth = imtools.smooth_gauss(mi_band_mask, cfg.psf_sigma)
        mi_smooth = mi_band_mask # we don't smooth here
        stats = stat_task.run(mi_smooth)

        if stats.stdev <= 0.0 or np.isnan(stats.stdev):
            msg = '***** {} | {} -- stddev = {} !!! ****'
            cfg.logger.warning(msg.format(cfg.tract, cfg.patch, stats.stdev))
            results = _null_return(cfg)
            return results

        if cfg.thresh_type.lower() == 'sb':
            cfg.logger.info('thresh type set to ' + cfg.thresh_type)
            low_th = cfg.thresh_low['thresh']
            flux_th = 10**(0.4 * (cfg.zpt - low_th)) * cfg.pixscale**2
            cfg.thresh_low['thresh'] = flux_th / stats.stdev
            high_th = cfg.thresh_high['thresh']
            flux_th = 10**(0.4 * (cfg.zpt - high_th)) * cfg.pixscale**2
            cfg.thresh_high['thresh'] = flux_th / stats.stdev
        elif cfg.thresh_type.lower() == 'stddev':
            cfg.logger.info('thresh type set to ' + cfg.thresh_type)
        else:
            raise Exception('invalid threshold type')

        cfg.logger.info('performing low threshold at '
                        '{:.2f} sigma'.format(cfg.thresh_low['thresh']))
        fpset_low = prim.image_threshold(
            mi_smooth, mask=mask, plane_name='THRESH_LOW', **cfg.thresh_low)
        cfg.logger.info('performing high threshold at '
                        '{:.2f} sigma'.format(cfg.thresh_high['thresh']))
        fpset_high = prim.image_threshold(
            mi_smooth, mask=mask, plane_name='THRESH_HIGH',
            **cfg.thresh_high)

        ############################################################
        # Get "cleaned" image, with noise replacement
        ############################################################

        cfg.logger.info('generating cleaned exposure')
        exp_clean = prim.clean(cfg.exp[cfg.band_detect],
                               fpset_low,
                               **cfg.clean)

        ############################################################
        # Remove small sources using thresholding. 
        # This is to remove **faint** and **small** sources.
        # The next step using SEP is to remove **bright** and **small** sources.
        ############################################################
        # Update to HUGS (JL 2023-12-17) #
        # Here we do two rounds of source detection. The first round
        # uses a finer grid for background and use bkg.rms() for detection.
        # This make sure that we detect faint sources in good SNR regions.
        # The second round uses a coarser grid for background and use
        # bkg.globalrms for detection. This make sure that we detect
        # sources in low SNR regions (which occur a lot to DECaLS DR10).
        ############################################################
        w = exp_clean.getWcs()
        img = exp_clean.getImage().getArray()
        
        mask = exp_clean.getMask()
        if 'SMALL' in mask.getMaskPlaneDict().keys():
            mask.removeAndClearMaskPlane('SMALL')
        mask_detect = afwImage.ImageF(exp_clean.getDimensions())
        
        # Round 1
        back = sep.Background(img, bw=64, bh=64)
        cat, segmap1 = sep.extract(img - back.back(), 3.5, err=back.rms(), minarea=10, 
                                deblend_nthresh=32, deblend_cont=0.001, 
                                segmentation_map=True)
        # Round 2
        back = sep.Background(img, bw=1200, bh=1200)
        cat, segmap2 = sep.extract(img - back.back(), 3., err=back.globalrms, minarea=5, 
                                deblend_nthresh=32, deblend_cont=0.001, filter_kernel=None,
                                segmentation_map=True)
        # Combine
        mask_detect.array[:] = segmap1 + segmap2
        footprint_set = afwDet.FootprintSet(mask_detect, afwDet.Threshold(0.01))
        mask.addMaskPlane('DETECTED')
        footprint_set.setMask(mask, 'DETECTED')

        exp_clean = prim.remove_small_sources_thresholding(exp_clean, cfg.hsc_small_sources_r_max, 
                                                           cfg.pixscale, cfg.rng)
        mi_clean = exp_clean.getMaskedImage()
        mask_clean = mi_clean.getMask()
        cfg.mi_clean = mi_clean

        ############################################################
        # use SEP to find and mask bright point-like sources
        ############################################################

        if cfg.sep_steps is not None:
            sep_stepper = SepLsstStepper(config=cfg.sep_steps)
            sep_stepper.setup_image(exp_clean, cfg.psf_sigma[cfg.band_detect] * 2.354, cfg.rng) # TODO: check this

            step_mask = cfg.exp.get_mask_array(band=cfg.band_detect,
                planes=['BRIGHT_OBJECT', 'NO_DATA', 'SAT'])
            sep_sources, _ = sep_stepper.run('sep_point_sources',
                                             mask=step_mask)

            cfg.logger.info('generating and applying sep ellipse mask')
            r_min = cfg.sep_min_radius
            sep_sources = sep_sources[sep_sources['fwhm'] < r_min] # Johnny used "flux_radius" here
            ell_msk = sep_ellipse_mask(
                sep_sources, sep_stepper.image.shape, cfg.sep_mask_grow)
            nimage_replace = utils.make_noise_image_jl(mi_clean, cfg.rng, back_size=32)[ell_msk]
            # nimage_replace = sep_stepper.noise_image[ell_msk]
            mi_clean.getImage().getArray()[ell_msk] = nimage_replace
            mask_clean.addMaskPlane('SMALL')
            mask_clean.getArray()[
                ell_msk] += mask_clean.getPlaneBitMask('SMALL')

        cfg.exp_clean = exp_clean
        # keep the original exposure before cleaning
        cfg.exp_ori = copy.deepcopy(cfg.exp)
        
        ############################################################
        # Clean images in both bands according to the mask we just
        # generated
        ############################################################
        cfg.logger.info('cleaning non-detection bands')
        replace = utils.get_mask_array(exp_clean.getMaskedImage(), planes=['CLEANED', "SMALL", 'BRIGHT_OBJECT'])
        for band in cfg.bands:
            mi_band = cfg.exp[band].getMaskedImage()
            noise_array = utils.make_noise_image_jl(mi_band, cfg.rng, replace, back_size=128)
            mi_band.getImage().getArray()[replace] = noise_array[replace]


        ############################################################
        # Detect sources and measure props with SExtractor
        ############################################################
        cfg.logger.info('detecting in {}-band'.format(cfg.band_detect))
        label = '{}'.format(cfg.brick)
        
        ############################################################
        # Smooth the original images for detection
        ############################################################
        cfg.logger.info('smoothing images for detection')
        exp_det = {}
        for band in cfg.bands:
            exp_det[band] = cfg.exp[band].clone()
            mi = exp_det[band].getMaskedImage()
            if band == 'g':
                extra_smooth = np.sqrt(depth_ratio)
            else:
                extra_smooth = 1
            mi = imtools.smooth_gauss(mi, 
                                      sigma=cfg.lsb_smooth_factor * cfg.psf_sigma[band] * extra_smooth,
                                      use_scipy=True, 
                                      inplace=False)
            exp_det[band].setMaskedImage(mi)
        cfg.exp_det = exp_det
        
        ############################################################
        # Run SExtractor on the cleaned image also do forced photometry
        ############################################################
        sources = Table()

        for band in cfg.bands:
            cfg.logger.info('measuring in {}-band'.format(band))
            dual_exp = None if band == cfg.band_detect else cfg.exp[band]
            sources_band = prim.detect_sources(
                cfg.exp_det[cfg.band_detect], 
                cfg.sex_config, cfg.sex_io_dir, label=label,
                dual_exp=dual_exp,
                delete_created_files=cfg.delete_created_files,
                original_fn=cfg.exp.fn[cfg.band_detect])
            if len(sources_band) > 0:
                sources = hstack([sources, sources_band])
                cfg.logger.info(f'total sources in {band}-band = {len(sources)}')
            else:
                cfg.logger.warn('**** no sources found by sextractor ****')
                results = _null_return(cfg, exp_clean)
                return results

        ############################################################
        # Verify detections in other bands in non-cleaned images using SExtractor
        ############################################################

        all_detections = sources.copy()

        for band in cfg.band_verify:
            cfg.logger.info('verifying dection in {}-band'.format(band))
            if band == 'g':
                cfg.sex_config['DETECT_THRESH'] = cfg.sex_config['DETECT_THRESH'] - 0.25
            sources_verify = prim.detect_sources(
                cfg.exp_det[band],
                cfg.sex_config, cfg.sex_io_dir,
                label=label, delete_created_files=cfg.delete_created_files,
                original_fn=cfg.exp.fn[band])
            if len(sources_verify) > 0:
                cfg.logger.info(f'total sources in {band}-band = {len(sources_verify)}')
                # match_masks, _ = xmatch(
                    # sources, sources_verify, max_sep=cfg.verify_max_sep)
                sources['re'] = sources[f'flux_radius_50_{cfg.band_detect}']
                # sources['re'] = sources['fwhm_r']
                # sources['re'] = np.sqrt(sources['a_image'] * sources['b_image'])
                match_masks, _ = xmatch_re(
                    sources, sources_verify, max_sep=cfg.verify_max_sep)
                txt = 'cuts: {} out of {} objects detected in {}-band'.format(
                    len(match_masks[0]), len(sources), band)
                cfg.logger.info(txt)
                if len(match_masks[0]) == 0:
                    cfg.logger.warn(
                        '**** no matched sources with ' + band + ' ****')
                    results = _null_return(cfg, exp_clean)
                    return results
                sources = sources[match_masks[0]]
            else:
                cfg.logger.warn(
                    '**** no sources detected in ' + band + ' ****')
                results = _null_return(cfg, exp_clean)
                return results

        mask_fracs = utils.calc_mask_bit_fracs(exp_clean)
        cfg.exp.patch_meta.small_frac = mask_fracs['small_frac']
        cfg.exp.patch_meta.cleaned_frac = mask_fracs['cleaned_frac']
        cfg.exp.patch_meta.bright_obj_frac = mask_fracs['bright_object_frac']

        cfg.logger.info('measuring mophology metrics')
        prim.measure_morphology_metrics(exp_clean.getImage().getArray(),
                                        sources)

        cfg.logger.info('task completed in {:.2f} min'.format(cfg.timer))

        results = Struct(all_detections=all_detections,
                         sources=sources,
                         hugs_exp=cfg.exp,
                         exp_clean=exp_clean,
                         exp_det=exp_det,
                         success=True,
                         synths=cfg.exp.synths)

        if reset_mask_planes:
            cfg.reset_mask_planes()

        return results

    except:
        cfg.logger.critical(
            'DECaLS brick {} failed'.format(cfg.brick))
        results = _null_return(cfg)
        return results


def _null_return(config, exp_clean=None):
    config.reset_mask_planes()
    return Struct(all_detections=None,
                  sources=None,
                  hugs_exp=config.exp,
                  exp_clean=exp_clean,
                  success=False,
                  synths=None)
