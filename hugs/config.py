from __future__ import division, print_function

import os
import sys
import logging
import yaml
import numpy as np
import time
from . import utils
from .exposure import HugsExposure, SynthHugsExposure, DecalsExposure
from .synths.catalog import GlobalSynthCat, generate_patch_cat

from .log import HugsLogger
logging.setLoggerClass(HugsLogger)

# def update_nested_dict(d, u):
#     for k, v in u.items():
#         if isinstance(v, dict):
#             d[k] = update_nested_dict(d.get(k, {}), v)
#         else:
#             d[k] = v
#     return d

def update_nested_dict(d, u):
    d_copy = d.copy()
    for k, v in u.items():
        if isinstance(v, dict):
            d_copy[k] = update_nested_dict(d_copy.get(k, {}), v)
        else:
            d_copy[k] = v
    return d_copy


class PipeConfig(object):
    """
    Class for parsing the hugs configuration.

    Parameters
    ----------
    config_fn : string, optional
        The parameter file name (must be a yaml file).
        If None, will use default config.
    brick : string, optional
        DECaLS brick name.
    log_level : string, optional
        Level of python logger.
    random_state : int, list of ints, RandomState instance, or None 
        If int or list of ints, random_state is the rng seed.
        If RandomState instance, random_state is the rng.
        If None, the rng is the RandomState instance used by np.random.
    run_name : sting, optional
        Label for this pipeline run.
    rerun_path : string, optional
        Path to rerun directory, e.g., f'decals/ngc5055/tracts/'. 
        If None, will use default path.
    host_setting : dict, optional
        Host-specific settings. If None, will use default settings in the 
        config file.
    """

    def __init__(self, config_fn=None, brick=None, host_setting=None,
                 log_level='info', random_state=None, log_fn=None,
                 run_name='hugs', rerun_path=None):

        # read parameter file & setup param dicts
        self.config_fn = config_fn if config_fn else utils.default_config_fn
        with open(self.config_fn, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        if host_setting is not None:
            params = update_nested_dict(params, host_setting)
            # write to file
        with open(os.path.join(params['hugs_io'], run_name + '.yml'), 'w') as f:
            yaml.dump(params, f, default_flow_style=False)

        self.data_dir = params['data_dir']
        self.rerun_path = rerun_path
        self.hugs_io = params['hugs_io']
        self.log_fn = log_fn
        self.min_good_data_frac = params['min_good_data_frac']
        self.bright_star = params['bright_star']
        # self.bright_star = star_setup['bright_star_thresh']
        self.inject_synths = params['inject_synths']
        self.coadd_label = params.pop('coadd_label', 'deepCoadd_calexp')
        self.use_andy_mask = params.pop('use_andy_mask', True)
        if self.inject_synths:
            self.synth_cat_type = params['synth_cat_type']
            self.synth_cat_fn = params.pop('synth_cat_fn', None)
            self.synth_check_masks = params.pop('synth_check_masks',
                                                ['BRIGHT_OBJECT'])
            self.synth_sersic_params = params.pop('synth_params', {})
            self.synth_image_params = params.pop('synth_image_params', {})
            self.synth_max_match_sep = params.pop('synth_max_match_sep', 5)
            self.synth_model = params.pop('model', 'sersic')

        self.thresh_type = params['thresh_type']
        self._thresh_low = params['thresh_low']
        self._thresh_high = params['thresh_high']
        self._clean = params['clean']
        self.rng = utils.check_random_state(random_state)
        self._clean['random_state'] = self.rng
        self.log_level = log_level
        self.run_name = run_name
        self._butler = None
        self._timer = None

        self.band_mask = params['band_mask']
        self.band_detect = params['band_detect']
        self.band_verify = params['band_verify']
        self.band_meas = params['band_meas']
        self.mean_seeing_sigma = params['mean_seeing_sigma']
        self.pixscale = params['pixel_scale']
        self.zpt = params['zpt']

        self.small_sources_r_max = params.pop('small_sources_r_max', None)
        self.lsb_smooth_factor = params.pop('lsb_smooth_factor', 1.75)
        self.lsb_smooth_scale = params.pop('lsb_smooth_scale', None)
        # setup for sextractor
        sex_setup = params['sextractor']
        self.sex_config = sex_setup['config']
        # in default
        self.sex_config['PHOT_APERTURES'] = '3,4,5,6,7,8,16,32,42,54'
        self.sex_config['PHOT_FLUXFRAC'] = '0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.8,0.9'
        self.delete_created_files = sex_setup['delete_created_files']
        self.sex_io_dir = sex_setup['sex_io_dir']
        self.verify_max_sep = sex_setup['verify_max_sep']

        self.circ_aper_radii = self.sex_config['PHOT_APERTURES'].split(',')
        self.circ_aper_radii = [
            utils.pixscale * float(a) / 2 for a in self.circ_aper_radii]
        self.num_apertures = len(self.circ_aper_radii)

        # setup for sep
        self.sep_steps = params.pop('sep_steps', None)
        if self.sep_steps is not None:
            sep_point_sources = self.sep_steps['sep_point_sources']
            self.sep_min_radius = sep_point_sources.pop('min_radius', 2.0)
            self.sep_mask_grow = sep_point_sources.pop('mask_grow', 5)
            self.sep_radius_colname = sep_point_sources.pop('radius_colname', 'flux_radius')

        # set patch id if given
        if brick is not None:
            self.set_brick_id(brick)
        else:
            self.brick = None

    def setup_logger(self, brick):
        """
        Setup the python logger.
        """
        name = 'hugs: {}'.format(brick)
        self.logger = logging.getLogger(name)
        self.logger._set_defaults(self.log_level.upper())
        self.logger.info('starting ' + name)

        if self.log_fn is not None:
            fh = logging.FileHandler(self.log_fn)
            self.logger.addHandler(fh)

    @property
    def timer(self):
        """
        Timer for pipeline. 
        """

        if self._timer is None:
            self._timer = time.time()
        else:
            delta_time = (time.time() - self._timer) / 60.0
            self._timer = time.time()
            return delta_time

    def reset_mask_planes(self):
        """
        Remove mask planes created by hugs-pipe.
        """

        for band in self.bands:
            mask = self.exp[band].getMaskedImage().getMask()
            try:
                utils.remove_mask_planes(mask, ['SMALL',
                                                'CLEANED',
                                                'THRESH_HIGH',
                                                'THRESH_LOW'])
            except Exception as e:
                self.logger.warning(e)

    def set_brick_id(self, brick):
        """
        Setup the DECaLS brick. This must be done before running the pipeline

        Parameters
        ----------
        brick : string
            DECaLS brick name.
        """
        self.brick = brick
        self.setup_logger(brick)

        # careful not to modify parameters
        self.thresh_low = self._thresh_low.copy()
        self.thresh_high = self._thresh_high.copy()
        self.clean = self._clean.copy()

        # get exposures
        bands = self.band_detect + self.band_verify + self.band_meas
        self.bands = ''.join(set(bands))

        if self.inject_synths:
            if self.synth_cat_type == 'global':
                assert self.synth_cat_fn is not None
                self.synth_cat = GlobalSynthCat(cat_fn=self.synth_cat_fn)
            elif self.synth_cat_type == 'on-the-fly':
                self.synth_cat = generate_patch_cat(
                    sersic_params=self.synth_sersic_params,
                    **self.synth_image_params)

            self.exp = SynthHugsExposure(self.synth_cat, tract, patch,
                                         self.bands, self.butler,
                                         rerun=self.data_dir,
                                         coadd_label=self.coadd_label,
                                         band_detect=self.band_detect,
                                         use_andy_mask=self.use_andy_mask,
                                         synth_model=self.synth_model)
        else:
            self.exp = DecalsExposure(brick, self.bands, 
                                      data_dir=os.path.join(self.data_dir, self.rerun_path),
                                      band_detect=self.band_detect, datatype='image')
            try:
                # if file exist
                rerun_path = "/".join(self.rerun_path.split('/')[:2]) + '/tracts/'
                invvar_file = os.path.join(self.data_dir, rerun_path, f'legacysurvey-{brick}-invvar-{self.band_detect}.fits')
                if os.path.exists(invvar_file):
                    self.exp_invvar = DecalsExposure(brick, self.bands, 
                                            data_dir=os.path.join(self.data_dir, rerun_path),
                                            band_detect=self.band_detect, datatype='invvar')
                else:
                    self.logger.warning('no invvar file found...')
            except FileNotFoundError:
                self.logger.warning('no invvar file found...')

        # clear detected mask and remove unnecessary plane
        for band in self.bands:
            mask = self.exp[band].getMaskedImage().getMask()
            utils.remove_mask_planes(mask, ['CR',
                                            'CROSSTALK',
                                            'DETECTED_NEGATIVE',
                                            'NOT_DEBLENDED',
                                            # 'SUSPECT',
                                            'UNMASKEDNAN',
                                            'INEXACT_PSF',
                                            'REJECTED',
                                            'CLIPPED'])

        try:
            self.psf_sigma = utils.get_psf_sigma(self.exp[self.band_detect])
        except AttributeError:
            self.logger.warning('no psf with exposure... using mean seeing')
            self.psf_sigma = self.mean_seeing_sigma

        # setup thresh low/high/det: n_sig_grow --> rgrow
        ngrow = self.thresh_low.pop('n_sig_grow')
        self.thresh_low['rgrow'] = int(ngrow * self.psf_sigma + 0.5)

        ngrow = self.thresh_high.pop('n_sig_grow')
        self.thresh_high['rgrow'] = int(ngrow * self.psf_sigma + 0.5)

        # convert clean min_pix to psf units
        if 'psf sigma' in str(self.clean['min_pix_low_thresh']):
            nsig = int(self.clean['min_pix_low_thresh'].split()[0])
            self.clean['min_pix_low_thresh'] = np.pi * \
                (nsig * self.psf_sigma)**2

        # clean n_sig_grow --> rgrow
        ngrow = self.clean.pop('n_sig_grow')
        if ngrow is None:
            self.clean['rgrow'] = None
        else:
            self.clean['rgrow'] = int(ngrow * self.psf_sigma + 0.5)
            self.logger.info('growing clean footprints with rgrow = {:.1f}'.
                             format(self.clean['rgrow']))

        # sextractor parameter file
        fn = '{}-{}.params'.format(brick, self.run_name)
        self.sex_config['PARAMETERS_NAME'] = os.path.join(self.sex_io_dir, fn)
