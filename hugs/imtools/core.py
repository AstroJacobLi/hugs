from __future__ import division, print_function

import os
import numpy as np
import lsst.daf.persistence
import lsst.afw.image as afwImage
import lsst.afw.display as afwDisp
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import scipy.ndimage as ndi
hscdir = os.environ.get('HSC_DIR')

__all__ = ['get_image_ndarray',
           'get_cutout', 
           'smooth_gauss', 
           'smooth_image', 
           'rmedian', 
           'unsharp_mask']


def get_image_ndarray(image):
    """
    Return numpy ndarray given exposure or image object.
    """
    if type(image)==lsst.afw.image.MaskedImageF:
        data = image.getImage().getArray().copy()
    elif type(image)==lsst.afw.image.ExposureF:
        mi = image.getMaskedImage()
        data = mi.getImage().getArray().copy()
    elif type(image)==np.ndarray:
        data = image
    else:
        raise TypeError('Invalid image dtype')
    return data


def get_cutout(center, size, exp=None, data_id=None, butler=None):
    """
    Generate a cutout of exposure. Must give exposure object or 
    data_id and, optionally, a butler.
     
    Parameters
    ----------
    center : tuple
        Center of desired cutout in the tract
        and patch system.
    size : float
        Size to grow bbox in all directions.
    exp : lsst.afw.image.ExposureF, optional
        Exposure from which to get cutout.    
    data_id : dict, optional
        HSC data ID.
    butler : lsst.daf.persistence.Butler, optional
        the butler.

    Returns
    -------
    cutout : lsst.afw.image.ExposureF
        Desired exposure cutout.
    """

    # get exposure
    if exp is None:
        if butler is None:
            butler = lsst.daf.persistence.Butler(hscdir)
        exp = butler.get('deepCoadd_calexp', data_id, immediate=True)

    # generate bbox and get cutout
    center = afwGeom.Point2I(int(center[0]), int(center[1]))
    bbox = afwGeom.Box2I(center, center)
    bbox.grow(size)
    bbox.clip(exp.getBBox(afwImage.PARENT))
    cutout = exp.Factory(exp, bbox, afwImage.PARENT)

    return cutout


def smooth_gauss(masked_image, sigma=2.0, nsigma=7.0, 
                 inplace=False, use_scipy=False, **kwargs):
    """
    Smooth image with a Gaussian kernel. 

    Parameters
    ----------
    masked_image : lsst.afw.image.imageLib.MaskedImageF
        Masked image object to be smoothed
    sigma : float
        Standard deviation of Gaussian
    nsigma : float, optional
        Number of sigma for kernel width
    inplace : bool, optional
        If True, smooth the image inplace. Else, return a clone 
        of the masked image. 
    use_scipy : bool, optional
        If True, perform smoothing using scipy.ndimage.

    Returns
    -------
    convolved_image : lsst.afw.image.imageLib.MaskedImageF
        The convolved masked image
    """
    if use_scipy:
        img_arr = get_image_ndarray(masked_image)
        img_arr_smooth = ndi.gaussian_filter(
            img_arr, sigma, truncate=nsigma, **kwargs)
        if inplace:
            convolved_image = masked_image
        else:
            # convolved_image = masked_image.Factory(masked_image.getBBox())
            convolved_image = masked_image.clone()
        convolved_image.getImage().getArray()[:] = img_arr_smooth
    else:
        width = (int(sigma*nsigma + 0.5) // 2)*2 + 1 # make sure it is odd
        gauss_func = afwMath.GaussianFunction1D(sigma)
        gauss_kern = afwMath.SeparableKernel(
            width, width, gauss_func, gauss_func)
        convolved_image = masked_image.Factory(masked_image.getBBox())
        afwMath.convolve(convolved_image, masked_image, gauss_kern,
                         afwMath.ConvolutionControl())
        if inplace:
            img_arr_smooth = convolved_image.getImage().getArray()
            masked_image.getImage().getArray()[:] = img_arr_smooth
    return convolved_image


def smooth_image(masked_image, kernel='exp', **kwargs):
    """
    Helper function for smoothing images with gaussian or 
    exponential kernels.

    Parameters
    ----------
    masked_image : lsst.afw.image.imageLib.MaskedImageF
        Masked image object to be smoothed
    kernel : string
        Kernel function (exp or gauss). 
    **kwargs : dict 
        Args for smooth_gauss or exp_kern.

    Returns
    -------
    convolved_image : lsst.afw.image.imageLib.MaskedImageF
        The convolved masked image
    """
    if kernel=='gauss':
        return smooth_gauss(masked_image, **kwargs)
    elif kernel=='exp':
        from photutils.utils.convolution import filter_data
        from .kernels import exp_kern
        img_arr = get_image_ndarray(masked_image)
        kern = exp_kern(alpha=kwargs['alpha'], size=kwargs['size'])
        img_arr_smooth = filter_data(img_arr, kern, mode='reflect')
        convolved_image = masked_image.Factory(masked_image.getBBox())
        convolved_image.getImage().getArray()[:] = img_arr_smooth
        return convolved_image


def _ring(r_inner, r_outer, dtype=np.int, invert=False):
    """
    Generate a 2D ring footprint.

    Paramters
    ---------
    r_inner : int
        The inner radius of ring in pixels.
    r_outer : int
        The outer radius of ring in pixels.
    dtype : data type, optional
        The data type of the output array
    invert : bool, optional
        If True, invert the ring kernel (i.e., 1 <--> 0).

    Returns
    -------
    fp : 2D ndarray
        The ring footprint with 1's in the 
        annulus and 0's everywhere else.
    """
    assert r_outer >= r_inner, 'must have r_outer >= r_inner'
    x = np.arange(-int(r_outer), int(r_outer)+1)
    r = np.sqrt(x**2 + x[:,None]**2)
    annulus = (r>r_inner) & (r<=r_outer)
    if invert:
        annulus = ~annulus
    r[annulus] = 1
    r[~annulus] = 0
    fp = r.astype(dtype)
    return fp


def rmedian(image, r_inner, r_outer, **kwargs):
    """
    Median filter image with a ring footprint. This
    function produces results similar to the IRAF 
    task of the same name (except this is much faster). 

    Parameters
    ----------
    image : ndarray, MaskedImageF, or ExposureF
       Original image data. 
    r_inner : int
        The inner radius of the ring in pixels.
    r_outer : int
        The outer radius of the ring in pixels.

    Returns
    -------
    filtered_data : ndarray
        Ring filtered image.
    """
    data = get_image_ndarray(image)
    fp = _ring(r_inner, r_outer, **kwargs)
    filtered_data = ndi.median_filter(data, footprint=fp)
    return filtered_data


def unsharp_mask(image, filter_size=9):
    """
    Generate an unsharp mask from the input image.

    Parameters
    ----------
    image : ndarray, MaskedImageF, or ExposureF
        Original image data.
    filter_size : int, optional
        Median filter size in pixels.

    Returns
    -------
    unsharp : ndarray
        Unsharp mask.
    """
    data = get_image_ndarray(image)
    filtered_data = ndi.median_filter(data, size=filter_size)
    unsharp = data - filtered_data
    return unsharp
