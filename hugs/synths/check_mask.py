from __future__ import division, print_function

import numpy as np
import lsst.geom
from .. import lsstutils
from ..log import logger


DEFAULT_PLANES = ['BRIGHT_OBJECT', 'NO_DATA']


def get_mask_array(exp, planes=DEFAULT_PLANES):
    mask = exp.getMaskedImage().getMask()
    arr = np.zeros(mask.getArray().shape, dtype=bool)
    for p in planes:
        if p in mask.getMaskPlaneDict().keys():
            arr |= mask.getArray() & mask.getPlaneBitMask(p) != 0
    return arr


def find_masked_synths(synth_cat, exp, planes=['BRIGHT_OBJECT']):

    if type(planes) == str:
        planes = [planes]

    afwcoords = lsstutils.make_afw_coords(synth_cat['ra', 'dec'])

    # get mask array and find masked synths
    xy0 = exp.getXY0()
    wcs = exp.getWcs()
    bbox = exp.getBBox()
    mask_arr = get_mask_array(exp, planes)
    masked= []
    for coord in afwcoords:
        pixel = wcs.skyToPixel(coord)
        if bbox.contains(lsst.geom.Point2I(pixel)):
            j, i = pixel - xy0
            masked.append(int(mask_arr[int(i), int(j)]))
        else:
            logger.warn('synth not in exposure')
            masked.append(0)

    masked = np.array(masked)

    planes  = ', '.join(planes)
    msg = '{} out of {} synths were masked as {}'
    logger.info(msg.format(np.sum(masked), len(synth_cat), planes))

    return masked
