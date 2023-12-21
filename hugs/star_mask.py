import os
import numpy as np
from astropy.table import Table, vstack
import astropy.units as u
from shapely.geometry import Polygon
from shapely import affinity
import rasterio.features
import sep
import traceback

import lsst.geom as geom
from astropy.coordinates import SkyCoord, match_coordinates_sky

def get_gaia_stars(exposure, size_buffer=1.4, tigress=True, logger=None):
    """
    Search for bright stars using GAIA catalogs on Tigress (`/tigress/HSC/refcats/htm/gaia_dr3_20230707`).
    For more information, see https://community.lsst.org/t/gaia-dr2-reference-catalog-in-lsst-format/3901.
    This function requires `lsstpipe`.

    Parameters
    ----------
    exposure: `lsst.afw.image.ExposureF` object
        Input image.
    size_buffer: float, optional
        Buffer size in units of image size. 
        This is used to search for stars in a slightly larger area than the image.
        Default: 1.4
    tigress: bool, optional
        Whether to search for GAIA stars on Princeton's Tigress.
    
    Return
    ------
    gaia_results: `astropy.table.Table` object
        A catalog of matched stars.
    """
    
    # Get the image size
    w = exposure.getWcs()
    bbox = exposure.getBBox()
    afw_coord = w.pixelToSky(bbox.centerY, bbox.centerX) # maybe wrong order
    img_size = (w.getPixelScale() * max(bbox.getDimensions())).asDegrees()
    
    if tigress:
        # Search for stars in Gaia DR3 catatlogs, which are stored in
        # `/tigress/HSC/refcats/htm/gaia_dr3_20230707`.
        gaia_dir = '/tigress/HSC/refcats/htm/gaia_dr3_20230707'
        try:
            from lsst.meas.algorithms.htmIndexer import HtmIndexer
            import lsst.geom as geom

            def getShards(afw_coord, radius):
                htm = HtmIndexer(depth=7)
                shards, onBoundary = htm.getShardIds(
                    afw_coord, radius * geom.degrees)
                return shards

        except ImportError as e:
            # Output expected ImportErrors.
            if logger is not None:
                logger.error(e)
                logger.error(
                    'LSST Pipe must be installed to query Gaia stars on Tigress.')
            print(e)
            print('LSST Pipe must be installed to query Gaia stars on Tigress.')

        # find out the Shard ID of target area in the HTM (Hierarchical triangular mesh) system
        if logger is not None:
            logger.info('    Taking Gaia catalogs stored in `Tigress`')
            print('    Taking Gaia catalogs stored in `Tigress`')

        shards = getShards(afw_coord, img_size * size_buffer) # slightly larger than the image size

        cat = vstack([Table.read(os.path.join(gaia_dir, f'{index}.fits')) for index in shards])
        cat['coord_ra'] = cat['coord_ra'].to(u.degree)
        cat['coord_dec'] = cat['coord_dec'].to(u.degree)
        
    else:
        # Search for stars
        from astroquery.gaia import Gaia
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
        gaia_results = Gaia.query_object_async(
            coordinate=SkyCoord(afw_coord.ra, afw_coord.dec, unit='deg'),
            width=img_size.asDegrees() * size_buffer * u.deg,
            height=img_size.asDegrees() * size_buffer * u.deg,
            verbose=False)

    # Trim this catalog a little bit
    # Ref: https://github.com/MerianSurvey/caterpillar/blob/main/caterpillar/catalog.py
    if cat:  # if not empty
        gaia_results = cat['coord_ra', 'coord_dec']
        gaia_results.rename_columns(['coord_ra', 'coord_dec'], ['ra', 'dec'])

        gaia_results['phot_g_mean_mag'] = -2.5 * np.log10(cat['phot_g_mean_flux'].value * 1e-9 / 3631)  # AB magnitude
        gaia_results['phot_bp_mean_mag'] = -2.5 * np.log10(cat['phot_bp_mean_flux'].value * 1e-9 / 3631)  # AB magnitude
        gaia_results['phot_rp_mean_mag'] = -2.5 * np.log10(cat['phot_rp_mean_flux'].value * 1e-9 / 3631)  # AB magnitude

        return gaia_results
    

def mask_cross(mask, x, y, length, width, rotate=0):
    """
    Mask a cross shape in the given mask, in place.
    Similar to `sep.mask_ellipse`, but for a cross shape.
    
    Parameters
    ----------
    mask : ndarray
        Mask to be modified.
    x, y : float
        Center of the cross.
    length, width : float
        Length and width of the cross.
    rotate : float, optional
        Rotation angle of the cross, in degrees.
        
    Returns
    -------
    None. The input mask is modified in place.
    """
    assert mask.dtype == bool, 'mask must be boolean'
    half_length = length/2
    half_width = width/2
    
    poly1 = Polygon([
        (x - half_length, y - half_width),
        (x - half_length, y + half_width),
        (x + half_length, y + half_width),
        (x + half_length, y - half_width)
        ])
    poly2 = affinity.rotate(poly1, 90, 'center')
    cross = poly1.union(poly2)
    if rotate != 0:
        cross = affinity.rotate(cross, rotate, 'center')
    flag = rasterio.features.rasterize([cross], out_shape=mask.shape).astype(bool)
    mask[flag] = True
    
    
def scott_star_mask(exposure, p1, p2, bright_thresh, tigress=True):
    """
    Star mask for DECaLS image, based on Scott Carlsten's code.
    Step 1: Do a detection for bright objects, and match with Gaia catalog.
    Step 2: Fit a relation between the segmap size and g-band magnitude. 
    Such relation is used to determine the size of the mask for faint stars.
    Step 3: Mask bright stars with both a circular mask and a cross mask.
    Step 4: Mask fainter stars with both a circular mask and a cross mask.
    
    Parameters
    ----------
    exposure : `lsst.afw.image.ExposureF` object
        DECaLS image.
    p1, p2 : float
        Parameters for the relation between the segmap size and g-band magnitude.
        The relation is given by `logarea = p1 * gmag + p2`.
        Smaller `p1` means larger mask size, and larger p2 means larger mask size.
        Changing `p1` will change the mask size contrast between bright and faint stars. 
    bright_thresh : float
        Bright star threshold in GAIA g-band, used to determine the mask size for bright stars.
        Stars brighter than this threshold will be masked aggressively.
        
    Returns
    -------
    mask : ndarray
        Star mask.
    """
    try:
        bbox = exposure.getBBox()
        
        # Get gaia star catalog
        gaia = get_gaia_stars(exposure, tigress=tigress)
        
        # Do a detection for bright objects, and match with Gaia catalog
        w = exposure.getWcs()
        img = exposure.getImage().getArray()
        back = sep.Background(img, bw=256, bh=256)
        img -= back.back()
        cat, _ = sep.extract(img, 50, err=back.rms(), minarea=10, 
                            deblend_nthresh=32, deblend_cont=0.01, 
                            segmentation_map=True)
        world = np.rad2deg(np.array(w.pixelToSkyArray(cat['x'], cat['y'])).T)

        cat_coord = SkyCoord(world, unit='deg')
        gaia_coord = SkyCoord(gaia['ra'], gaia['dec'])
        
        # Fit a relation between the segmap size and g-band magnitude
        # this relation is used to determine the size of the mask for faint stars
        inds, dist, _ = match_coordinates_sky(cat_coord, gaia_coord)
        matched = (dist < 0.5 * u.arcmin)
        logarea = np.log10(cat['npix'])[matched]
        gmag = gaia['phot_g_mean_mag'][inds[matched]]

        logarea = logarea[gmag < 19]
        gmag = gmag[gmag < 19]

        if len(gmag) > 0:
            poly = np.polyfit(gmag, logarea, 1)
            good_flag = abs(logarea - np.poly1d(poly)(gmag)) < 0.5 # remove outliers that are >0.5 far away
            poly = np.polyfit(gmag[good_flag], logarea[good_flag], 1)
        else:
            poly = [-0.18, 5.4]
        poly[0] *= p1
        poly[1] += p2
        
        # Mask bright stars. We have a circular mask and a cross mask.
        # For these bright stars, we also rotate the cross mask by 45 degrees.
        bright_buffer = 400
        x, y = w.skyToPixelArray(np.deg2rad(gaia['ra'].data), 
                                np.deg2rad(gaia['dec'].data))
        flag = (x > bbox.getBeginX() - bright_buffer) & (
            x < bbox.getEndX() + bright_buffer) & (
            y > bbox.getBeginY() - bright_buffer) & (
            y < bbox.getEndY() + bright_buffer) # only mask stars within the bbox + buffer
        bright_gaia = gaia[flag & (gaia['phot_g_mean_mag'] < bright_thresh)]
        
        mask_bright = np.zeros_like(img).astype(bool)
        for star in bright_gaia:
            ra, dec = star['ra'], star['dec']
            pix = w.skyToPixel(geom.SpherePoint(geom.Angle(ra, geom.degrees), geom.Angle(dec, geom.degrees)))
            x, y = pix.x, pix.y
            bokeh_size = 900 * np.exp(-star['phot_g_mean_mag'] / 7.5)
            if not np.isfinite(bokeh_size):
                bokeh_size = 200
            bokeh_size = np.min([bokeh_size, 1800]) # max size is 1800
            sep.mask_ellipse(mask_bright, x, y, bokeh_size, bokeh_size, 0, r=1)
            mask_cross(mask_bright, x, y, 5 * bokeh_size, 2 * bokeh_size//5, rotate=0)
            if star['phot_g_mean_mag'] < 12:
                mask_cross(mask_bright, x, y, 3 * bokeh_size, 1.5 * bokeh_size//8, rotate=45)
            
        # Mask fainter stars
        size_buffer = 0
        x, y = w.skyToPixelArray(np.deg2rad(gaia['ra'].data), np.deg2rad(gaia['dec'].data))
        flag = (x > bbox.getBeginX() - size_buffer) & (
            x < bbox.getEndX() + size_buffer) & (
            y > bbox.getBeginY() - size_buffer) & (
            y < bbox.getEndY() + size_buffer) # only mask stars within the bbox + buffer
        faint_gaia = gaia[flag]
        faint_gaia = faint_gaia[faint_gaia['phot_g_mean_mag'] < 20]
        
        mask_faint = np.zeros_like(img).astype(bool)
        for star in faint_gaia:
            ra, dec = star['ra'], star['dec']
            pix = w.skyToPixel(geom.SpherePoint(geom.Angle(ra, geom.degrees), geom.Angle(dec, geom.degrees)))
            x, y = pix.x, pix.y
            if np.isnan(x) or np.isnan(y):
                continue
            size = np.poly1d(poly)(star['phot_g_mean_mag'])
            if not np.isfinite(size):
                size = 3.5
            size = int((10**(size)/np.pi)**0.5)
        #     size *= 2
            if size > 1800:
                size = 1800
            sep.mask_ellipse(mask_faint, x, y, size, size, 0, r=1)
            mask_cross(mask_faint, x, y, 3 * size, 2 * size//5)
            
        mask = mask_bright | mask_faint
        return mask
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print('Scott star mask failed. Return empty mask.')
        # return np.zeros_like(exposure.getImage().getArray()).astype(bool)