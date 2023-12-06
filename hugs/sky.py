import sep

def sky_bkg_rms(mi, back_size=128, back_filtersize=5):
    """
    Return the RMS of the sky background in the current image.
    
    Parameters
    ----------
    mi: numpy.ndarray
        Masked image.
    back_size: int, optional
        Size of the background mesh.
    back_filtersize: int, optional
        Size of the background mesh filter.
    """
    mi = mi.byteswap().newbyteorder()
    bkg = sep.Background(mi, bw=back_size, bh=back_size, 
                         fw=back_filtersize, fh=back_filtersize)
    return bkg.back(), bkg.rms()

