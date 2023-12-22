import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.visualization import ZScaleInterval

__all__ = ['plot_sep_sources']


def plot_sep_sources(image, catalog, ec='lime', scale_ell=6, subplots=None,
                     mark_centers=False, subplot_kw=dict(figsize=(10, 10)),
                     ell_type='ab', per_lo=1.0, per_hi=99.0, mask=None,
                     mask_kws=dict(cmap='Blues_r', alpha=0.5)):

    fig, ax = subplots if subplots is not None else plt.subplots(**subplot_kw)

    if image is not None:
        if len(image.shape) == 2:
            vmin, vmax = np.percentile(image, [per_lo, per_hi])
            ax.imshow(image, vmin=vmin, vmax=vmax,
                      origin='lower', cmap='gray_r')
        else:
            ax.imshow(image, origin='lower')
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

    ax.set(xticks=[], yticks=[])

    for src in catalog:
        if ell_type == 'ab':
            a = src['a']
            b = src['b']
        elif 'kronrad' in ell_type:
            kronrad = src[ell_type]
            a = kronrad * src['a']
            b = kronrad * src['b']
        e = Ellipse((src['x'], src['y']),
                    width=scale_ell*a,
                    height=scale_ell*b,
                    angle=src['theta']*180/np.pi, fc='none', ec=ec)
        ax.add_patch(e)

        if mark_centers:
            ax.plot(x_c, y_c, 'r+')

    if mask is not None:
        mask = mask.astype(float)
        mask[mask == 0.0] = np.nan
        ax.imshow(mask, **mask_kws)

    return fig, ax

def plot_sex_sources(image, catalog, ec='lime', scale_ell=6, subplots=None,
                     mark_centers=False, subplot_kw=dict(figsize=(10, 10)),
                     ell_type='ab', per_lo=1.0, per_hi=99.8, mask=None,
                     mask_kws=dict(cmap='Blues_r', alpha=0.5)):
    """
    Plot sources from SExtractor catalog, as returned by hugs. 
    
    Parameters
    ----------
    image : numpy.ndarray
        Image array.
    catalog : astropy.table.Table
        SExtractor catalog.
    ec : str
        Color of ellipse edges.
    scale_ell : float
        Scaling factor for ellipse size.
    subplots : tuple
        Tuple of matplotlib.pyplot.subplots.
    mark_centers : bool
        Whether mark the centers of the sources.
    subplot_kw : dict
        Keyword arguments for matplotlib.pyplot.subplots.
    ell_type : str
        Type of ellipse to plot. Options are 'ab' or 'kron_rad'.
    per_lo : float
        Lower percentile for image scaling.
    per_hi : float
        Upper percentile for image scaling.
    mask : numpy.ndarray
        Mask array.
    mask_kws : dict
        Keyword arguments for mask imshow.
    """

    fig, ax = subplots if subplots is not None else plt.subplots(**subplot_kw)

    if image is not None:
        if len(image.shape) == 2:
            vmin, vmax = np.percentile(image, [per_lo, per_hi])
            ax.imshow(image, vmin=vmin, vmax=vmax,
                      origin='lower', cmap='gray_r')
        else:
            ax.imshow(image, origin='lower')
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

    ax.set(xticks=[], yticks=[])

    for src in catalog:
        if ell_type == 'ab':
            a = src['a_image']
            b = src['b_image']
        elif ell_type == 'kron_rad':
            kronrad = src[ell_type]
            a = kronrad * src['a_image']
            b = kronrad * src['b_image']
        else:
            raise ValueError('ell_type must be `ab` or `kron_rad`')
        e = Ellipse((src['x_image'], src['y_image']),
                    width=scale_ell*a,
                    height=scale_ell*b,
                    angle=src['theta_image']*180/np.pi, 
                    fc='none', ec=ec)
        ax.add_patch(e)

        if mark_centers:
            ax.plot(src['x_image'], src['y_image'], 'r+')

    if mask is not None:
        mask = mask.astype(float)
        mask[mask == 0.0] = np.nan
        ax.imshow(mask, **mask_kws)

    return fig, ax

def show_step(img, ax, vmin, vmax, title, seg=None, alpha=0.6, cmap=plt.cm.gnuplot):

    ax.imshow(img, cmap='gray_r', vmin=vmin, vmax=vmax, origin='lower')

    if seg is not None:
        if type(seg) is not list:
            seg = [seg]
        if type(alpha) is not list:
            alpha = [alpha]*len(seg)
        for s, a in zip(seg, alpha):
            ax.imshow(s, alpha=a, cmap=cmap, vmin=0, vmax=1, origin='lower')

    ax.set_title(title, fontsize=15)


###############################################################################
# Below are from Scott's code. For completeness plot.
###############################################################################

def plot_radii(ax, mags, mu0s, dmod, re_col='black'):
    import matplotlib.patheffects as path_effects
    mu0s, mags = np.meshgrid(mu0s, mags)
    mues = mu0s + 1.822
    mue_avgs = mues - 0.699

    rads = np.sqrt(10**((mue_avgs-mags-dmod)/2.5)/2/np.pi)
    CS = ax.contour(mu0s, mags, rads,
                    levels=[4, 8, 12, 30],
                    colors='LimeGreen', lw=4)
    CS.levels = [int(val) for val in CS.levels]
    fmt = r'${%r}$ "'
    labels = ax.clabel(CS, CS.levels, inline=True,
                       fmt=fmt,
                       colors=re_col)
    # for tt in labels:
    #    tt.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),
    # path_effects.Normal()])


def draw_lg_trends(ax, dmod):
    """
    Mass-size relation of Local Group???
    """
    dist = 10*10**(dmod/5)
    mstars = np.logspace(4, 9)
    mu_avge_v = (np.log10(mstars)-19.23)/-0.51
    lg_r_e = 0.23*np.log10(mstars) - 1.93
    r_e = (10**lg_r_e*1000)/dist * 206265

    # mu_avge_v = mu_e_v# - 0.699
    Mv = mu_avge_v - 2.5*np.log10(2*np.pi*r_e**2) - dmod
    Mr = Mv - 0.4

    mu0_v = (mu_avge_v+0.699) - 1.822
    mu0_r = mu0_v - 0.4
    ax.plot(mu0_r, Mr, '-', color='darkorchid', lw=3)


def completeness_plot(sims, dmod, mag='m_g', mu='mu_0_g',
                      xlabel=r'$\mu_0(g)\,[\mathrm{mag\;arcsec}^{-2}]$', ylabel=r'$M_g$', labelsize=19,
                      xbins=10, ybins=10, re_col='LimeGreen'):
    """
    Completeness plot from Scott's paper
    """
    from matplotlib.ticker import (AutoMinorLocator)

    left, width = 0.15, 0.55
    bottom, height = 0.15, 0.55
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    fig = plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    bins1 = np.unique(sims[mag])
    bins2 = np.unique(sims[mu])
    bins1 = np.linspace(bins1[0]-0.5*(bins1[1]-bins1[0]),
                        bins1[-1]+0.5*(bins1[1]-bins1[0]),
                        len(bins1)+1)
    bins2 = np.linspace(bins2[0]-0.5*(bins2[1]-bins2[0]),
                        bins2[-1]+0.5*(bins2[1]-bins2[0]),
                        len(bins2)+1)
    # print(bins1, bins2)
    # range1 = [np.min(sims[mag]), np.max(sims[mag])]
    # range2 = [np.min(sims[mu]), np.max(sims[mu])]
    # bins1 = np.linspace(*range1, xbins)
    # bins2 = np.linspace(*range2, ybins)
    # print(bins1, bins2)

    detected = sims[sims['match'] > 0]
    dets, b1, b2 = np.histogram2d(detected[mag], detected[mu], [bins1, bins2])
    tot, b1, b2 = np.histogram2d(sims[mag], sims[mu], [bins1, bins2])
    img = dets/tot
    img[np.isnan(img)] = 0
    img = axScatter.imshow(img, origin='lower',
                           extent=[bins2[0], bins2[-1], bins1[0], bins1[-1]],
                           cmap='magma', vmin=0, vmax=1)
    axScatter.set_ylabel(ylabel, fontsize=labelsize)
    axScatter.set_xlabel(xlabel, fontsize=labelsize)

    axScatter.xaxis.set_minor_locator(AutoMinorLocator())
    axScatter.yaxis.set_minor_locator(AutoMinorLocator())
    cbaxes = plt.gcf().add_axes([left, bottom-0.12, width, 0.02])
    cb = plt.colorbar(img, cax=cbaxes, orientation="horizontal",
                      label=r'$\mathrm{Detection\;Efficiency}$')

    # Marginalized histograms
    axHisty.hist(sims[mag], bins=bins1, histtype='stepfilled',
                 color='silver', orientation='horizontal')
    axHisty.hist(detected[mag], bins=bins1, histtype='stepfilled',
                 color='gray', orientation='horizontal')
    # if np.isfinite(detected['r_e_sim'][0]):
    #    axHisty.hist(detected['mag_sim']-dmod, bins=bins1, histtype='step', color='red', orientation='horizontal',lw=3)
    axHisty.set_ylim([bins1[0], bins1[-1]])
    axHisty.set_yticklabels([])
    axHisty.set_xlabel(r'$\mathrm{Number}$', fontsize=16)

    axHistx.hist(sims[mu], bins=bins2, histtype='stepfilled',
                 color='silver', label=r'$\mathrm{injected}$')
    axHistx.hist(detected[mu], bins=bins2, histtype='stepfilled',
                 color='gray', label=r'$\mathrm{recovered}$')
    # if np.isfinite(detected['r_e_sim'][0]):
    #    axHistx.hist(detected['mu0_sim'], bins=bins2, histtype='step', color='red',lw=3, label=r'$\mathrm{measured}$')
    axHistx.set_xlim([bins2[0], bins2[-1]])
    axHistx.set_xticklabels([])
    axHistx.set_ylabel(r'$\mathrm{Number}$', fontsize=16)

    axHistx.xaxis.set_minor_locator(AutoMinorLocator())
    axHistx.yaxis.set_minor_locator(AutoMinorLocator())
    axHisty.xaxis.set_minor_locator(AutoMinorLocator())
    axHisty.yaxis.set_minor_locator(AutoMinorLocator())

    axHistx.legend(bbox_to_anchor=(1.01, 1), loc=2,
                   borderaxespad=0., frameon=False)

    axScatter.set_ylim([bins1[0], bins1[-1]])
    axScatter.set_xlim([bins2[0], bins2[-1]])

    plot_radii(axScatter, bins1, bins2, dmod, re_col=re_col)
    # draw_lg_trends(axScatter, dmod)

    return fig
