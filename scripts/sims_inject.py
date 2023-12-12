# will add the fake galaxies here
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import astropy.io.fits as fits
from astropy.table import Table, Column
import os
from astropy import wcs
import subprocess
from scipy.signal import medfilt as medfilt
import multiprocessing as mp
import matplotlib.patheffects as path_effects
from object_download import host_list
from decals_sims import junk2
from decals_sims import find_ind


px_sc = 0.262
dat_dir = '/scratch/gpfs/jiaxuanl/Data/SALAD/'


def add_gal_single_field(tracti, tractg, imag_bins, mu0_bins, work_fil, gals_per_field, dist, name):
    tractname = tracti.split('-')[-3]
    if 'p' in tractname:
        np.random.seed(int(tractname.split('p')[0])*int(tractname.split('p')[1]))
    else:
        np.random.seed(int(tractname.split('m')[0])*int(tractname.split('m')[1]))
    tracti = fits.open(tracti)
    tractg = fits.open(tractg)
    w = wcs.WCS(tracti[0].header)
    imshap = np.shape(tracti[0].data)

    # gal_props = np.zeros((gals_per_field, 12))
    dmod = 5.*np.log10(dist*10**5)

    for ii in range(gals_per_field):
        mag = np.random.choice(imag_bins)
        mu0 = np.random.choice(mu0_bins)
        col = np.random.uniform(0.3,0.6,size=1)
        mue = mu0 + 1.822
        mu_avge = mue - 0.699
        reff = np.sqrt(10**((mu_avge-mag-dmod)/2.5)/2/np.pi)
        xcen = int(np.random.uniform(5*reff/px_sc, imshap[1]-5*reff/px_sc))
        ycen = int(np.random.uniform(5*reff/px_sc, imshap[0]-5*reff/px_sc))
        crds = w.all_pix2world([[xcen, ycen]], 1)[0]
        #in order are ra, dec, m (r or i), m (g), mu0, mue, mu_avge, reff (arcsec), tract_ra, tract_dec, x, y
        gal_props[ii,0] = crds[0]
        gal_props[ii,1] = crds[1]
        gal_props[ii,2] = mag
        gal_props[ii,3] = mag + col
        gal_props[ii,4] = mu0
        gal_props[ii,5] = mue
        gal_props[ii,6] = mu_avge
        gal_props[ii,7] = reff
        if 'p' in tractname:
            gal_props[ii,8] = int(tractname.split('p')[0])
            gal_props[ii,9] = int(tractname.split('p')[1])
        elif 'm' in tractname:
            gal_props[ii,8] = int(tractname.split('m')[0])
            gal_props[ii,9] = -int(tractname.split('m')[1])
        gal_props[ii, 10] = xcen
        gal_props[ii, 11] = ycen


        xx, yy = np.meshgrid(np.arange(-4*reff/px_sc, 4*reff/px_sc),np.arange(-4*reff/px_sc, 4*reff/px_sc))
        dims = np.shape(xx)
        if dims[0] > 1000:
            xx, yy = np.meshgrid(np.arange(-2*reff/px_sc, 2*reff/px_sc),np.arange(-2*reff/px_sc, 2*reff/px_sc))
            dims = np.shape(xx)

        rad_dists = np.sqrt(xx**2+yy**2)*px_sc/(reff/1.678)

        gal_im_i = gal_props[ii,4] + 1.0857*rad_dists
        gal_im_g = gal_props[ii,4] + gal_props[ii,3] - gal_props[ii,2] + 1.0857*rad_dists

        gal_im_i = 10**((gal_im_i-22.5)/-2.5) * px_sc**2
        gal_im_g = 10**((gal_im_g-22.5)/-2.5) * px_sc**2
        tracti[0].data[ycen-dims[0]//2:ycen-dims[0]//2+dims[0], xcen-dims[1]//2:xcen-dims[1]//2+dims[1]] += gal_im_i
        tractg[0].data[ycen-dims[0]//2:ycen-dims[0]//2+dims[0], xcen-dims[1]//2:xcen-dims[1]//2+dims[1]] += gal_im_g


    tracti.writeto(work_fil + 'decals/'+name+'/tracts_sims/legacysurvey-'+tractname+'-image-'+'r'+'.fits')
    tractg.writeto(work_fil + 'decals/'+name+'/tracts_sims/legacysurvey-'+tractname+'-image-'+'g'+'.fits')

    return gal_props



def inject_galaxies(name, ncores, work_fil, gals_per_field, mag1, mu1, dist, nfields, idx=0):


    proc=subprocess.Popen('ls -d '+dat_dir+'decals/'+name+'/tracts/legacysurvey-*-r.fits', shell=True, stdout=subprocess.PIPE, )
    indexlists=proc.communicate()[0]
    directories  = indexlists.split()
    tracts = [i.decode("utf-8") for i in directories]
    tracts = [i.split('-')[-3] for i in tracts]
    if nfields is not None:
        tracts = tracts[10:nfields+10]

    print(tracts, len(tracts))

    if os.path.exists(work_fil+'decals/'+name) == False:
        os.system('mkdir ' + work_fil + 'decals/'+name)
        os.system('mkdir ' + work_fil + 'decals/'+name+'/tracts_sims/')
        os.system('mkdir ' + work_fil + 'decals/'+name+'/search_output/')
        os.system('mkdir ' + work_fil + 'decals/'+name+'/search_output/cands')
        os.system('mkdir ' + work_fil + 'decals/'+name+'/search_output/stars')
        os.system('mkdir ' + work_fil + 'decals/'+name+'/search_output/simfits')
        os.system('mkdir ' + work_fil + 'decals/'+name+'/tracts_sims/det_tmp')
    else:
        os.system('rm ' + work_fil + 'decals/'+name+'/tracts_sims/*.fits')
        os.system('rm ' + work_fil + 'decals/'+name+'/tracts_sims/det_tmp/*')

    imag_bins = np.linspace(mag1, mag1+9*0.5, 10)
    mu0_bins = np.linspace(mu1, mu1+9*0.5, 10)

    p = mp.Pool(processes=ncores)
    res = []
    for i in range(0, len(tracts)):
        tractnamei = tracts[i] + '-' + 'r'
        tractnameg = tracts[i] + '-' + 'g'
        tracti = dat_dir + 'decals/' + name + '/tracts/legacysurvey-'+tracts[i]+'-image-'+'r'+'.fits'
        tractg = dat_dir + 'decals/' + name + '/tracts/legacysurvey-'+tracts[i]+'-image-'+'g'+'.fits'
        if ncores > 1:
            res.append(p.apply_async(add_gal_single_field, args=(tracti, tractg, imag_bins, mu0_bins, work_fil, gals_per_field, dist, name)))
        else:
            res.append(add_gal_single_field(tracti, tractg, imag_bins, mu0_bins, work_fil, gals_per_field, dist, name))
    p.close()
    p.join()

    if ncores>1:
        res = [x.get() for x in res]
    tot_props = res[0]
    for i in range(1,len(res)):
        tot_props = np.append(tot_props, res[i], axis=0)

    np.savetxt(dat_dir+'decals/'+name+'/sim_results/gal_props_' + str(idx), tot_props)







if __name__ == '__main__':

    names = ['ngc5457']



    for name in names:
        changes = host_list()
        ind = find_ind(changes, name)
        d = changes[ind]
        dist = d['dist']

        changes = junk2()
        ind = find_ind(changes, name)
        d = changes[ind]
        gals_per_field = d['gals_p_field']
        mag1 = d['mag1']
        mu1 = d['mu1']
        use_scratch = d['use_scratch']
        niter = d['niter']
        if 'nfields' in d:
            nfields = d['nfields']
        else:
            nfields = None

        ncores = 8

        if use_scratch:
            work_fil = '/home/sgc/s/'
        else:
            work_fil = dat_dir


        inject_galaxies(name, ncores, work_fil, gals_per_field, mag1, mu1, dist, nfields)
