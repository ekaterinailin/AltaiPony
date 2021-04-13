from k2sc.core import *
from k2sc.k2io import DataReader
from k2sc.k2data import K2Data
from k2sc.ls import fasper
from numpy import *
import math as mt
from time import time, sleep
from k2sc.cdpp import cdpp
from k2sc.de import DiffEvol
from numpy.random import normal
import warnings
from collections import namedtuple

from k2sc.detrender import Detrender
from k2sc.kernels import kernels, BasicKernel, BasicKernelEP, QuasiPeriodicKernel, QuasiPeriodicKernelEP


# ALTAIPONY MOD: Here is where I use my own modification
from altaipony.utils import medsig, sigma_clip
from k2sc.standalone import psearch

import lightkurve


def detrend(dataset,campaign=5,splits=None,quiet=False,save_dir='.',seed=0,flux_type='pdc',default_position_kernel='SqrExp',
           kernel=None, kernel_period=None,p_mask_center=None,p_mask_period=None,p_mask_duration=None,
           tr_nrandom=400,tr_nblocks=6,tr_bspan=50,de_npop=100,de_niter=150,de_max_time=300,
           ls_max_fap=-50,ls_min_period=0.05,ls_max_period=25,  max_sigma=5,outlier_mwidth=25):
    '''This is a function to detrend a single k2sc dataset, outside of the framework of 
    logging that is in bin/k2sc, but duplicating the functionality of the
    local function there 'detrend'. This is here to permit access to k2sc from lightkurve, or other
    similar light curve wrapping packages that might like to use k2sc.

    Arguments

    campaign: Number of the K2 campaign.
    splits: time values to split the light curve into separate segments for systematics - if you have a campaign, it will choose default splits from default_splits.
    quiet: does not currently do anything.
    save_dir: directory to save results. Default is current directory.
    seed: seed for random number draws for optimization.
    flux_type: either 'sap' or 'pdc'. Use either the Simple Aperture Photometry or the Presearch Data Conditioning (ie Kepler pipeline corrected) lightcurve as input.
    default_position_kernel: which kernel should we use for the GP in position? Defaults to squared exponential.
    kernel: set a time kernel explicitly. If this is None, then it finds this automatically by checking for periodicity and choosing either a periodic or a basic (squared exponential) kernel. \
        options are 'quasiperiodic', 'basic_ep', 'periodic', 'quasiperiodic_ep', 'basic'.
    kernel_period: if you are using a kernel with periodicity, this lets you set the periodicity manually.
    p_mask_center, p_mask_period, ,p_mask_duration: mask a planet transit out from the GP fitting by setting its zero-epoch, period and duration. All three must be set.
    tr_random, tr_nblocks, tr_bspan: When training the GP, the light curve is broken up into random blocks. These give the number of random samples, number of sample blocks, and span of a single block.\
        Do not change this unless you know what you are doing - but it can be useful if you have very long or short light curves and are getting errors.
    de_npop, de_niter, de_max_time: settings for the Differential Evolution global optimizer. Cut down one or all of these numbers to spend less time searching for a global optimum - at your peril.
    ls_max_fap, ls_min_period, ls_max_period: In searching for a period with a Lomb-Scargle periodogram, this sets the maximum Lomb-Scargle log10(false alarm) threshold to use the periodic kernel, \
        and min and max periods to search.
    outlier_sigma, outlier_mwidth: The sigma and window width to be used in outlier clipping.

    '''

    ## Setup the logger
    ## ----------------
    ds   = dataset
    Result  = namedtuple('SCResult', 'detrender pv tr_time tr_position cdpp_r cdpp_t cdpp_c warn')
    results = []  # a list of Result tuples, one per aperture
    masks   = []  # a list of light curve masks, one per aperture 


    ## Define the splits
    ## -----------------

    default_splits = {3:[2154,2190], 4:[2240,2273], 5:[2344], 6:[2390,2428], 7:[2468.5,2515],8:[2579,2598.5],102:[2778],11:[2830],12:[2915,2951],
                      13:[2998,3033],14:[3085,3123.75],15:[3170,3207.5],16:[3297.5,3331],17:[3367,3400],18:[3425,3460]}
    
    # ALTAIPONY MOD
    # ------------------------------------------------------
    if campaign is not None:
        splits = default_splits[campaign]
    elif splits is None and ds.campaign not in default_splits.keys():
        print('The campaign not known and no splits given.')
    elif splits is not None:
        splits = splits
    elif splits is None and ds.campaign not in default_splits.keys():
        print('The campaign not known and no splits given.')
    # ------------------------------------------------------

    ## Periodic signal masking
    ## -----------------------
    if p_mask_center and p_mask_period and p_mask_duration:
        ds.mask_periodic_signal(p_mask_center, p_mask_period, p_mask_duration)

    ## Initial outlier and period detection
    ## ------------------------------------
    ## We carry out an initial outlier and period detection using
    ## a default GP hyperparameter vector based on campaign 4 fits
    ## done using (almost) nonprintrmative priors.

    for iset in range(ds.nsets):
        flux = ds.fluxes[iset]
        mask  = isfinite(flux) 
        mask &= ~(ds.mflags[iset] & M_PERIODIC).astype(bool)  # Apply the transit mask, if any
        mask &= ~(ds.quality & 2**20).astype(bool)            # Mask out the thruster firings
        inputs = transpose([ds.time, ds.x, ds.y])
        masks.append(mask)

        detrender = Detrender(flux, inputs, mask = mask, splits = splits,
                              kernel = BasicKernelEP(),
                              tr_nrandom = tr_nrandom,
                              tr_nblocks = tr_nblocks,
                              tr_bspan = tr_bspan)
    
        ttrend,ptrend = detrender.predict(detrender.kernel.pv0+1e-5, components=True)
        cflux = flux-ptrend+median(ptrend)-ttrend+median(ttrend)
        cflux /= nanmedian(cflux)

        ## Iterative sigma-clipping
        ## ------------------------
        # ALTAIPONY MOD
        # ------------------------------------------------------
        print('Starting initial outlier detection at ' + str(max_sigma) + 'sigma.')
        omask = mask & sigma_clip(cflux, max_iter=10, max_sigma=max_sigma, mexc=mask)
        # ------------------------------------------------------
        ofrac = (~omask).sum() / omask.size
        if ofrac < 0.25:
            mask &= omask
            print('  Flagged %i (%4.1f%%) outliers.' % ((~omask).sum(), 100*ofrac))
        else:
            print('  Found %i (%4.1f%%) outliers. Not flagging.' % ((~omask).sum(), 100*ofrac))

        ## Lomb-Scargle period search
        ## --------------------------
        if ofrac < 0.9:
            print('Starting Lomb-Scargle period search')
            nflux = flux - ptrend + nanmedian(ptrend)
            ntime = ds.time - ds.time.mean()
            pflux = poly1d(polyfit(ntime[mask], nflux[mask], 9))(ntime)

            period, fap = psearch(ds.time[mask], (nflux-pflux)[mask], ls_min_period, ls_max_period)
        
            if fap < 1e-50:
                ds.is_periodic = True
                ds.ls_fap    = fap
                ds.ls_period = period
        else:
            print('Too many outliers, skipping the Lomb-Scargle period search')

    ## Kernel selection
    ## ----------------
    if kernel:
        print('Overriding automatic kernel selection, using %s kernel as given in the command line' % kernel)
        if 'periodic' in kernel and not kernel_period:
            print('Need to give period (--kernel-period) if overriding automatic kernel detection with a periodic kernel. Quitting.')
            return 0
        kernel = kernels[kernel](period=kernel_period)
    else:
        print('  Using %s position kernel' % default_position_kernel)
        if ds.is_periodic:
            print('  Found periodicity p = {:7.2f} (fap {:7.4e} < 1e-50), will use a quasiperiodic kernel'.format(ds.ls_period, ds.ls_fap))
        else:
            print('  No strong periodicity found, using a basic kernel')

        if default_position_kernel.lower() == 'sqrexp':
            kernel = QuasiPeriodicKernel(period=ds.ls_period)   if ds.is_periodic else BasicKernel() 
        else:
            kernel = QuasiPeriodicKernelEP(period=ds.ls_period) if ds.is_periodic else BasicKernelEP()


    ## Detrending
    ## ----------
    for iset in range(ds.nsets):
        if ds.nsets > 1:
            name = 'Worker {:d} <{:d}-{:d}>'.format(mpi_rank, dataset.epic, iset+1)
        random.seed(seed)
        tstart = time()
        
        inputs = transpose([ds.time,ds.x,ds.y])
        detrender = Detrender(ds.fluxes[iset], inputs, mask=masks[iset],
                              splits=splits, kernel=kernel, tr_nrandom=tr_nrandom,
                              tr_nblocks=tr_nblocks, tr_bspan=tr_bspan)
        de = DiffEvol(detrender.neglnposterior, kernel.bounds, de_npop)

        ## Period population generation
        ## ----------------------------
        if isinstance(kernel, QuasiPeriodicKernel):
            de._population[:,2] = clip(normal(kernel.period, 0.1*kernel.period, size=de.n_pop),
                                          ls_min_period, ls_max_period)
        ## Hyperparameter optimisation
        ## ---------------------------
        if isfinite(ds.fluxes[iset]).sum() >= 100:
            ## Global hyperparameter optimisation
            ## ----------------------------------
            print('Starting global hyperparameter optimisation using DE')
            tstart_de = time()
            for i,r in enumerate((de(de_niter))):
                print('  DE iteration %3i -ln(L) %4.1f', i, de.minimum_value)
                tcur_de = time()
                if ((de._fitness.ptp() < 3) or (tcur_de - tstart_de > de_max_time)) and (i>2):
                    break
            print('  DE finished in %i seconds', tcur_de-tstart_de)
            print('  DE minimum found at: %s', array_str(de.minimum_location, precision=3, max_line_width=250))
            print('  DE -ln(L) %4.1f', de.minimum_value)

            ## Local hyperparameter optimisation
            ## ---------------------------------
            print('Starting local hyperparameter optimisation')
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
                    pv, warn = detrender.train(de.minimum_location)
            except ValueError as e:
                print('Local optimiser failed, %s' % e)
                print('Skipping the file')
                return
            print('  Local minimum found at: %s', array_str(pv, precision=3))

            ## Trend computation
            ## -----------------
            (mt,tt),(mp,tp) = map(lambda a: (nanmedian(a), a-nanmedian(a)), detrender.predict(pv, components=True))

            ## Iterative sigma-clipping
            ## ------------------------
            print('Starting final outlier detection')
            flux = detrender.data.unmasked_flux
            cflux = flux-tp-tt
            cflux /= nanmedian(cflux)

            mper = ~(ds.mflags[iset] & M_PERIODIC).astype(bool)  # Apply the transit mask, if any
            mthf = ~(ds.quality & 2**20).astype(bool)            # Mask out the thruster firings
            minf = isfinite(cflux)
            # ALTAIPONY MOD
            # ------------------------------------------------------
            mlow, mhigh = sigma_clip(cflux, max_iter = 10, max_sigma = max_sigma, 
                                     separate_masks = True, mexc = mper&mthf)
            # ------------------------------------------------------
            ds.mflags[iset][~minf]  |= M_NOTFINITE
            ds.mflags[iset][~mhigh]  |= M_OUTLIER_U
            ds.mflags[iset][~mlow]   |= M_OUTLIER_D

            print('  %5i too high', (~mhigh).sum())
            print('  %5i too low',  (~mlow).sum())
            print('  %5i not finite', (~minf).sum())

            ## Detrending and CDPP computation
            ## -------------------------------
            print('Computing time and position trends')
            dd = detrender.data
            cdpp_r = cdpp(dd.masked_time,   dd.masked_flux)
            cdpp_t = cdpp(dd.unmasked_time, dd.unmasked_flux-tp,    exclude=~dd.mask)
            cdpp_c = cdpp(dd.unmasked_time, dd.unmasked_flux-tp-tt, exclude=~dd.mask)
        else:
            print('Skipping dataset %i, not enough finite datapoints')
            cdpp_r, cdpp_t, cdpp_c, warn = -1, -1, -1, -1
            mt, mp = nan, nan
            tt = full_like(detrender.data.unmasked_flux, nan)
            tp = full_like(detrender.data.unmasked_flux, nan)
            pv = full(kernel.npar, nan)
            detrender.tr_pv = pv.copy()            

        result = Result(detrender, pv, tt+mt, tp+mp, cdpp_r, cdpp_t, cdpp_c, warn)
        print('  CDPP - raw - %6.3f', cdpp_r)
        print('  CDPP - position component removed - %6.3f', cdpp_t)
        print('  CDPP - full reduction - %6.3f', cdpp_c)
        print('Detrending time',time()-tstart)
        
        return result, detrender

class k2sc_lc(lightkurve.KeplerLightCurve):
    '''
    This class is a wrapper for lightkurve (github.com/KeplerGO/lightkurve) so it can call k2sc. This is a work in progress at both ends and the aim is currently to include TessLightCurve objects as well.

    Example use is shown in k2sc/notebooks/lightkurve.py, where you will want to instantiate a k2sc_lc object by

    tpf = KeplerTargetPixelFile.from_archive(212300977) # WASP-55
    lc = tpf.to_lightcurve() # load some data either as a tpf or just straight up as a lightcurve
    lc = lc.remove_nans() # don't know why the quality flags are weird
    lc.primary_header = tpf.hdu[0].header
    lc.data_header = tpf.hdu[1].header
    lc.pos_corr1 = tpf.hdu[1].data['POS_CORR1'][tpf.quality_mask]
    lc.pos_corr2 = tpf.hdu[1].data['POS_CORR2'][tpf.quality_mask]
    
    # now the magic happens
    lc.__class__ = k2sc_lc
    lc.k2sc()
    '''

    def get_k2data(self):
        try:
            x, y = self.pos_corr1.value, self.pos_corr2.value
        except:
            x, y = self.centroid_col.value, self.centroid_row.value
        dataset = K2Data(self.targetid,
                 time = self.time.value,
                      cadence = self.cadenceno.value,
                      quality = self.quality.value,
                      fluxes  = self.flux.value,
                      errors  = self.flux_err.value,
                      x       = x,
                      y       = y,
                      primary_header = self.primary_header,
                      data_header = self.data_header,
                      campaign=self.campaign)
        return dataset

    def k2sc(self,**kwargs):
        dataset = self.get_k2data()
        results, self.meta["detrender"] = detrend(dataset,campaign=self.campaign,**kwargs) # see keyword arguments from detrend above
        self["tr_position"] = results.tr_position
        self["tr_time"] = results.tr_time 
        self.meta["pv"] = results.pv # hyperparameters 
        self["corr_flux"] = self.flux.value - self.tr_position.value + nanmedian(self.tr_position.value) 
        self.meta["cdpp_r"] = results.cdpp_r
        self.meta["cdpp_t"] = results.cdpp_t
        self.meta["cdpp_c"] = results.cdpp_c
