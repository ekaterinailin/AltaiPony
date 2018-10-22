import numpy as np
import pandas as pd
import datetime
import logging
import copy
import random
from scipy.stats import binned_statistic

LOG = logging.getLogger(__name__)


def inject_fake_flares(lc, mode='hawley2014', gapwindow=0.1, fakefreq=.25,
                        inject_before_detrending=False):

    '''
    Create a number of events, inject them in to data
    Use grid of amplitudes and durations, keep ampl in relative flux units
    Keep track of energy in Equiv Dur.
    Duration defined in minutes
    Amplitude defined multiples of the median error


    Parameters:
    -------------
    lc: FlareLightCurve
        contains info about flare start and stop in lc.flares
    mode : 'hawley2014' or 'rand'
        de-trending mode
    gapwindow : 0.1 or float

    fakefreq : .25 or float
        flares per day
    inject_before_detrending : True or bool
        By default, flares are injected before the light curve is detrended.
    Returns:
    ------------
    FlareLightCurve with fake flare signatures

    '''

    def _equivalent_duration(time, flux):
        '''
        Compute the Equivalent Duration of a fake flare.
        This is the area under the flare, in relative flux units.

        Parameters:
        -------------
        time : numpy array
            units of DAYS
        flux : numpy array
            relative flux units
        Return:
        ------------
        p : float
            equivalent duration of a single event in units of seconds
        '''
        x = time * 60.0 * 60.0 * 24.0
        integral = np.sum(np.diff(x) * flux[:-1])
        return integral


    LOG.debug(str() + '{} FakeFlares started'.format(datetime.datetime.now()))
    if inject_before_detrending == True:
        typ, typerr = 'flux', 'flux_err'
        LOG.info('Injecting before detrending.')
    elif inject_before_detrending == False:
        typ, typerr = 'detrended_flux', 'detrended_flux_err'
        LOG.info('Injecting after detrending.')
    fakeres = pd.DataFrame()
    fake_lc = copy.deepcopy(lc)
    medflux = np.nanmedian(fake_lc.__dict__[typ])
    fake_lc.__dict__[typ] = fake_lc.__dict__[typ]/medflux -1.
    fake_lc.__dict__[typerr] = fake_lc.__dict__[typerr]/medflux
    nfakesum = int(np.rint(fakefreq * (lc.time.max() - lc.time.min())))
    t0_fake = np.zeros(nfakesum, dtype='float')
    ed_fake = np.zeros(nfakesum, dtype='float')
    dur_fake = np.zeros(nfakesum, dtype='float')
    ampl_fake = np.zeros(nfakesum, dtype='float')
    ckm = 0
    for (le,ri) in fake_lc.gaps:
        gap_fake_lc = fake_lc[le:ri]
        nfake = int(np.rint(fakefreq * (gap_fake_lc.time.max() - gap_fake_lc.time.min())))

        LOG.debug('Inject {} fake flares into a {} datapoint long array.'.format(nfake,ri-le))

        real_flares_in_gap = lc.flares[(lc.flares.istart >= le) & (lc.flares.istop <= ri)]
        error = gap_fake_lc.__dict__[typerr]
        flux = gap_fake_lc.__dict__[typ]
        time = gap_fake_lc.time
        distribution  = generate_fake_flare_distribution(np.nanmedian(error),
                                                         nfake,
                                                         mode=mode)
        dur_fake[ckm:ckm+nfake], ampl_fake[ckm:ckm+nfake] = distribution
        #loop over the numer of fake flares you want to generate
        for k in range(ckm, ckm+nfake):
    	    # generate random peak time, avoid known flares
    	    isok = False
    	    while isok is False:
    	        # choose a random peak time
    	        t0 =  random.uniform(np.min(time),np.max(time))
                # Are there any real flares to deal with?
    	        if real_flares_in_gap.tstart.shape[0]>0:
                    # Are there any real flares happening at peak time?
                    # Fake flares should not overlap with real ones.
                    b = ( real_flares_in_gap[(t0 >= real_flares_in_gap.tstart) &
                                             (t0 <= real_flares_in_gap.tstop)].
                                            shape[0] )
                    if b == 0:
                        isok = True
    	        else:
                    isok = True
    	        t0_fake[k] = t0
    	        fl_flux = aflare(time, t0, dur_fake[k], ampl_fake[k])
    	        ed_fake[k] = _equivalent_duration(time, fl_flux)
                    	    # inject flare in to light curve
    	    fake_lc.__dict__[typ][le:ri] = fake_lc.__dict__[typ][le:ri] + fl_flux
        ckm += nfake

    #error minimum is a safety net for the spline function if mode=3
    fake_lc.__dict__[typerr] = max( 1e-10, np.nanmedian( pd.Series(fake_lc.__dict__[typ]).
                                              rolling(3, center=True).
                                              std() ) )*np.ones_like(fake_lc.__dict__[typ])

    injected_events = {'duration_d' : dur_fake, 'amplitude' : ampl_fake,
                       'ed_inj' : ed_fake, 'peak_time' : t0_fake}
    fake_lc.fake_flares = fake_lc.fake_flares.append(pd.DataFrame(injected_events),
                                                     ignore_index=True,
                                                     sort=True)

    return fake_lc

def generate_fake_flare_distribution(std, nfake, ampl=(5e-1,5e2), dur=(5e-1,2e2),
                                     mode='hawley2014', scatter=False):

    '''
    Creates different distributions of fake flares to be injected into light curves.

    rand: Flares are distibuted evenly in dur and ampl
    hawley2014: Flares are distributed in a strip around to a power law with
    exponent alpha, see fig. 10 in Hawley et al. 2014

    Parameters:
    -----------
    std: standard deviation of quiescent light curve
    nfake: number of fake flares to be created
    ampl: amplitude range in relative (only for 'rand' mode) flux units,
          default=(5e-1,5e3) for consistency with 'hawley2014'
    dur: duration range (only for 'rand' mode) in minutes,
         default=(5e-1,2e2) for consistency with 'hawley2014'
    mode: distribution of fake flares in (duration,amplitude) space,
          default='hawley2014'
    scatter: saves a scatter plot of the distribution for the injected sample,
             default='False'

    Returns:
    -------
    dur_fake: durations of generated fake flares in days
    ampl_fake: amplitudes of generated fake flares in relative flux units
    '''

    if mode=='rand':

        dur_fake =  (np.random.random(nfake) * (dur[1] - dur[0]) + dur[0])
        ampl_fake = (np.random.random(nfake) * (ampl[1] - ampl[0]) + ampl[0])*std
        lndur_fake = np.log10(dur_fake)
        lnampl_fake = np.log10(ampl_fake)
        dur_fake = dur_fake / 60. / 24.

    elif mode=='hawley2014':

        c_range=np.array([np.log10(5)-6.,np.log10(5)-4.]) #estimated from fig. 10 in Hawley et al. 2014
        alpha=2.                                        #estimated from fig. 10 in Hawley et al. 2014
        ampl=(np.log10(2.*std),np.log10(10000.*std))

        lnampl_fake = (np.random.random(nfake) * (ampl[1] - ampl[0]) + ampl[0])
        lndur_fake=np.zeros(nfake, dtype='float')
        rand=np.random.random(nfake)
        dur_max = (1./alpha) * (lnampl_fake-c_range[0]) #log(tmax)
        dur_min = (1./alpha) * (lnampl_fake-c_range[1]) # log(tmin)

        for a in range(0,nfake):
            lndur_fake[a]= (rand[a] * (dur_max[a] - dur_min[a]) + dur_min[a])

        ampl_fake = np.power(np.full(nfake,10), lnampl_fake)
        dur_fake=np.power(np.full(nfake,10), lndur_fake) / 60. / 24.

    return dur_fake, ampl_fake


def aflare(t, tpeak, dur, ampl, upsample=False, uptime=10):
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723

    Use this function for fitting classical flares with most curve_fit
    tools.

    Note: this model assumes the flux before the flare is zero centered

    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    dur : float
        The duration of the flare
    ampl : float
        The amplitude of the flare
    upsample : bool
        If True up-sample the model flare to ensure more precise energies.
    uptime : float
        How many times to up-sample the data (Default is 10)

    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    fwhm = dur/2. # crude approximation for a triangle shape, should be even less

    if upsample:
        dt = np.nanmedian(np.diff(t))
        timeup = np.linspace(min(t)-dt, max(t)+dt, t.size * uptime)

        flareup = np.piecewise(timeup, [(timeup<= tpeak) * (timeup-tpeak)/fwhm > -1.,
                                        (timeup > tpeak)],
                                    [lambda x: (_fr[0]+                       # 0th order
                                                _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                                _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                                _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                                _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                                     lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                                _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                                    ) * np.abs(ampl) # amplitude

        # and now downsample back to the original time...
        ## this way might be better, but makes assumption of uniform time bins
        # flare = np.nanmean(flareup.reshape(-1, uptime), axis=1)

        ## This way does linear interp. back to any input time grid
        # flare = np.interp(t, timeup, flareup)

        ## this was uses "binned statistic"
        downbins = np.concatenate((t-dt/2.,[max(t)+dt/2.]))
        flare,_,_ = binned_statistic(timeup, flareup, statistic='mean',
                                     bins=downbins)

    else:
        print(t,tpeak,fwhm)
        flare = np.piecewise(t, [(t<= tpeak) * (t-tpeak)/fwhm > -1.,
                                 (t > tpeak)],
                                [lambda x: (_fr[0]+                       # 0th order
                                            _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                            _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                            _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                            _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                                 lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                            _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                                ) * np.abs(ampl) # amplitude

    return flare

def merge_fake_and_recovered_events(injs, recs):
    """
    Helper function that merges the DataFrames containing injected fake flares
    with the recovered events.

    Parameters
    -----------
    injs : DataFrame
        injected flares
    recs : DataFrame
        recovered flares

    Return
    ------
    DataFrame with both recovered and unrecovered events. The former contain
    additional info about recovered energy and captured datapoints.
    """
    recs['temp'] = 1
    injs['temp'] = 1
    merged = injs.merge(recs,how='outer')
    merged_recovered = merged[(merged.tstart < merged.peak_time) & (merged.tstop > merged.peak_time)]
    rest = injs[~injs.amplitude.isin(merged_recovered.amplitude.values)]
    merged_all = merged_recovered.append(rest,sort=True).drop('temp',axis=1)
    return merged_all

    #
    # # Create a test lightcurve with flares here:
    # # out_lc = pd.DataFrame({'flux_raw':fake_lc.__dict__[typ]*medflux,'time':lc.time,
    # #                        'error':max(1e-10,np.nanmedian(pd.Series(fake_lc.__dict__[typ]*medflux).rolling(3, center=True).std())),
    # #                        'flags':lc.flags})
    # # out_lc.to_csv('test_suite/test/testlc.csv')
    # istart, istop, new_lc['flux_model'] = MultiFind(new_lc, dlr, mode=mode,
    #                                       gapwindow=gapwindow)
    #
    # h = {'ed_fake':ed_fake,
    #           'rec_fake': np.zeros(nfakesum) ,'ed_rec':np.zeros(nfakesum),
    #           'ed_rec_err':np.zeros(nfakesum),'istart_rec':np.zeros(nfakesum),
    #           'istop_rec':np.zeros(nfakesum)}
    #
    # if len(istart)>0: # in case no flares are recovered, even after injection
    #
    #     dfh = pd.DataFrame({'tr':lc.time[istart].values,'ir':lc.time[istart].index.values,
    #                         'tp':lc.time[istop].values,'ip':lc.time[istop].index.values})
    #     for k in range(nfake): # go thru all recovered flares
    #         # do any injected flares overlap recovered flares?
    #         rb = dfh[(t0_fake[k] >= dfh.tr) & (t0_fake[k] <= dfh.tp)]
    #         if rb.empty==False:
    #             rb = rb.iloc[0]
    #             h['rec_fake'][k] = 1
    #             h['ed_rec'][k], h['ed_rec_err'][k] = help.ED(rb.ir, rb.ip,
    #                                                          new_lc, err=True)
    #             h['istart_rec'][k], h['istop_rec'][k] = rb.ir, rb.ip
    #             istart = np.delete(istart,rb.ir)
    #             istop = np.delete(istop,rb.ip)
    #
    # df = pd.DataFrame(h)
    # del h
    # fakeres = fakeres.append(df, ignore_index=True)
    #
    # if display == True:
    #     print('Display fake flare injection')
    #     fig, ax = plt.subplots(figsize=(10,4))
    #     ax = help.Plot(new_lc, ax, istart=istart, istop=istop, onlybit=20.)
    #     plt.show()
    #     #plt.savefig('{}_fakes_injected.png'.format(outfile), dpi=300)
    #     plt.close()
    #
    # if savefile is True:
    #     header = ['min_time','max_time','std_dev','nfake','min_amplitude',
    #               'max_amplitude','min_duration','max_duration','tstamp',]
    #     if glob.glob(outfile)==[]:
    #         dfout = pd.DataFrame()
    #     else:
    #         dfout = pd.read_json(outfile)
    #
    #     tstamp = clock.asctime(clock.localtime(clock.time()))
    #     outrow = [[item] for item in [min(time), max(time), std, nfake, min(ampl_fake),
    #                                   max(ampl_fake),min(dur_fake),max(dur_fake),tstamp]]
    #     dfout = dfout.append(pd.DataFrame(dict(zip(header,outrow))),
    #                              ignore_index=True)
    #     dfout.to_json(outfile)
    #
    #     fakeres : pandas DataFrame
    #
    #
    #         # Injection/recovery results with columns 'ed_fake' (float, injected EDs),
    #         # 'rec_fake' (bool, recovered or not), 'ed_rec' (float, recovered ED),
    #         # 'ed_rec_err' (float, uncertainty of recovered ED),
    #         # 'istart_rec', 'istop_rec' (int, locations of recovered fake flares)
    #
    # return fakeres
