import numpy as np
import pandas as pd
import datetime
import logging
import copy
import random
from scipy.stats import binned_statistic

LOG = logging.getLogger(__name__)


def generate_fake_flare_distribution(nfake, ampl=[1e-4, 1e2], dur=[7e-3, 2],
                                     rat=[1e-3,1e4], mode='loglog', **kwargs ):

    '''
    Creates different distributions of fake flares to be injected into light curves.

    "uniform": Flares are distibuted evenly in duration and amplitude space.
    "hawley2014": Flares are distributed in a strip around a power law with
    exponent alpha, see Fig. 10 in Hawley et al. (2014).
    "loglog":

    Parameters
    -----------
    nfake: int
        Number of fake flares to be created.
    ampl: [1e-4, 1e2] or list of floats
        Amplitude range in relative flux units.
    dur: [10, 2e4] or list of floats
        Duration range in days.
    mode: 'loglog', 'hawley2014', 'uniform_ratio', or 'uniform'
        Distribution of fake flares in (duration, amplitude) space.
    kwargs : dict
        Keyword arguments to pass to mod_random

    Return
    -------
    dur_fake: durations of generated fake flares in days
    ampl_fake: amplitudes of generated fake flares in relative flux units
    '''
    def generate_range(n, tup, **kwargs):
        return (mod_random(n, **kwargs) * (tup[1] - tup[0]) + tup[0])

    if mode=='uniform':

        dur_fake =  generate_range(nfake, dur, **kwargs)
        ampl_fake = generate_range(nfake, ampl, **kwargs)

    elif mode=='uniform_ratio':
        dur_fake =  generate_range(nfake, dur, **kwargs)
        ampl_fake = generate_range(nfake, ampl, **kwargs)
        rat_fake = ampl_fake/dur_fake
        misfit = np.where(~((rat_fake < rat[1]) & (rat_fake > rat[0])))

        while len(misfit[0]) > 0:
            dur_fake_mf =  generate_range(len(misfit[0]), dur, **kwargs)
            ampl_fake_mf = generate_range(len(misfit[0]), ampl, **kwargs)
            dur_fake[misfit] = dur_fake_mf
            ampl_fake[misfit] = ampl_fake_mf
            rat_fake = ampl_fake/dur_fake
            misfit = np.where(~((rat_fake < rat[1]) & (rat_fake > rat[0])))
            
    elif mode=='hawley2014':

        c_range = np.array([np.log10(5) - 6., np.log10(5) - 4.])                #estimated from Fig. 10 in Hawley et al. (2014)
        alpha = 2                                                               #estimated from Fig. 10 in Hawley et al. (2014)
        ampl_H14 = [np.log10(i) for i in ampl]
        lnampl_fake = (mod_random(nfake, **kwargs) * (ampl_H14[1] - ampl_H14[0]) + ampl_H14[0])
        rand = mod_random(nfake, **kwargs)
        dur_max = (1./alpha) * (lnampl_fake - c_range[0])
        dur_min = (1./alpha) * (lnampl_fake - c_range[1])
        lndur_fake = np.array([rand[a] * (dur_max[a] - dur_min[a]) +
                              dur_min[a]
                              for a in range(nfake)])
        ampl_fake = np.power(np.full(nfake,10), lnampl_fake)
        dur_fake = np.power(np.full(nfake,10), lndur_fake)

    elif mode=='loglog':
        def generate_loglog(dur, ampl, nfake):

            lnampl = [np.log10(i) for i in ampl]
            lnampl_fake = generate_range(nfake, lnampl, **kwargs)
            lndur = [np.log10(i) for i in dur]
            lndur_fake = generate_range(nfake, lndur, **kwargs)
            return lndur_fake, lnampl_fake

        lndur_fake, lnampl_fake = generate_loglog(dur, ampl, nfake)
        rat_min, rat_max = [np.log10(i) for i in rat]
        lnrat_fake = lnampl_fake-lndur_fake
        misfit = np.where(~((lnrat_fake < rat_max) & (lnrat_fake > rat_min)))
        wait = 0

        while len(misfit[0]) > 0:
            wait+=1
            lndur_misfit, lnampl_misfit = generate_loglog(dur, ampl, len(misfit[0]))
            lndur_fake[misfit] = lndur_misfit
            lnampl_fake[misfit] = lnampl_misfit
            lnrat_fake = lnampl_fake-lndur_fake
            misfit = np.where(~((lnrat_fake < rat_max) & (lnrat_fake > rat_min)))
            if wait > 100:
                LOG.exception('Generating fake flares takes too long.'
                              'Reconsider dur_factor, ampl_factor, and ratio_factor.')
                raise ValueError

        ampl_fake = np.power(np.full(nfake,10), lnampl_fake)
        dur_fake = np.power(np.full(nfake,10), lndur_fake)

    return dur_fake, ampl_fake

def mod_random(x, d=False, seed=667):
    """
    Helper function that generates deterministic
    random numbers if needed for testing.

    Parameters
    -----------
    d : False or bool
        Flag to set if random numbers shall be deterministic.
    seed : 5 or int
        Sets the seed value for random number generator.
    """
    if d == True:
        np.random.seed(seed)
        return np.random.rand(x)
    else:
        np.random.seed()#do not remove: seed is fixed otherwise!
        return np.random.rand(x)

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
    merged_all = merged_recovered.append(rest).drop('temp',axis=1)
    return merged_all

def merge_complex_flares(data):
    """
    The injection procedure sometimes introduces complex flares. These are
    recovered multiple times, according to the number of simple flare signatures
    they consist of. Merge these by adopting common times, the maximum recovered
    equivalent duration and respective error. Add injected equivalent durations.

    Parameters
    -----------
    data : DataFrame
        Columns: ['amplitude', 'cstart', 'cstop', 'duration_d', 'ed_inj', 'ed_rec',
       'ed_rec_err', 'istart', 'istop', 'peak_time', 'tstart', 'tstop','ampl_rec']

    Return
    -------
    DataFrame with the same columns as the input but with complex flares merged
    together. A new 'complex' column contains the number of simple flares
    superimposed in a given event.
    """
    data = data.fillna(0)
    size = len(data.cstart[data.cstart == 0])
    maximum = data.cstop.max()+1e9
    data.loc[data.cstart == 0.,'cstart'] = np.arange(maximum,maximum+3*size,3)
    data.loc[data.cstop == 0.,'cstop'] = np.arange(maximum+1,maximum+3*size+1,3)
    g = data.groupby(['cstart','cstop'])
    data_wo_overlaps = pd.DataFrame(columns=np.append(data.columns.values,'complex'))
    for (start, stop), d in g:
        if d.shape[0] > 1:
            row = {
            'complex' : d.shape[0],
            'peak_time' : d.peak_time[d.amplitude.idxmax()],
            'amplitude' : d.amplitude.max(),
            'cstart' : d.cstart.min(),
            'cstop' : d.cstop.max(),
            'duration_d' : d.duration_d.max(),
            'ed_inj' : d.ed_inj.sum(),
            'ed_rec' : d.ed_rec.max(),
            'ed_rec_err' : d.ed_rec_err.max(),
            'istart' : d.istart.min(),
            'istop' : d.istop.max(),
            'tstart' : d.tstart.min(),
            'tstop' : d.tstop.max(),
            'ampl_rec' : d.ampl_rec.max()}
            e = pd.DataFrame(row, index=[0])
        else:
            x = d.to_dict()
            x['complex'] = 1
            e = pd.DataFrame(x)
        data_wo_overlaps = data_wo_overlaps.append(e, ignore_index=True)
    data_wo_overlaps.loc[data_wo_overlaps.cstart >= maximum,'cstart'] = np.zeros(size)
    data_wo_overlaps.loc[data_wo_overlaps.cstop >= maximum,'cstop'] = np.zeros(size)
    return data_wo_overlaps

def recovery_probability(data, bins=30, bintype='log', fixed_bins=False):
    """
    Calculate a look-up table that returns the recovery probability of a flare
    with some true equivalent duration in seconds.

    Parameters
    -----------
    data : DataFrame
        Table with columns that contain injected equivalent duration and info
        whether this flare was recovered or not.
    bins : 30 or int
        Size of look-up table.
    bintype : 'log' or 'lin'

    fixed_bins : False or bool

    Return
    ------
    DataFrame that gives bin edges in equivalent duration and the recovery
    probability in these bins.
    """
    data['rec'] = data.ed_rec.astype(bool).astype(float)
    if fixed_bins == False:
        num = min(int(np.rint(data.shape[0]/100)), bins + 1)
    else:
        num = bins + 1
    if bintype == 'log':
        bins = np.logspace(np.log10(data.ed_inj.min()*.99),
                           np.log10(data.ed_inj.max()*1.01),
                           num=num)
    elif bintype == 'lin':
        bins = np.linspace(data.ed_inj.min(), data.ed_inj.max(), num=num)
    else:
        LOG.error('Bintype not recongnised. Use log or lin.')
    group = data.groupby(pd.cut(data.ed_inj,bins))
    rec_prob = (pd.DataFrame({'min_ed_inj' : bins[:-1],
                             'max_ed_inj' : bins[1:],
                             'mid_ed_inj' : (bins[:-1]+bins[1:])/2.,
                             'rec_prob' : group.rec.mean()})
                             .reset_index()
                             .drop('ed_inj',axis=1))

    return rec_prob

def equivalent_duration_ratio(data, bins=30, bintype='log', fixed_bins=False):
    """
    Calculate a look-up table that returns the ratio of a flare's recovered
    equivalent duration to the injected one.

    Parameters
    -----------
    data : DataFrame
        Table with columns that contain injected and recovered equivalent
        durations of synthetic flares.
    bins : 30 or int
        Maximum size of look-up table.
    bintype : 'log' or 'lin'

    fixed_bins : False or bool

    Return
    ------
    DataFrame that gives bin edges in equivalent duration and the ratio of
    equivalent durations in these bins.
    """
    d = data[data.ed_rec>0]
    d = d[['ed_inj','ed_rec']]
    d['rel'] = (d.ed_rec/d.ed_inj).astype(float)
    if fixed_bins == False:
        num = min(int(np.rint(data.shape[0]/100)), bins + 1)
    else:
        num = bins + 1

    if bintype=='log':
        bins = np.logspace(np.log10(d.ed_rec.min() * .99),
                           np.log10(d.ed_rec.max() * 1.01),
                           num=num)
    elif bintype == 'lin':
        bins = np.linspace(d.ed_rec.min(), d.ed_rec.max(), num=num)
    group = d.groupby(pd.cut(d.ed_rec,bins))
    ed_rat = (pd.DataFrame({'min_ed_rec' : bins[:-1],
                             'max_ed_rec' : bins[1:],
                             'mid_ed_rec' : (bins[:-1]+bins[1:])/2.,
                             'rel_rec' : 1/group.rel.mean()})
                             .reset_index()
                             .drop('ed_rec',axis=1)
                             .dropna(how='any'))

    return ed_rat


def resolve_complexity(data, complexity='all'):
    """
    Either deal with only simple or complex flares or ignore the difference and
    just give the fraction of complex flares in the synthetic sample.
    """
    if complexity == 'simple_only':
        data = data[data.complex == 1]
        data.loc[:,'complex_fraction'] = 0.
        return data
    elif complexity == 'complex_only':
        data = data[data.complex > 1]
        data.loc[:,'complex_fraction'] = 0.
        return data
    elif complexity == 'all':
        count_complex = data.complex.astype(float).sum()
        size = data.shape[0]
        data['complex_fraction'] = (count_complex-size)/size
        return data
