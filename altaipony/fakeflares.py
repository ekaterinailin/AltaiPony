import numpy as np
import pandas as pd
import logging
from scipy.stats import binned_statistic
from scipy import special

LOG = logging.getLogger(__name__)


def flare_model(model, *params):
    """
    Flare model wrapper.

    Parameters
    ----------
    model : str
        Name of the flare model to use.
    params : tuple
        Parameters of the flare model.

    Returns
    -------
    array
        Flare model.
    """
    if model == 'davenport2014':
        return flare_model_davenport2014(*params)
    elif model == 'mendoza2022':
        return flare_model_mendoza2022(*params)
    else:
        raise ValueError(f'Unknown flare model: {model}')

def generate_fake_flare_distribution(nfake, ampl=[1e-4, 5], dur=[0.005, 0.012], mode="uniform", **kwargs ):

    '''
    Creates different distributions of fake flares to be injected into light curves.

    "uniform": Flares are distibuted evenly in duration and amplitude space.
   
    Parameters
    -----------
    nfake: int
        Number of fake flares to be created.
    ampl: [1e-4, 1e2] or list of floats
        Amplitude range in relative flux units.
    dur: [10, 2e4] or list of floats
        Duration range in days.
    mode: 'uniform'
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

    return dur_fake, ampl_fake

def mod_random(x, d=False, seed=667):
    """
    Helper function that generates deterministic
    random numbers if needed for testing.

    Draws from its own ``numpy.random.Generator``, so numpy's global random
    state is neither used nor changed.

    Parameters
    -----------
    x : int
        Number of random numbers, uniform in [0, 1).
    d : False or bool
        Flag to set if random numbers shall be deterministic.
    seed : 667 or int
        Seed of the generator if ``d`` is True.
    """
    rng = np.random.default_rng(seed if d else None)
    return rng.random(x)


def template_equivalent_duration(time, flux):
    """
    Equivalent duration of an injected flare template.

    Sums the relative flare flux over the time grid the same way
    ``altai.equivalent_duration`` does for a detected flare, so injected
    and recovered ED are comparable.

    Parameters
    -----------
    time : array
        Time in days.
    flux : array
        Flare flux in relative units (the template, without the quiescent
        level).

    Return
    -------
    float : equivalent duration in seconds
    """
    x = np.asarray(time) * 60.0 * 60.0 * 24.0
    return np.sum(np.diff(x) * np.asarray(flux)[:-1])

def flare_eqn(t, ampl):
    '''
    The equation that defines the shape for the Continuous Flare Model
    '''
    #Values were fit & calculated using MCMC 256 walkers and 30000 steps

    A,B,C,D1,D2,f1 = [0.9687734504375167, -0.251299705922117, 0.22675974948468916,
                      0.15551880775110513, 1.2150539528490194, 0.12695865022878844]

    # We include the corresponding errors for each parameter from the MCMC analysis

    A_err,B_err,C_err,D1_err,D2_err,f1_err = [0.007941622683556804,0.0004073709715788909,0.0006863488251125649,
                                              0.0013498012884345656,0.00453458098656645,0.001053149344530907 ]

    f2 = 1 - f1

    # Long before the peak, exp() overflows to inf while erfc() underflows to
    # 0, and inf * 0 gives NaN.  The template is exactly 0 there.
    with np.errstate(over="ignore", invalid="ignore"):
        eqn = ((1 / 2) * np.sqrt(np.pi) * A * C * f1 * np.exp(-D1 * t + ((B / C) + (D1 * C / 2)) ** 2)
                            * special.erfc(((B - t) / C) + (C * D1 / 2))) + ((1 / 2) * np.sqrt(np.pi) * A * C * f2
                            * np.exp(-D2 * t+ ((B / C) + (D2 * C / 2)) ** 2) * special.erfc(((B - t) / C) + (C * D2 / 2)))
    eqn = np.where(np.isfinite(eqn), eqn, 0.0)
    return eqn * ampl


def flare_model_mendoza2022(t, tpeak, fwhm, ampl, upsample=False, uptime=10):
    '''
    The Continuous Flare Model evaluated for single-peak (classical) flare events.
    Use this function for fitting classical flares with most curve_fit
    tools.
    References
    --------------
    Tovar Mendoza et al. (2022) DOI 10.3847/1538-3881/ac6fe6
    Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Jackman et al. (2018) https://arxiv.org/abs/1804.03377
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The center time of the flare peak
    fwhm : float
        The Full Width at Half Maximum, timescale of the flare
    ampl : float
        The amplitude of the flare
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
        A continuous flare template whose shape is defined by the convolution of a Gaussian and double exponential
        and can be parameterized by three parameters: center time (tpeak), FWHM, and ampitude
    '''

    t_new = (t - tpeak) / fwhm

    if upsample:

        dt = np.nanmedian(np.diff(np.abs(t_new)))

        timeup = np.linspace(min(t_new) - dt, max(t_new) + dt, t_new.size * uptime)

        flareup = flare_eqn(timeup, ampl)

        # and now downsample back to the original time...

        downbins = np.concatenate((t_new - dt / 2.,[max(t_new) + dt / 2.]))

        flare,_,_ = binned_statistic(timeup, flareup, statistic='mean',bins=np.sort(downbins))
    else:

        flare = flare_eqn(t_new, ampl)

    return flare 


def flare_model_davenport2014(t, tpeak, dur, ampl, upsample=False, uptime=10):
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

    fwhm = dur/2. # crude approximation for a triangle shape would be dur/2.

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

def match_flares(injected, recovered):
    """
    Match injected flares to detected flares: an injected flare is recovered
    if its peak time lies within a detection (``tstart <= peak_time <=
    tstop``).

    Parameters
    -----------
    injected : DataFrame
        injected flares, with at least a ``peak_time`` column
    recovered : DataFrame
        detected flares (``FlareLightCurve.flares``), with at least
        ``tstart`` and ``tstop`` columns

    Return
    ------
    matched : DataFrame
        ``injected`` with a column ``recovered`` (bool) and the columns of
        the matching detection (NaN if missed). If several detections
        contain the peak, the first one is used.
    false_positives : DataFrame
        the rows of ``recovered`` that contain no injected peak
    """
    recovered = recovered.reset_index(drop=True)
    tstart = recovered.tstart.to_numpy(dtype=float)
    tstop = recovered.tstop.to_numpy(dtype=float)
    rows, used = [], set()
    for _, inj in injected.iterrows():
        row = {**inj.to_dict(), "recovered": False}
        hits = np.flatnonzero((tstart <= inj.peak_time) & (inj.peak_time <= tstop))
        if len(hits) > 0:
            row.update(recovered=True, **recovered.loc[hits[0]].to_dict())
            used.update(hits)
        rows.append(row)
    columns = [*injected.columns, "recovered", *recovered.columns]
    return (pd.DataFrame(rows, columns=columns),
            recovered.drop(index=sorted(used)))
