import copy
import warnings

import pandas as pd
import numpy as np

from scipy.optimize import fmin
import os

CWD = os.path.dirname(os.path.abspath(__file__))


class FFD(object):
    """Flare frequency distribution.
    alpha and beta refer to a power law that
    can be used to model the FFD.

    dN/dE = beta * E^(-alpha)

    N - number of flares
    E - energy or equivalent duration


    Attributes:
    -----------
    f : DataFrame
        flare table in the FlareLightCurve.flares format
        with extra columns for flare target identifiers
    alpha : float
        power law exponent
    alpha_err : float
        power law exponent uncertainty
    beta : float
        power law intercept
    beta_err : float
        power law intercept uncertainty
    tot_obs_time: float
        total observing time during which
        the flares in f were detected
    ID : str
        column name in f for the flare target identifier
    ed : array
        EDs in cumulative FFD, sorted
    freq : array
        frequencies of EDs in cumulative FFD, sorted like ed
    count_ed : array
        frequency adjusted ed sample
    multiple_stars : bool
        True when ed_and_freq was called with multiple_stars
        flag set


    """
    def __init__(self, f=None, alpha=None, alpha_err=None,
                 beta=None, beta_err=None, tot_obs_time=1.,
                 ID=None, multiple_stars=False):

        self.f = f
        self.alpha = alpha
        self.alpha_err = alpha_err
        self.beta = beta
        self.beta_err = beta_err
        self.tot_obs_time = tot_obs_time
        self._ed = None
        self._freq = None
        self._count_ed = None
        self.ID = ID
        self._multiple_stars = multiple_stars

    # Set all the setters and getters for attributes
    # that only methods should be allowed to change:

    @property
    def multiple_stars(self):
        return self._multiple_stars

    @multiple_stars.setter
    def multiple_stars(self, multiple_stars):
        print(f"Setting multiple_stars flag with {multiple_stars}.")
        self._multiple_stars = multiple_stars

    @property
    def ed(self):
        return self._ed

    @ed.setter
    def ed(self, ed):
        print(f"Setting ED with new values, size {len(ed)}.")
        self._ed = ed

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, freq):
        print(f"Setting frequency values with new values, size {len(freq)}.")
        self._freq = freq

    @property
    def count_ed(self):
        return self._count_ed

    @count_ed.setter
    def count_ed(self, count_ed):
        print(f"Setting frequency adjusted count values "
              f"with new values, size {len(count_ed)}.")
        self._count_ed = count_ed

# -----------------------------------------------------------------------

    def ed_and_freq(self, energy_correction=False,
                    recovery_probability_correction=False,
                    multiple_stars=False):
        """Take the flare table and return the FFD with
        different or no corrections. tot_obs_time is used to
        convert counts to frequencies and defines its unit.

        Parameters:
        ------------
        energy_correction: bool, default False
            use ed_corr instead of ed_rec
        recovery_probability_correction: bool, default False
            multiply inverse recovery probabilities instead
            of assuming the recovery probability was 1
        multiple_stars: bool, default False
            apply a first order approximation to account
            for the effects of stacking FFDs of stars with
            different detection thresholds

        Return:
        -------
        ed, freq, count_ed - equivalent durations and corresponding
                             cumulative frequencies, and frequency
                             adjusted event sample. See `_ed_and_counts`
                             method for details.
        """
        # Convert human readable cases to keywords
        if ((energy_correction is False) &
            (recovery_probability_correction is False)):
            key = "no_corr"

        elif ((energy_correction is True) &
              (recovery_probability_correction is False)):
            key = "ed_corr"

        elif ((energy_correction is True) &
              (recovery_probability_correction is True)):
            key = "edrecprob_corr"

        else:
            raise KeyError("This set of parameters for energy "
                           "correction, recovery probability "
                           "correction is not implemented. You must"
                           " set energy_correction=True if you wish to "
                           "set recovery_probability_correction=True.")

        return self._ed_and_counts(key, multiple_stars)

    def _ed_and_counts(self, key, multiple_stars):
        """Sub function to ed_and_func.

        Parameters:
        ------------
        key : str
            defines type of correction to apply to FFD
        multiple_stars: bool
            if True will use a first order approximation to
            account for stacking FFDs of multiple stars

        Return:
        -------
        ed, freq, count_ed - equivalent durations and corresponding
                             cumulative frequencies, and frequency
                             adjusted event sample
        """

        # df, ID, col are flare table, identifier column name in df,
        # and column name for the ED array in df in each of the
        # functions below.

        # Each function return two arrays: sorted flare EDs or energies,
        # and their respective frequencies.

        def cum_dist(df, col, ID):
            """simple cumulative distribution."""

            return (np.arange(1, df[col].shape[0] + 1, 1) / self.tot_obs_time,
                    np.ones_like(df[col].values))

        def get_msf_cum_dist(df, col, ID):
            """simple cumulative distribution
            accounting for multiple stars with different
            detection thresholds in FFDs"""

            freq = _get_multistar_factors(df, ID, col)
            self.multiple_stars = True
            return (np.cumsum(1 / freq) / self.tot_obs_time,
                    1 / freq)

        def cum_dist_rec_prob(df, col, ID):
            """cumulative distribution accounting for
            recovery probabilities of individual flares"""
            freq = (np.cumsum(1. / df.recovery_probability.values) /
                    self.tot_obs_time)
            return freq, 1. / df.recovery_probability.values

        def get_msf_cumdist_recprob(df, col, ID):
            """cumulative distribution accounting for
            recovery probabilities of individual flares
            and multiple stars with different detection
            thresholds in FFDs"""

            freq_ = _get_multistar_factors(df, ID, col)
            self.multiple_stars = True
            cfreq = (np.cumsum(1. / df.recovery_probability.values / freq_) /
                    self.tot_obs_time)
            return cfreq, 1. / df.recovery_probability.values / freq_

        # Different keys call different correction procedures
        vals = {"no_corr": {False: ["ed_rec", cum_dist],
                            True: ["ed_rec", get_msf_cum_dist]},
                "ed_corr": {False: ["ed_corr", cum_dist],
                            True: ["ed_corr", get_msf_cum_dist]},
                "edrecprob_corr": {False: ["ed_corr", cum_dist_rec_prob],
                                   True: ["ed_corr", get_msf_cumdist_recprob]}
                }
        # make a copy to sort safely without affecting self.f
        df = self.f.copy(deep=True)
        # retrieve ED type (corrected or not), and function for counts
        col, func = vals[key][multiple_stars]
        df = df.sort_values(by=col, ascending=False)

        ed = df[col].values  # get the right EDs
        # get the (corrected) flare counts
        freq, counts = func(df, col, self.ID)  

        self.ed = ed
        self.freq = freq
        self.count_ed = _get_frequency_corrected_ed_sample(ed, counts)

        return self.ed, self.freq, self.count_ed

    def fit_beta_to_powerlaw(self, mode="ED"):
        '''Fit beta via non-linear least squares to a power
        law with given alpha using the cumulative
        FFD. Generate uncertainty using jackknife algorithm.

        Parameters:
        -----------
        mode : str
            ED or energy will set the starting value for the
            least square minimization

        Return:
        -------
        _beta, beta, beta_err -  array, float, float
            jackknife sample of beta values, mean beta, beta uncertainty
        '''
        def LSQ(x0, ed, freq, alpha):
            zw = ((x0 /
                   (np.power(ed, alpha - 1.) * (alpha - 1.)) - freq)**2).sum()
            return np.sqrt(zw)

        N = len(self.ed)
        if N == 0:
            raise ValueError('No data.')

        # jackknife uncertainty
        x0starts = {'ED': 10, 'energy': 1e25}
        _beta = np.array([fmin(LSQ, x0=x0starts[mode],
                              args=(np.delete(self.ed, i),
                                    np.delete(self.freq, i),
                                    self.alpha),
                              disp=0)[0] for i in range(N)])

        # cumulative beta = beta_cum
        beta = _beta.mean()
        beta_err = np.sqrt((N - 1) / N * ((_beta - beta)**2).sum())

        # propagate errors on alpha to beta
        beta_err = (np.sqrt(beta_err**2 * (self.alpha - 1.)**2 +
                            beta**2 * self.alpha_err**2))

        # set attributes
        self.beta = beta
        self.beta_err = beta_err

        return _beta, self.beta, self.beta_err

    def plot_powerlaw(self, ax, custom_xlim=None, **kwargs):
        '''
        Plot the power law fit to the FFD. [No tests]

        Parameters:
        -----------
        ax : matplotlibe Axes object
            plot to insert the power law in to
        custom_xlim : 2-tuple
            minimum, maximum ED/energy value for power law
        kwargs : dict
            Keyword arguments to pass to plt.plot()

        Return:
        --------
        3 power law points to construct a line
        in log-log representation.
        '''
        if custom_xlim is None:
            x = np.linspace(np.nanmin(self.ed), np.nanmax(self.ed), 3)
        else:
            mi, ma = custom_xlim
            x = np.linspace(mi, ma, 3)
        y = self.beta / np.abs(self.alpha - 1.) * np.power(x, -self.alpha + 1.)
        a = ax.plot(x, y, **kwargs)
        return a, x, y

    def fit_powerlaw(self, alims=[1.01, 3.]):
        '''
        Calculate the un-biased ML power law estimator
        from Maschberger and Kroupa (2009), sections
        3.1.4. and 3.1.5. by simply minimizing the equation in
        ML_powerlaw_estimator.

        Parameters:
        ------------
        alims:
            parameter range for power law exponent

        Return:
        -------
        alpha, alpha_err - float, float
            power law exponent and its jackknife uncertainty
        '''
        # use frequency adjusted ED sample?
        ed = self._get_ed()

        # solve eq. 9 using scipy.fmin, define jacknife uncertainty
        N = len(ed)
        _alpha = np.array([fmin(_ML_powerlaw_estimator, x0=2.,
                               args=(np.delete(ed, i),), disp=0)[0]
                          for i in range(N)])

        # alpha is the mean value
        alpha = _alpha.mean()

        # uncertainty is the standard deviation
        sig_alpha = np.sqrt((N - 1) / N * ((_alpha - alpha)**2).sum())

        self.alpha = alpha
        self.alpha_err = sig_alpha

        return self.alpha, self.alpha_err

    def is_powerlaw_truncated(self, rejection=(.15, .05), nthresh=100):
        '''
        Apply the exceedance test recommended by
        Maschberger and Kroupa 2009.

        Parameters:
        ------------
        rejection : tuple of floats < 1.
            above these thresholds the distribution
            can be suspected to be truncated
        nthresh : int
            Number at which to use the more permissive
            or more restrictive truncation rejection
            limit, i.e. value 0 or 1 in `rejection`

        Return:
        ---------
        True if power law not consistent with an un-truncated power law
        False if power law is consitent with an un-truncated power law
        '''
        ed = self._get_ed()

        mean, std = _calculate_average_number_of_exceeding_values(ed,
                                                                  self.alpha,
                                                                  500)

        if self.alpha > 2.:
            warnings.warn('Power law exponent is steep. '
                          'Power of statistical tests decreases '
                          'according to Maschberger and Kroupa 2009.')
            
        if len(ed) >= nthresh:
            truncation_limit = rejection[1]
            
        else:
            truncation_limit = rejection[0]

        truncated = (mean / len(ed) > truncation_limit)

        return truncated

    def is_powerlaw(self, sig_level=0.05):
        '''
        Test if we must reject the power law hypothesis
        judging by the stabilised Kolmogorov-Smirnov
        statistic, suggested by Maschberger and Kroupa
        2009.

        Parameters:
        -----------
        sig_level : float < 1.
            significance level for the hypothesis test

        Returns:
        ---------
        True if we cannot reject the power law hypothesis.
        False if we must reject the power law hypothesis.
        '''
        ed = self._get_ed()

        truncated = self.is_powerlaw_truncated()

        KS = _stabilised_KS_statistic(ed, alpha=self.alpha,
                                      truncated=truncated)

        limit = _calculate_KS_acceptance_limit(len(self.ed),
                                               sig_level=sig_level)

        ispowerlaw = KS < limit

        if ispowerlaw is False:

            warnings.warn('Kolmogorov-Smirnov tells us to reject'
                           r' the power law hypothesis at p={}.'
                           ' KS={}, limit={}'.format(sig_level, KS, limit))
        return ispowerlaw

    def _get_ed(self):
        """Get ED array either for a single star sample
        or a multiple stars sample, depending on `multiple_stars`
        flag.

        Return:
        -------
        ed - sample of flare energies 
        """

        if self.multiple_stars is True:
            
            ed = self.count_ed
                
        elif self.multiple_stars is False:
            
            ed = self.ed
        
        return ed
    
    
def _calculate_average_number_of_exceeding_values(data, alpha, n, **kwargs):
    '''
    Parameters:
    -----------
    ffd : FFD object

    n : int
        number of samples to average
    kwargs : dict
        Keyword arguments to pass to
        :func:calculate_number_of_exceeding_values

    Returns:
    --------
    (mean, std) : (float, float)
        average number number of exceeding values
        and standard deviation
    '''

    assert alpha is not None
    assert data is not None
    
    exceedance_statistic = [_calculate_number_of_exceeding_values(data, 
                                                                  alpha, 
                                                                  **kwargs) 
                            for i in range(n)]
    exceedance_statistic = np.array(exceedance_statistic)
    
    return np.nanmean(exceedance_statistic), np.nanstd(exceedance_statistic)


def _calculate_number_of_exceeding_values(data, alpha, maxlim=1e8, **kwargs):
    '''
    Helper function that mimicks data similar
    to the observations (same alpha and size)
    and returns a sample from an untruncated
    distribution. The number of values that
    exceeds the maximum in the actual data is
    returned.

    Parameters:
    -----------
    data : array
        observed values
    alpha : float
        best-fit power law exponent to the data
    maxlim : float > 1.
        factor to simulate an untruncated
        version of the given power law
        distribution
    kwargs : dict
        Keyword arguments to pass to
        :func:generate_random_power_law_distribution

    Return:
    --------
    int : number of exceeding values
    '''
    pdist = generate_random_power_law_distribution(np.min(data),
                                                   np.max(data) * maxlim,
                                                   -alpha + 1,
                                                   size=data.shape[0],
                                                   **kwargs)

    if np.isnan(pdist).any():
        raise ValueError('Fake power law distribution for the'
                         ' exceedance test could not be generated.'
                         ' Check your inputs.')
    return len(np.where(pdist > np.max(data))[0])


def _get_multistar_factors(dataframe, ID, sort):
    """Returns an array of factors to apply
    to the detected flares. Accounts for the
    number of targets with different detection
    thresholds that contribute to the sample
    
    factor = 1 / number of targets that contribute
    their flares above a given flare energy.
    
    This is a first order approximation that is
    assuming that the least energetic flare in a 
    light curve is just above the detection limit.
    
    If the smallest flare is significantly above
    the detection limit of a target's light curve
    small energy flare frequencies will be 
    overestimated. 
    
    The steeper the FFD the better the approximation.
    
    
    Parameters:
    -----------
    dataframe: DataFrame
        flare table with ID and sort columns
    ID: str
        column name for star ID in dataframe
    sort: str
        column name for energies or EDs dataframe
    
    Return: 
    -------
    multistar factor array
    """
    freq = []
    # make a copy to safely sort the dataframe
    df = dataframe.copy(deep=True)
    
    # check if sort exists, otherwise throw error
    try:
        df = df.sort_values(by=sort, ascending=True)
    except:
        raise KeyError(f"The flare table needs a {sort} column.")
    
    # loop over sorted dataframe and find the number of targets
    # that contribution to the sub-frame
    for i in range(df.shape[0]):
        
        try:
            f = df.iloc[:i + 1]  # sub-frame
            freq.append(len(set(f[ID].values)))  # append number of targets
            
        except KeyError:  # where are the unique target ID
            
            raise KeyError("Pass the column name of target IDs "
                           "to the FFD constructor: ID = ???.")
        
    # factor for maximum energy goes first and must be 1, the rest < 1
    return np.array(freq[::-1]) / freq[-1] 


def _ML_powerlaw_estimator(alpha, ed):
    '''
    Power law maximum likelihood estimator from
    Maschberger and Kroupa (2009),
    formula (9).

    Parameters:
    -----------
    alpha : float
        approximate value for power law exponent  
    ed : array
        ED or energy array
    
    Return:
    --------
    absolute value of left side of formula (9)
    To find MLE for alpha, minimize this term.
    '''
    if np.array(alpha <= 1.).any():
        # Power law exponent must be >1.
        return np.nan
        
    n = len(ed)
    if n == 0:
        raise ValueError('No data.')
    
    # Calculate Y variable in formula (9)
    Y = ed.min()
    if Y < 0:
        raise ValueError('Negative value encountered in data.')
    
    # De-bias alpha following Maschberger and Kroupa 2009    
    alpha = _de_bias_alpha(n, alpha)
    
    # Calculate the remaining variables in formula (9)
    Yexp = (np.power(Y, 1 - alpha))
    T = np.log(ed).sum()
    Z = _de_biased_upper_limit(ed, alpha)
    Zexp = (np.power(Z, 1 - alpha))

    return (np.abs(n / (alpha - 1) +
                   n * ((Zexp * np.log(Z) - Yexp * np.log(Y)) /
                        (Zexp - Yexp)) -
                   T))
    

def _de_biased_upper_limit(data, a):
    '''
    De-biases the upper limits for a
    ML power law exponent estimator.
    Uses formular (13) (and (14)) from
    Maschberger and Kroupa (2009).

    Parameters:
    -----------
    data : Series or array
        data that is suspected to follow
        a power law relation
    a : float or array of floats
        quasi de-biased ML estimator for alpha
        (de_bias_alpha before inserting here!)

    Returns:
    ---------
    Quasi de-biased upper limit.
    '''
    if len(data) == 0:
        raise ValueError('No data.')
    if (data < 0).any():
        raise ValueError('Negative values '
                         'encountered in data.')
    Xn = data.max()
    X1 = data.min()
    if Xn == X1:
        raise ValueError('Data range is zero.')
    n = len(data)
    G = (1. - a) * np.log(Xn / X1)  # (14)
    base = 1. + (np.exp(G) - 1.) / n
    exponent = 1. / (1. - a)
    return Xn * np.power(base, exponent)


def _de_bias_alpha(n, alpha):
    '''
    De-biases the power law value
    according to Maschberger and Kroupa (2009),
    formula (12).

    Paramaters:
    ------------
    n : int
        Size of the data
    alpha : float or array of floats
        Power law exponent value from ML estimator

    Returns:
    -----------
    quasi de-biased ML estimator for alpha
    '''
    if np.array(np.isnan(n) | np.isnan(np.array(alpha)).any()):
        raise ValueError('de_bias_alpha: one or '
                         'both arg(s) is/are NaN')
    return (alpha - 1.) * n / (n - 2) + 1.


def _stabilised_KS_statistic(data, alpha, truncated):
    '''
    Calculate the stabilised KS statistic
    from Maschberger and Kroupa 2009, Eqn. (21)
    orginally from Michael 1983, and Kimber 1985.

    Parameters:
    --------------
    data : array
        observed values that are suspected
        to follow a power law relation
    kwargs : dict
        Keyword arguments to pass to
        :func:calculate_cumulative_powerlaw_distribution
    Return:
    --------
    float - stablised KS statistic
    '''
    sorted_data = np.sort(data)
    pp = _calculate_cumulative_powerlaw_distribution(sorted_data,
                                                     alpha, truncated)
    y = (np.arange(1, len(pp) + 1) - .5) / len(pp)
    argument = (_apply_stabilising_transformation(y) -
                _apply_stabilising_transformation(pp))
    return np.max(np.abs(argument))


def _calculate_cumulative_powerlaw_distribution(data, alpha, truncated):
    '''
    Calculates the cumulative powerlaw distribution
    from the data, given the best fit power law exponent
    for y(x) ~ x^(-alpha).
    Eq. (2) in Maschberger and Kroupa 2009.

    Parameters:
    -----------
    data : array
        observed values that are suspected
        to follow a power law relation, sorted in
        ascending order
    alpha : float
        best-fit power law exponent
    truncated : bool
        True if the power law distribution is truncated

    Returns:
    ---------
    array : cumulative distribution
    '''
    if alpha <= 1.:
        raise ValueError('This distribution function is only'
                         ' valid for alpha > 1., see also '
                         'Maschberger and Kroupa 2009.')
    data = np.sort(data)
    
    def expa(x, alpha):
        return np.power(x, 1. - alpha)

    if truncated:
        CDF = ((expa(data, alpha) - expa(np.min(data), alpha)) /
               (expa(np.max(data), alpha) - expa(np.min(data), alpha)))
        
    if ~truncated:
        CDF = 1. - expa(data / np.min(data), alpha)
        
    # fix a -0. value that occurs as the first value
    CDF[np.where(CDF == 0.)[0]] = 0.
    
    return CDF


def _calculate_KS_acceptance_limit(n, sig_level=0.05):
    '''
    Above this limit we must reject the null-hypothesis.
    In our context, this is the hypothesis that the dis-
    tribution follows a given power law.

    Parameters:
    -----------
    n : int
        sample size
    sig_level : 0 < float < 1.
        significance level
    '''
    if ((sig_level >= 1.) | (sig_level <= 0.)):
        raise ValueError('Pass a valid significance level.')
    
    if n == 0:
        raise ValueError('No data to calculate KS_acceptance limit.')
    
    elif ((n <= 35) & (n > 0)):
    
        t = (pd.read_csv(f'{CWD}/static/KS_leq_35_values.csv',
                           delimiter='|', skiprows=1, header=None,
                           names=['n', .9, .95, .99])
             .set_index('n')
             .astype(float))
        return t.loc[n, 1 - sig_level]
    
    elif n > 35:
        
        return np.sqrt(-.5 * np.log((sig_level) / 2.)) / np.sqrt(n)

    
def _apply_stabilising_transformation(u):
    '''
    Applies the right-tail stabilising
    transformation from Kimber 1985 to
    a potentially power law distributed
    sample. Eq. 19 in Maschberger and Kroupa 2009.

    Used in :func:stabilised_KS_statistic

    Parameters:
    ------------
    u : array
        cumulative distribution

    Returns:
    -----------
    array : stabilised distribution
    '''
    u = np.array(u)
    if (u < 0).any():
        # validate input for sqrt
        raise ValueError("CDF values must be positive.")
    u = .5 + .5 * u
    S0 = 2. / np.pi * np.arcsin(np.sqrt(u))
    return 2. * S0 - 1.


def _get_frequency_corrected_ed_sample(ed, counts):
    """Use the corrected counts to create duplicates
    of flares in the sample so that it represents
    the corrected distribution.
    
    Parameters:
    -----------
    ed : array
        measured equivalent durations
    counts : array of floats
        their corrected number frequencies
        
    Return:
    -------
    array of equivalent durations that is adjusted
    for underlying count frequencies using duplicates
    """
    eds = []
    
    for e, n in zip(ed, counts):
        eds.append(int(np.rint(n)) * [e])
        
    return np.array([i for E in eds for i in E])


def generate_random_power_law_distribution(a, b, g, size=1, seed=None):
    """Power-law generator for pdf(x)\propto x^{g-1}
    for a<=x<=b
    """
    if seed is not None:
        np.random.seed(seed)
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag) * r)**(1. / g)
