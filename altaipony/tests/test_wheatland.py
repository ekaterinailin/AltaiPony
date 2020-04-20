import pytest
import numpy as np

from ..utils import generate_random_power_law_distribution
from ..wheatland import (occurrence_probability_posterior,
                         uninformative_prior,
                         flaring_rate_likelihood,
                         logit,
                         BayesianFlaringAnalysis,
                         calculate_joint_posterior_distribution,
                         generate_fake_data,
                         )


def test_logit():
    '''Test logit function by logifying
    some random function'''

    def func(x,a,b):
        return x / (b - a)
    log_uninf_prior = logit(func)
    up = func(3,2,6)
    logup = log_uninf_prior(3,2,6)
    assert np.log(0.75) == logup


def test_flaring_rate_likelihood():
    '''Test flaring rate likelihood with a
       5 flares per day rate.
    '''
    # Find the likelihood

    rates = np.linspace(1e-1, 20, 10000)
    posterior = flaring_rate_likelihood(rates, 75, 15., norm=True)

    #-----------------------------------------

    # Check some values:
    assert np.sum(posterior) == pytest.approx(1.)
    assert rates[np.argmax(posterior)] == pytest.approx(5., rel=1e-3)
    assert rates[np.argmin(np.abs(np.max(posterior)/2.-posterior))] ==  pytest.approx(4.351065, rel=1e-4)


def test_uninformative_prior():
    '''Test some exceptions that should be mostly ignored.'''

    # Define test values:
    vals = [(1,2,3), (2,1,3), (np.nan, 4, 10), (3, 2, 2)]

    # Run prior calculation on values:
    res = []
    for rate, minrate, maxrate in vals:
        res.append(uninformative_prior(rate, minrate, maxrate))

    # Check results:
    assert res == [0,0.5,0,0]


def test_occurrence_probability_posterior():
    '''Test the posterior using a uniform prior'''

    # Simulate the results from Wheatland 2004 in Fig. 1:
    # Use their precise values:

    t = 5 #total observation time in days
    cadence = 4 #observations per hour
    obstimes = np.linspace(3000,3000+t,t*24*4) # 15 min cadence observations
    flaresperday = 5. # average flaring rate in flares per day
    np.random.seed(3000)
    times = obstimes[np.where(np.random.poisson(lam=1. / 24. / cadence * flaresperday, size=t*24*4))[0]]
    size = len(times)
    events = generate_random_power_law_distribution(1, 100000, -.8, size=size, seed=778)
    #time interval to consider for prediction
    Tprime = 5. # if bayesian blocks used: bins[-1] - bins[-2]
    mined = 100 # min ED value we want to predict a rate for, same as S2 in Wheatland paper
    deltaT = 1. # predict rate of flares above threshold for deltaT days in the futures
    alpha = 1.8 # fix power law exponent for now
    threshed = 1. # detection sensitivity limit
    # number of observations
    Mprime = size# if bayesian blocks used: values[-1]

    # Find the posterior distribution:

    x = np.linspace(1e-8,1-1e-8,10000)
    predicted_distr = occurrence_probability_posterior(x, alpha, mined, Tprime,
                                                       Mprime, deltaT, threshed)

    #--------------------------------------------------

    # Check some values:
    # TODO: use more restrictive check because you know how to seed numpy random generators now.

    assert x[np.argmax(predicted_distr)] > 0.110
    assert x[np.argmax(predicted_distr)] < 0.131
    assert x[np.argmin(np.abs(np.max(predicted_distr)/2.-predicted_distr))] ==  pytest.approx( 0.12741274, rel=1e-4)

    # --------------------------------------------------

    # If debugging is needed later:
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
    # ax.plot(x, predicted_distr, color="r",
    #         label='ED={} s,  t={} d'.format(mined, deltaT))

    # ax.set_xlabel(r'$\varepsilon$: probability of one event'+'\nabove ED within t after last observation')
    # ax.set_ylabel(r'normalized $P_{\varepsilon}(\varepsilon)$')
    # ax.set_xlim(0,0.25)
    # ax.legend()


def test_sample_posterior_with_mcmc():
    '''Integration test using the example
    in Wheatland (2004) and our own
    likelihoods and priors.
    '''

    # Create some fake data:

    t = 5 #total observation time in days, must be int
    cadence = 4 #observations per hour
    obstimes = np.linspace(3000,3000+t,t*24*cadence) # 15 min cadence observations
    flaresperday = 10. # average flaring rate in flares per day
    times = obstimes[np.where(np.random.poisson(lam=1. / 24. / cadence * flaresperday, size=t * 24 * cadence))[0]]
    size = len(times)
    alpha_prior = 1.8 # fix power law exponent for now
    events = generate_random_power_law_distribution(1, 1000, -alpha_prior + 1., size=size, seed=788)
    #time interval to consider for prediction
    Tprime = 5#np.max(times) -  np.min(times)# if bayesian blocks used: bins[-1] - bins[-2]
    mined = 100 # min ED value we want to predict a rate for, same as S2 in Wheatland paper
    deltaT = 1.# predict rate of flares above threshold for deltaT days in the futures
    threshed = 1 # detection sensitivity limit
    Mprime = size# if bayesian blocks used: values[-1]
    rate_prior = flaresperday / np.abs(alpha_prior - 1.) * np.power(mined, -alpha_prior +1.) # evaluate cumulative FFD fit at mined
    eps_prior = 1 - np.exp(-rate_prior * deltaT) # calculate poisson probability.  Eqn (5) in Wheatland (2004)

    #---------------------------------

    # Define your log-likelihood function:

    def loglikelihood(theta, *args):
        def prior(x):
            return uninformative_prior(x, 1.25, 2.25)
        return calculate_joint_posterior_distribution(theta, *args, prior)

    # Create a BayesianAnalysisObject

    BFA = BayesianFlaringAnalysis(mined=mined, Tprime=Tprime, deltaT=deltaT, alpha_prior=alpha_prior, eps_prior=eps_prior,
                              threshed=threshed, Mprime=Mprime, events=events, loglikelihood=loglikelihood)

    # Run MCMC to sample the posterior distribution

    BFA.sample_posterior_with_mcmc()

    #---------------------------------

    # Check that the function ran through

    assert BFA.samples.shape == (31800, 2) #this goes wrong if default values for steps/cutoff/nwalkers are changed
    assert BFA.samples[:,1].max() < 2.25 # as defined by prior on alpha
    assert BFA.samples[:,1].min() > 1.25 # as defined by prior on alpha
    assert BFA.samples[:,0].min() > 0 # beta is always positive

    # Check that the results are correct maybe?


def test_calculate_percentiles():
    '''Test if the percentiles come out
    correctly in a random gauss shaped sample.
    '''
    # Create some data with a seed:

    np.random.seed(seed=590)
    samples = np.array([np.random.normal(loc=1.0, scale=.2, size=1000),
                        np.random.normal(loc=2.0, scale=.4, size=1000)]).T
    BFA = BayesianFlaringAnalysis(samples=samples)

    #-----------------------------------

    # Run the function:

    pct = BFA.calculate_percentiles()

    #-----------------------------------

    # Check the results:

    assert pct[0] == (1.008030476417989, 0.19516752006028892, 0.21571271977738948)
    assert pct[1] == (1.980568570082061, 0.4199525752578359, 0.37256627621632066)


def test_generate_fake_data():
    
    # ---------------------------------------------------------
    # Integration test
    fake = generate_fake_data(13, 120, 5, 2., seed=9000)
    assert fake["deltaT"] == 13. 
    assert fake["deltaT"] == fake["Tprime"]
    assert fake["alpha_prior"] ==  2.
    assert fake["beta_prior"] ==  5.
    assert fake["threshed"] == 1. 
    assert fake["mined"] ==  pytest.approx(78.78287887 * 10., rel=.01)
    assert fake["beta_prior"] == 5. 
    assert fake["eps_prior"] == pytest.approx(0.07919338333736559)
    print(len(fake["events"]))
    assert len(fake["events"]) == pytest.approx(65,abs=10) # 65 would be the perfect as in 5x13
    assert (fake["events"]>1.).all()

    # -----------------------------------------------------------
    # In case of too high flaring rate relative to time sampling:
    with pytest.raises(ValueError):
        fake = generate_fake_data(13, 120, 1000, 2., seed=9000)
