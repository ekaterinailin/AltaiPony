import pytest
import copy

import numpy as np
import pandas as pd

import logging

from ..utils import generate_random_power_law_distribution
from ..wheatland import generate_fake_data
from ..ffd import (FFD,
                   _get_multistar_factors,
                   _ML_powerlaw_estimator,
                   _de_biased_upper_limit,
                   _de_bias_alpha,
                   _calculate_percentile_max_ed,
                   _calculate_max_ed,
                   _stabilised_KS_statistic,
                   _calculate_KS_acceptance_limit,
                   _apply_stabilising_transformation,
                    )

#------------------------------ TESTING FFD() ------------------------------------------

def test_init_FFD():
    
    # Generate a flare table
    a, b, g, size = 10, 1e3, -1, 200
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)
    df = pd.DataFrame({"ed_rec": pwl,
                       "ed_corr": pwl * 1.2,
                       "recovery_probability":.8,
                       "TIC":list(np.arange(1,21))*10
                       })
    
    # init an FFD object
    ffd = FFD(df)
    
    # check correct initialisation
    assert ffd.f.shape[0] == 200
    assert (ffd.f.columns.values == ['ed_rec', 'ed_corr',
                                     'recovery_probability', 
                                     'TIC']).all()

#---------------------------------------------------------------------------------------

#--------------------- TESTING SETTER OUTPUTS IN PROPERTIES ----------------------------

def test_setter_outputs(caplog):
    """Test all setters to work in both silent and loud modes.
    Using the caplog fixture from pytest to get logging outputs.""" 
    with caplog.at_level(logging.INFO):
        simple_ffd = FFD()
    print(caplog)

    assert ('No total observing time given. Set to 1. '
            'You are now working with number counts instead of frequency.' in caplog.text)

    # multiple_stars
    with caplog.at_level(logging.INFO):
        simple_ffd._multiple_stars = True
    assert ('Setting multiple_stars flag with True.' in caplog.text) == False
    with caplog.at_level(logging.INFO):    
        simple_ffd.multiple_stars = True
    assert 'Setting multiple_stars flag with True.' in caplog.text
    
    a = [1,2,3]

    # ed
    s = f"Setting ED with new values, size {len(a)}."
    with caplog.at_level(logging.INFO):
        simple_ffd._ed = a
    assert (s in caplog.text) == False
    with caplog.at_level(logging.INFO):    
        simple_ffd.ed = a
    assert s in caplog.text

    # freq
    s = f"Setting frequency values with new values, size {len(a)}."
    with caplog.at_level(logging.INFO):    
        simple_ffd._freq = a
    assert (s in caplog.text) == False
    with caplog.at_level(logging.INFO):
        simple_ffd.freq = a
    assert s in caplog.text

    # count_ed
    with caplog.at_level(logging.INFO):
        simple_ffd._count_ed = a
    s = (f"Setting frequency adjusted count values "
         f"with new values, size {len(a)}.")
    assert (s in caplog.text) == False
    with caplog.at_level(logging.INFO):    
        simple_ffd.count_ed = a
    assert s in caplog.text
    
#---------------------------------------------------------------------------------------

#--------------------- TESTING ed_and_freq() ---------------------

def test_ed_and_freq():
    """Tests _ed_and_counts under the hood."""
    
    # Generate a flare table
    a, b, g, size = 10, 1e3, -1, 200
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)
    df = pd.DataFrame({"ed_rec":pwl,
                   "ed_corr":pwl*1.2,
                   "recovery_probability":.8,
                   "TIC":list(np.arange(1,21))*10
                    })
    
    # Init an FFD object
    ffd = FFD(df)

    # Check if the results are correct for all combinations of parameters:
    
    
    # SINGLE STAR SAMPLE
    # -----------------------------------------------------------------
    
    #------------------------------------------------------------------
    # no correction
    ed, freq, counts = ffd.ed_and_freq(energy_correction=False,
                               recovery_probability_correction=False,
                                multiple_stars=False)
    assert (freq == np.arange(1,201)).all()
    assert (ed == df.ed_rec.sort_values(ascending=False)).all()
    # Default setting ...
    assert ffd.tot_obs_time == 1.
    # ... implies that smallest frequency = 1
    assert freq[0] == 1
    #------------------------------------------------------------------
    
    #------------------------------------------------------------------
    # only energy correction
    ed, freq, counts = ffd.ed_and_freq(energy_correction=True,
                               recovery_probability_correction=False,
                                multiple_stars=False)
    assert (freq == np.arange(1,201)).all()
    assert (ed == df.ed_corr.sort_values(ascending=False)).all()
    # Default setting ...
    assert ffd.tot_obs_time == 1.
    # ... implies that smallest frequency = 1
    assert freq[0] == 1
    #------------------------------------------------------------------
    
    #------------------------------------------------------------------
    # energy and frequency correction
    ed, freq, counts = ffd.ed_and_freq(energy_correction=True,
                               recovery_probability_correction=True,
                                multiple_stars=False)

    assert (ed == df.ed_corr.sort_values(ascending=False)).all()
    assert (freq == np.arange(1,201)/.8).all()
    # Default setting ...
    assert ffd.tot_obs_time == 1.
    # ... implies that smallest frequency = 1 / recovery probablity
    assert freq[0] == 1 / .8
    #------------------------------------------------------------------

    # MUTLIPLE STARS IN SAMPLE
    # -------------------------------------------------------------------

    ffd = FFD(df)

    # You must pass a Key to ID
    with pytest.raises(KeyError):
        ed, freq, counts = ffd.ed_and_freq(energy_correction=False,
                               recovery_probability_correction=False,
                                multiple_stars=True)

    ffd = FFD(df,ID="TIC")
    
    #------------------------------------------------------------------
    # no correction
    ed, freq, counts = ffd.ed_and_freq(energy_correction=False,
                               recovery_probability_correction=False,
                                multiple_stars=True)
    #frequencies always increase in FFD
    assert (np.diff(freq) > 0.).all()
    #ED array is the same as the ed_rec column
    assert (ed == df.ed_rec.sort_values(ascending=False)).all()
    # When multiple stars are involved with different minimum
    # detected energies, the frequency adjusted ED array must be larger
    # than the original one
    assert len(ffd.count_ed) > len(ffd.freq)
    # The ultiple_stars flag must be set!
    assert ffd.multiple_stars
    # Default setting ...
    assert ffd.tot_obs_time == 1.
    # ... implies that smallest frequency = 1
    assert freq[0] == 1
    
    #------------------------------------------------------------------
    # only energy correction
    ed, freq, counts = ffd.ed_and_freq(energy_correction=True,
                               recovery_probability_correction=False,
                                multiple_stars=True)

    #frequencies always increase in FFD
    assert (np.diff(freq) > 0.).all()
    #ED array is the same as the ed_corr column
    assert (ed == df.ed_corr.sort_values(ascending=False)).all()
    # When multiple stars are involved with different minimum
    # detected energies, the frequency adjusted ED array must be larger
    # than the original one
    assert len(ffd.count_ed) > len(ffd.freq)
    # The ultiple_stars flag must be set!
    assert ffd.multiple_stars
    # Default setting ...
    assert ffd.tot_obs_time == 1.
    # ... implies that smallest frequency = 1
    assert freq[0] == 1
    _f = copy.copy(freq)
    #------------------------------------------------------------------

    #------------------------------------------------------------------
    # energy and frequency correction
    ed, freq, counts = ffd.ed_and_freq(energy_correction=True,
                               recovery_probability_correction=True,
                                multiple_stars=True)

    #frequencies always increase in FFD
    assert (np.diff(freq) > 0.).all()
    #ED array is the same as the ed_rec column
    assert (ed == df.ed_corr.sort_values(ascending=False)).all()
    # When multiple stars are involved with different minimum
    # detected energies, the frequency adjusted ED array must be larger
    # than the original one
    assert len(ffd.count_ed) > len(ffd.freq)
    # The ultiple_stars flag must be set!
    assert ffd.multiple_stars
    # Default setting ...
    assert ffd.tot_obs_time == 1.
    # ... implies that smallest frequency = 1 / recovery_probability
    assert freq[0] == 1 / .8
    # Adjusting for recovery probability gives a values divided
    # by it
    assert _f/.8 == pytest.approx(freq)
    #------------------------------------------------------------------
    
    # Check failing case:
    # -------------------------------------------------------------
    
    # You must set energy_correction if you want to use 
    # recovery_probability correction
    with pytest.raises(KeyError):
        ed, freq, counts = ffd.ed_and_freq(energy_correction=False,
                               recovery_probability_correction=True,
                                multiple_stars=True)
        
    with pytest.raises(KeyError):
        ed, freq, counts = ffd.ed_and_freq(energy_correction=False,
                               recovery_probability_correction=True,
                                multiple_stars=False)

#---------------------------------------------------------------------------------------

#--------------------------- TESTING fit_powerlaw() -------------------------------
        
def test_fit_mmle_powerlaw():
    # Generate a flare table
    a, b, g, size = 10, 1e3, -1, 200
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)
    df = pd.DataFrame({"ed_rec":pwl,
                       "ed_corr":pwl*1.2,
                       "recovery_probability":.8,
                       "TIC":list(np.arange(1,21))*(size//20)
                       })

    # init an FFD object
    ffd = FFD(f=df)
    ed, freq, counts = ffd.ed_and_freq()
    
    # mode 'mmle'
    ffd.fit_powerlaw("mmle")
    assert (1.963553855895996, 0.08012203082491737) == (ffd.alpha, ffd.alpha_err)
    assert (1753.1677389526367, 140.50464845948764) == (ffd.beta, ffd.beta_err)

    # mode 'mcmc'
    ffd.fit_powerlaw("mcmc")
    assert ((2.041785081531872, 0.07598158206969874, 0.07111524587912488) ==
            (ffd.alpha, ffd.alpha_up_err, ffd.alpha_low_err))
    assert ((2302.96755340424, 670.4364587925747, 503.09507205326236) ==
            (ffd.beta, ffd.beta_up_err, ffd.beta_low_err))

#---------------------------------------------------------------------------------------

#--------------------------- TESTING fit_mmle_powerlaw() -------------------------------
        
def test_fit_mmle_powerlaw():
    # Generate a flare table
    a, b, g, size = 10, 1e3, -1, 200
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)
    df = pd.DataFrame({"ed_rec":pwl,
                       "ed_corr":pwl*1.2,
                       "recovery_probability":.8,
                       "TIC":list(np.arange(1,21))*(size//20)
                       })

    # init an FFD object
    ffd = FFD(f=df)
    ed, freq, counts = ffd.ed_and_freq()
    
    # check the result
    assert (1.963553855895996, 0.08012203082491737) == ffd.fit_mmle_powerlaw()

#---------------------------------------------------------------------------------------

#------------------------------ TESTING is_powerlaw_truncated() ------------------------

cases = [(1000, False), (900, False), (800, False),
         (200, True), (20, True)]

@pytest.mark.parametrize("l,i", cases)
def test_is_powerlaw_truncated(l,i):
    a, b, g, size = 10, 1e3, -1, 200
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)
    f = pd.DataFrame({"ed_rec":pwl})
    simple_truncated_ffd = FFD(f=f[f.ed_rec < l])#truncate at l seconds 
    simple_truncated_ffd.alpha = 2.
    ed, freq, counts = simple_truncated_ffd.ed_and_freq()
    assert simple_truncated_ffd.is_powerlaw_truncated() == i
    
#---------------------------------------------------------------------------------------

#------------------------------ TESTING is_powerlaw() ----------------------------------

def test_is_powerlaw():

    a, b, g, size = 10, 1e3, -1., 200
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)

    # init an FFD object
    ffd = FFD(f=pd.DataFrame({"ed_rec":pwl}))
    ffd.ed_and_freq()
    
    ffd.alpha = 2.
    # pwl is a power law with exponent 2
    assert ffd.is_powerlaw()

    ffd.alpha = 2.3
    # pwl is not a power law with exponent 2
    assert not ffd.is_powerlaw()


    a, b, g, size = 10, 1e3, -1., 20
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)

    # init an FFD object
    ffd = FFD(f=pd.DataFrame({"ed_rec":pwl}))
    ffd.ed_and_freq()

    ffd.alpha = 2.
    # pwl is a power law with exponent 2
    assert ffd.is_powerlaw()

    ffd.alpha = 2.3
    # pwl is not a power law with exponent 2 but 20 is too small of a sample
    assert ffd.is_powerlaw()

    a, b, g, size = 10, 1e3, -1., 50
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)
    ffd = FFD(f=pd.DataFrame({"ed_rec":pwl}))

    with pytest.raises(TypeError): #throw error when alpha is missing
        ffd.is_powerlaw()

#---------------------------------------------------------------------------------------

#------------------------------ TESTING _get_ed() ----------------------------------

def test__get_ed():

    a, b, g, size = 10, 1e3, -1., 200
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)

    # init an FFD object

    # case multiple_stars==False
    ffd = FFD(f=pd.DataFrame({"ed_rec":pwl}))
    ffd.ed_and_freq()
    
    assert (ffd._get_ed() == np.sort(pwl)[::-1]).all()

    # case multiple_stars==True
    pwl = np.sort(pwl)[::-1]
    ffd = FFD(f=pd.DataFrame({"ed_rec":pwl, "ID": [1]*(len(pwl)-1) + [2]}),
              ID="ID")
    ffd.ed_and_freq(multiple_stars=True)
    assert (ffd._get_ed() == list(pwl) + [np.min(pwl)]).all()


# -------------------- TESTING fit_mcmc_powerlaw() -------------------------------------

def test_fit_mcmc_powerlaw():

    # ----------------------------------------------------------
    # Run an integration test:

    # Generate fake FFD dictionary
    fake = generate_fake_data(13, 120, 5, 1.7, seed=9000)

    # Create an instance of an FFD
    ffd = FFD()
    ffd.f = pd.DataFrame({"ed_rec":fake["events"]})
    ffd.tot_obs_time = fake['Tprime'] 
    ffd.beta_prior = fake["beta_prior"]
    ffd.alpha_prior = fake["alpha_prior"]
    ed, freq, counts = ffd.ed_and_freq()

    # Call the function you wish to test
    np.random.seed(42)
    BFA = ffd.fit_mcmc_powerlaw()

    # Keep this for de-bugging
    fig = BFA.show_corner_plot()

    # check if the results are approximately right
    assert ffd.alpha == pytest.approx(1.7, rel=.1) 
    assert ffd.beta == pytest.approx(5., rel=.2)
    assert ffd.beta_err is None
    assert ffd.alpha_err == pytest.approx(0.075, rel=.1)
    assert ffd.alpha_up_err == pytest.approx(0.080, rel=.1)
    assert ffd.alpha_low_err == pytest.approx(0.073, rel=.1)
    assert ffd.beta_up_err == pytest.approx(.827, rel=.1)
    assert ffd.beta_low_err == pytest.approx(.720, rel=.1)
    assert ffd.beta_prior == 5.
    assert ffd.alpha_prior == 1.7
    assert len(ffd.ed) == len(fake["events"])
    
    # ----------------------------------------------------------
    # Run a failing test

    # This should because ffd.f is not given:
    ffd = FFD()
    with pytest.raises(ValueError):
        BFA = ffd.fit_mcmc_powerlaw()

#---------------------------------------------------------------------------------------

# -------------------- TESTING _get_multistar_factors() ---------------------------------

def test__get_multistar_factors():
    
    # Generate a flare table
    N = 20
    testdf = pd.DataFrame({"ID":np.arange(N)%4,
                           "sortcol":np.arange(200, 200-N, -1)})
    
    # call _get_multistar_factors
    f = _get_multistar_factors(testdf, "ID", "sortcol")
    
    # Check if the result is as expected
    assert (f == np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                           1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25])).all()

    # If the required keys don't exist throw errors
    with pytest.raises(KeyError):
        testdf = pd.DataFrame({"ID":np.arange(N)%4})
        f = _get_multistar_factors(testdf, "ID", "sortcol")

    with pytest.raises(KeyError):
        testdf = pd.DataFrame({"sortcol":np.arange(200, 200-N, -1)})
        f = _get_multistar_factors(testdf, "ID", "sortcol")

# --------------------------------------------------------------------------------------

#-------------------------- TESTING _ML_powerlaw_estimator() ---------------------------
    
def test__ML_powerlaw_estimator():
    dataformat = [np.array, pd.Series]
    for df in dataformat:
        ed = df([1,1,1,1,2,2,4])
        x0 = .9
        assert np.isnan(_ML_powerlaw_estimator(x0, ed))
        x0 = 1.5
        assert _ML_powerlaw_estimator(x0, ed) == pytest.approx(1.6190804181576444)
        ed = df([])
        with pytest.raises(ValueError):
            _ML_powerlaw_estimator(x0, ed)
        ed = df([-3,2,2,2])
        with pytest.raises(ValueError):
            _ML_powerlaw_estimator(x0, ed)
        ed = df([1,1,1,1])
        with pytest.raises(ValueError):
            _ML_powerlaw_estimator(x0, ed)

#---------------------------------------------------------------------------------------

#------------------------ TESTING _de_biased_upper_limit() -----------------------------

            
def test__de_biased_upper_limit():
    dataformat = [np.array, pd.Series]
    for df in dataformat:
        data = df([])
        with pytest.raises(ValueError):
            _de_biased_upper_limit(data, 2.)
        data = df([1,10, 100])
        assert _de_biased_upper_limit(data, 1000000.) == pytest.approx(100., rel=1e-4)
        data = df([1,1,1])
        with pytest.raises(ValueError):
            _de_biased_upper_limit(data, 3.)
        data = df([-1,1,1])
        with pytest.raises(ValueError):
            _de_biased_upper_limit(data, 3.)

#---------------------------------------------------------------------------------------

#----------------------------- TESTING _de_bias_alpha() --------------------------------
            
def test__de_bias_alpha():
    assert _de_bias_alpha(200,1) == 1.
    with pytest.raises(ValueError):
        _de_bias_alpha(np.nan,2)
    with pytest.raises(ValueError):
        _de_bias_alpha(30,np.nan)
    with pytest.raises(ValueError):
        _de_bias_alpha(np.nan,np.nan)
    with pytest.raises(ZeroDivisionError):
        _de_bias_alpha(2,2)
        
#---------------------------------------------------------------------------------------

#-------------------------- TESTING _calculate_max_ed() --------------------------------

def test__calculate_max_ed():
    # fake data
    data = np.linspace(10,1e3,100)
    alpha = 2.

    # apply function once
    maxval = _calculate_max_ed(data, alpha, maxlim=1e8, seed=10)

    assert maxval == pytest.approx(808.1118136567375)

    # apply it again

    maxval = _calculate_max_ed(data, alpha, maxlim=1e8, seed=2000)
    assert maxval == pytest.approx(1372.7172156831662)

    # test failure mode alpha or data is not finite
    with pytest.raises(TypeError):
        maxval = _calculate_max_ed(data, None, maxlim=1e8, seed=2000)

    with pytest.raises(TypeError):
        maxval = _calculate_max_ed(None, alpha, maxlim=1e8, seed=2000)

    with pytest.raises(AssertionError):
        maxval = _calculate_max_ed(data, np.nan, maxlim=1e8, seed=2000)

    with pytest.raises(AssertionError):
        maxval = _calculate_max_ed([np.nan, 1., 2., 3. ,4.],
                                   alpha, maxlim=1e8, seed=2000)

#---------------------------------------------------------------------------------------

#---------------------- TESTING _calculate_percentile_max_ed() -------------------------

def test__calculate_percentile_max_ed():
    # fake data
    data = np.linspace(10,1e3,100)
    alpha = 2.
    n = 10000
    percentile = 3.

    # apply function
    vals, edcrit = _calculate_percentile_max_ed(data, alpha, n, percentile)
    assert len(vals) == n
    assert edcrit == pytest.approx(289, abs=20)
    
    # test failure modes
    with pytest.raises(AssertionError):
        vals, edcrit = _calculate_percentile_max_ed(data, alpha, n, 200)
    with pytest.raises(AssertionError):
        vals, edcrit = _calculate_percentile_max_ed(data, alpha, n, -3)
    with pytest.raises(AssertionError):
        vals, edcrit = _calculate_percentile_max_ed(data, alpha, n, np.nan)
        
#---------------------------------------------------------------------------------------

#------------------------ TESTING _stabilised_KS_statistic() -----------------------------

def test__stabilised_KS_statistic():
    sizes = [1e2,1e3,1e4]
    minval, maxval = 10, 1e4
    datas = [generate_random_power_law_distribution(minval, maxval, -1., size=int(size),
                                                    seed=10) for size in sizes]
    
    KSlist = [_stabilised_KS_statistic(data, 2., False) for data in datas]
    assert KSlist[0] > KSlist[1]
    assert KSlist[1] > KSlist[2]
    

#---------------------------------------------------------------------------------------

#---------------------- TESTING _calculate_KS_acceptance_limit() -----------------------

def test__calculate_KS_acceptance_limit():
    with pytest.raises(ValueError):
        _calculate_KS_acceptance_limit(0, sig_level=0.05)
    with pytest.raises(ValueError):
        _calculate_KS_acceptance_limit(0, sig_level=-0.05)
    with pytest.raises(ValueError):
        _calculate_KS_acceptance_limit(0, sig_level=1.05)
    assert (_calculate_KS_acceptance_limit(100, sig_level=0.05)
            == pytest.approx(0.13581, rel=1e-4))
    assert (_calculate_KS_acceptance_limit(100, sig_level=0.01)
            == pytest.approx(0.16276, rel=1e-4))
    assert (_calculate_KS_acceptance_limit(100, sig_level=0.01)
            > _calculate_KS_acceptance_limit(1000, sig_level=0.01))
    
#---------------------------------------------------------------------------------------

#--------------------- TESTING _apply_stabilising_transformation() ---------------------

def test__apply_stabilising_transformation():
    u = [.1,.2,.3]
    assert _apply_stabilising_transformation(u).shape[0] == 3
    assert (np.diff(u) > 0).all()

    # giving an empty arrays returns an empty array
    assert (_apply_stabilising_transformation([]) == np.array([])).all()

    # Passing negative values throws an error
    with pytest.raises(ValueError):
        _apply_stabilising_transformation([-1.,.2,.4])
