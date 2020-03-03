import pytest
import copy

import numpy as np
import pandas as pd

from ..ffd import (FFD,
                   _get_multistar_factors,
                   _ML_powerlaw_estimator,
                   _de_biased_upper_limit,
                   _de_bias_alpha,
                   _calculate_average_number_of_exceeding_values,
                   _calculate_number_of_exceeding_values,
                   _stabilised_KS_statistic,
                   _calculate_KS_acceptance_limit,
                   _apply_stabilising_transformation,
                   generate_random_power_law_distribution
                    )

def test_init_FFD():
    
    # Generate a flare table
    a, b, g, size = 10, 1e3, -1, 200
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)
    df = pd.DataFrame({"ed_rec":pwl,
                   "ed_corr":pwl*1.2,
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

def test_ed_and_freq():
    """Tests _ed_and_counts, too."""
    
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
        
def test_fit_powerlaw():
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
    assert (1.963553855895996, 0.08012203082491737) == ffd.fit_powerlaw()

#------------------------------ TESTING is_powerlaw_truncated() ---------------

cases = [(1000, False), (900, False), (800, False),
         (200, True), (20, True)]

@pytest.mark.parametrize("l,i", cases)
def test_is_powerlaw_truncated(l,i):
    a, b, g, size = 10, 1e3, -1, 200
    pwl = generate_random_power_law_distribution(a, b, g, size=size, seed=80)
    f = pd.DataFrame({"ed_rec":pwl})
    simple_truncated_ffd = FFD(f=f[f.ed_rec < l])#truncate at 200s 
    simple_truncated_ffd.alpha = 2.
    ed, freq, counts = simple_truncated_ffd.ed_and_freq()
    assert simple_truncated_ffd.is_powerlaw_truncated() == i
    
#--------------------------------------------------------------------------------

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

    with pytest.raises(AssertionError): #throw error when alpha is missing
        ffd.is_powerlaw()
    
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
        

def test__calculate_average_number_of_exceeding_values():
    data = np.linspace(10,1e4, 300)
    alpha = 2.
    mean, std = _calculate_average_number_of_exceeding_values(data, alpha, 1000, seed=2311)
    assert mean == 3.
    assert std == 0.
    mean, std = _calculate_average_number_of_exceeding_values(data, alpha, 1000, seed=10)
    assert mean == 0.
    assert std == 0.


def test__calculate_number_of_exceeding_values():
    data = np.linspace(10,1e4, 300)
    assert _calculate_number_of_exceeding_values(data, 2., seed=10) == 0
    assert _calculate_number_of_exceeding_values(data, 2., seed=2311) == 3
    with pytest.raises(ValueError):
        _calculate_number_of_exceeding_values(np.arange(3), 2., seed=2311)
        

def test__stabilised_KS_statistic():
    sizes = [1e2,1e3,1e4]
    minval, maxval = 10, 1e4
    datas = [generate_random_power_law_distribution(minval, maxval, -1., size=int(size), seed=10) for size in sizes]
    
    KSlist = [_stabilised_KS_statistic(data, 2., False) for data in datas]
    assert KSlist[0] > KSlist[1]
    assert KSlist[1] > KSlist[2]
    

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
    

def test__apply_stabilising_transformation():
    u = [.1,.2,.3]
    assert _apply_stabilising_transformation(u).shape[0] == 3
    assert (np.diff(u) > 0).all()

    # giving an empty arrays returns an empty array
    assert (_apply_stabilising_transformation([]) == np.array([])).all()

    # Passing negative values throws an error
    with pytest.raises(ValueError):
        _apply_stabilising_transformation([-1.,.2,.4])