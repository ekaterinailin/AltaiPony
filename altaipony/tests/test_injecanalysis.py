import os
import pytest

import numpy as np
import pandas as pd

import matplotlib

from ..injrecanalysis import (characterize_flares,
                              tile_up_injection_recovery,
                              multiindex_into_df_with_nans,
                              percentile,
                              wrap_characterization_of_flares,
                              plot_heatmap,
                              _heatmap,
                              setup_bins,
                             )
from ..flarelc import FlareLightCurve
                             
from . import PACKAGEDIR                             

def test_wrap_characterization_of_flares():
    # Create a fake injection-recovery data set

    np.random.seed(20) #seed the random number generator
    n=int(1e5) # size

    amplrec = np.random.rand(n)*5
    amplitude = amplrec*1.2

    durrec = np.random.rand(n)*.2
    durrecsample = (durrec // (2. / 60. / 24.)) * (2. / 60. / 24.)
    tstart = np.random.rand(n) * 12000
    tstop = tstart + durrecsample
    dur = durrec*.5

    edinj = np.random.rand(n)*5000
    edrec = edinj*.8
    edrec[np.where(np.random.rand(n) < 0.1)] = 0.

    injrec = pd.DataFrame({"ampl_rec":amplrec,
                            "amplitude":amplitude,
                            "duration_d":dur,
                            "ed_inj":edinj,
                            "ed_rec":edrec,
                            "ed_rec_err":edrec*.1,
                            "dur":durrecsample,
                            "tstart":tstart,
                            "tstop":tstop,
                            })

    # read in test flares with realistic values from file:
    testflares = pd.read_csv(os.path.join(PACKAGEDIR,
                                          "tests/testfiles/testflares.csv"))

    #---------------------------------------------------------
    # Call the function
    fl = wrap_characterization_of_flares(injrec, testflares, ampl_bins=70, dur_bins=160)


    #---------------------------------------------------------
    # Check some numbers

    assert fl.shape[0] == 5
    assert (fl.ed_rec.values / 0.8 == fl.ed_corr.values).all()
    assert (fl.ed_ratio_count.values < 15).all()  # the numbers in the bins are not too high
    assert (fl.ed_ratio_count.values > 5).all()  # the numbers in the bins are not too low
    assert (fl.recovery_probability_count.values < 30).all()  # the numbers in the bins are not too high
    assert (fl.recovery_probability_count.values > 15).all()  # the numbers in the bins are not too low
    assert len(np.where(fl.isnull())[0]) == 0 # no NaNs
    assert (fl.ed_ratio.values == 0.8).all() # should be by design
    assert (fl.amplitude_ratio * 1.2 ==  1.).all() # should be by design
    # recovery probability is .9 for injected but there are shifts induced to this values by ED ratio correction:
    assert (fl.recovery_probability.values > .75).all()
    assert (fl.ed_corr_err.values ==
            pytest.approx(np.array([6.04627777e+02, 1.06333934e+02,
                                    5.90000000e-01, 4.30000000e-01,
                                    5.80000000e-01])))
    assert fl.amplitude_corr_err.values == pytest.approx(0) # shoulb be by data set design
    assert (fl.duration_corr_err.values < (2. / 60. / 24.)).all() #uncertainties smaller than time resolution by design


def test_characterize_flares():
    #-------------------------------
    # Create a fake data set
    # First injection recovery data:

    np.random.seed(20)
    n=int(1e5)
    amplrec = np.random.rand(n)*10
    amplitude = amplrec*1.2
    durrec = np.random.rand(n)*.6
    dur = durrec*.5
    edinj = np.random.rand(n)*5000
    edrec = edinj*.8
    dftest0 = pd.DataFrame({"ampl_rec":amplrec,
                            "amplitude":amplitude,
                            "duration_d":dur,
                            "ed_inj":edinj,
                            "ed_rec":edrec,
                            "dur":durrec,
                            "rec":(np.random.rand(n) > 0.1).astype(float)})

    # Then, flare detections:

    n=int(10)
    amplrec = np.random.rand(n)*10
    edrec = np.random.rand(n)*4000
    tstop = np.random.rand(n)*2000.
    tstart = tstop - .3
    testflares = pd.DataFrame({"ampl_rec":amplrec,
                               "ed_rec":edrec,
                               "tstop": tstop,
                               "tstart": tstart})
    #-------------------------------
    # Call the function:

    fl, ds = characterize_flares(testflares, dftest0,
                             ampl_bins=np.linspace(0, 12, 11),
                             dur_bins=np.linspace(0, .6, 11))

    #-------------------------------
    # Do some tests:

    assert fl.recovery_probability.values == pytest.approx(.9, rel=.05)
    assert (fl.duration_ratio.values == 2.).all()
    assert (fl.amplitude_ratio.values == 10. / 12.).all()
    assert (fl.ed_ratio.values == .8).all()
    assert (fl.amplitude_corr * fl.amplitude_ratio == fl.ampl_rec).values.all()
    assert (fl.duration_corr * fl.duration_ratio == fl.dur).values.all()
    assert (fl.ed_corr * fl.ed_ratio).values == pytest.approx(fl.ed_rec, rel=.05)
    assert fl.shape[1] == 16
    assert fl.shape[0] == 10
    assert len(ds.keys()) == 4
    assert fl.recovery_probability_count.values == pytest.approx(2000, rel=.05)

def test_tile_up_injection_recovery():
    np.random.seed(20)
    n=int(1e5)
    amplrec = np.random.rand(n)*10
    amplitude = amplrec*1.2
    durrec = np.random.rand(n)*.6
    dur = durrec*.5
    edinj = np.random.rand(n)*5000
    edrec = edinj*.8
    dftest0 = pd.DataFrame({"ampl_rec":amplrec,
                           "amplitude":amplitude,
                          "duration_d":dur,
                          "ed_inj":edinj,
                          "ed_rec":edrec,
                          "dur":durrec,
                          "rec":(np.random.rand(n) > 0.5).astype(float)})


    for t, vali, res in [("recovery_probability", "rec", .5),
                   ("ed_ratio", "edrat", .8),
                   ("amplitude_ratio", "amplrat", 10./12.),
                   ("duration_ratio","durrat", 2)]:
        resdf, val = tile_up_injection_recovery(dftest0, t, ampl="amplitude", dur="duration_d",
                                       otherfunc = "count",
                                       ampl_bins=np.linspace(0, 12, 11),
                                       dur_bins=np.linspace(0, .3, 11))

        assert resdf["count"].values.shape[0] == 100
        assert resdf["count"].mean() == 1000.
        assert val == vali
        assert resdf[val].median() == pytest.approx(res, rel=1e-2)

    with pytest.raises((KeyError, AttributeError)):
        dftest1 = pd.DataFrame()
        resdf, val = tile_up_injection_recovery(dftest1, "recovery_probability")

    dftest0.loc[dftest0.rec==0.,["ampl_rec","dur","ed_rec"]] = np.nan
    resdf, val = tile_up_injection_recovery(dftest0, "recovery_probability")

    for t, vali, res, counts in [("recovery_probability", "rec", .5, 1000),
                   ("ed_ratio", "edrat", .8, 500),
                   ("amplitude_ratio", "amplrat", 10./12., 500),
                   ("duration_ratio","durrat", 2, 500)]:
        resdf, val = tile_up_injection_recovery(dftest0, t, ampl="amplitude", dur="duration_d",
                                       otherfunc = "count",
                                       ampl_bins=np.linspace(0, 12, 11),
                                       dur_bins=np.linspace(0, .3, 11))
        print(t, resdf["count"].median())
        assert resdf["count"].values.shape[0] == 100
        assert resdf["count"].median() == pytest.approx(counts, rel=1e-2)
        assert val == vali
        assert resdf[val].median() == pytest.approx(res, rel=1e-2)


def test_multiindex_into_df_with_nans():
    np.random.seed(30)
    i1 = pd.interval_range(start=0, end=5)
    i2 = pd.interval_range(start=0, end=8)
    index = pd.MultiIndex.from_product([i1,i2], names=['ampl_rec', 'dur'])
    testdf = pd.DataFrame(np.random.rand(40), index=index, columns=["edrat"])
    s = pd.Series({"ampl_rec":.1, "dur":0.5})
    assert multiindex_into_df_with_nans(s, testdf) == pytest.approx(.6441435606)
    for v1,v2 in [(np.nan, .5), (1., np.nan),
                  (np.nan, np.nan), (6,3), (9,9)]:
        s = pd.Series({"ampl_rec":v1, "dur":v2})
        assert np.isnan(multiindex_into_df_with_nans(s, testdf))


def test_percentile():
    s = pd.Series(np.linspace(0,10,11), name="smth")
    assert percentile(s,50) == 5.
    assert percentile(s,16) == 1.6
    assert np.isnan(percentile(s, np.nan))
    s = pd.Series(np.zeros(10), name="smth")
    assert percentile(s,50) == percentile(s,70)
    s[:] = np.nan
    assert np.isnan(percentile(s,16))
    s[2] = 1.
    assert percentile(s,1) == percentile(s,99)
    


def test_plot_heatmap():

    df = pd.read_csv(os.path.join(PACKAGEDIR,
                                          "tests/testfiles/injrec_TIC1539914_s9.csv"))
    dft, val = tile_up_injection_recovery(df,"ed_ratio")
    assert isinstance(plot_heatmap(dft, val), matplotlib.figure.Figure)

    dft, val = tile_up_injection_recovery(df,"recovery_probability")
    assert isinstance(plot_heatmap(dft, val), matplotlib.figure.Figure)

    dft, val = tile_up_injection_recovery(df,"amplitude_ratio")
    assert isinstance(plot_heatmap(dft, val), matplotlib.figure.Figure)

    dft, val = tile_up_injection_recovery(df,"duration_ratio")
    assert isinstance(plot_heatmap(dft, val), matplotlib.figure.Figure)



def test__heatmap():
    # Create a minimal empty light curve with an ID
    flcd = FlareLightCurve(targetid="GJ 1243", time=np.linspace(10,1))
    ampl_bins, dur_bins, flares_per_bin = None, None, 20
    
    # If inj-rec data are missing, throw an error.
    with pytest.raises(AttributeError) as err:
        _heatmap(flcd, "recovery_probability", ampl_bins, dur_bins, flares_per_bin)
    
    # Path to test file
    path = "altaipony/tests/testfiles/gj1243_injrec.csv"
    flcd.load_injrec_data(path)
    
    # Test the default case
  #  _heatmap(flcd, "recovery_probability", ampl_bins, dur_bins, flares_per_bin)
    _heatmap(flcd, "ed_ratio", ampl_bins, dur_bins, flares_per_bin)
    
    # Check other cases of bin specifications
    for typ in ["recovery_probability", "ed_ratio"]:
        # only dur_bins
        ampl_bins, dur_bins, flares_per_bin = None, 5, 20
        _heatmap(flcd, typ, ampl_bins, dur_bins, flares_per_bin)

        # only ampl_bins
        ampl_bins, dur_bins, flares_per_bin = 3, None, 20
        _heatmap(flcd, typ, ampl_bins, dur_bins, flares_per_bin)

        # both dur_bins and ampl_bins
        ampl_bins, dur_bins, flares_per_bin = 3, 9, None
        _heatmap(flcd, typ, ampl_bins, dur_bins, flares_per_bin)
        
        # one of the two is defined as array
        ampl_bins, dur_bins, flares_per_bin = 3, np.linspace(5e-4,.004,20), None
        _heatmap(flcd, typ, ampl_bins, dur_bins, flares_per_bin)
        
        # It should also work if you give an array for one bin only
        ampl_bins, dur_bins, flares_per_bin = None, np.linspace(5e-4,.004,20), 7
        _heatmap(flcd, typ, ampl_bins, dur_bins, flares_per_bin)
        
        
def test_setup_bins():
    # Test a number of cases that could possibly be passed.
    
    # A failing example
    # --------------------------------------------------------------
    with pytest.raises(ValueError) as e:
        val = np.linspace(0,130,30)
        injrec = pd.DataFrame({"duration_d": val, "dur": val,
                               "amplitude": val, "ampl_rec": val })
        flares = pd.DataFrame({"dur": val, "ampl_rec": val })
        setup_bins(injrec, flares, ampl_bins=None, 
                   dur_bins=None, flares_per_bin=None)

    # A working example
    # --------------------------------------------------------------

    # Hybrid use case

    # Setup values
    val = np.linspace(0,130,30)
    injrec = pd.DataFrame({"duration_d": val, "dur": val, 
                           "amplitude": val, "ampl_rec": val })
    flares = pd.DataFrame({"dur": val, "ampl_rec": val })


    # Run setup_bins
    a, d = setup_bins(injrec, flares, ampl_bins=np.linspace(0,30,10),
                      dur_bins=3, flares_per_bin=None)


    # Do some checks
    assert len(a) == 10
    assert len(d) == 3
    assert (a == np.linspace(0,30,10)).all()
    assert (d == np.linspace(0,130,3)).all()

    # Another working example
    # --------------------------------------------------------------
    # Hybrid use case

    # Setup values
    val = np.linspace(0,130,300)
    injrec = pd.DataFrame({"duration_d": val, "dur": val, 
                           "amplitude": val, "ampl_rec": val })
    flares = pd.DataFrame({"dur": val, "ampl_rec": val })

    # Run setup_bins
    a, d = setup_bins(injrec, flares, ampl_bins=np.linspace(0,30,10),
                      dur_bins=None, flares_per_bin=3)

    assert len(a) == 10
    assert len(d) == 10
    assert (a == np.linspace(0, 30, len(a))).all()
    assert (d == np.linspace(0, 130, len(d))).all()

    # The lazy example
    # --------------------------------------------------------------
    # Just pass a number of flares per bin

    # Setup values
    val = np.linspace(0,130,300)
    injrec = pd.DataFrame({"duration_d": val, "dur": val, 
                           "amplitude": val, "ampl_rec": val })
    flares = pd.DataFrame({"dur": val, "ampl_rec": val })

    # Run setup_bins
    a, d = setup_bins(injrec, flares, ampl_bins=None,
                      dur_bins=None, flares_per_bin=3)

    # Do some checks
    assert len(a) == 10
    assert len(d) == 10
    assert (a == np.linspace(0, 130, len(a))).all()
    assert (d == np.linspace(0, 130, len(d))).all()

    # The control freak example
    # --------------------------------------------------------------
    # Set the bin edges manually.

    # Setup values
    val = np.linspace(0,130,300)
    injrec = pd.DataFrame({"duration_d": val, "dur": val, 
                           "amplitude": val, "ampl_rec": val })
    flares = pd.DataFrame({"dur": val, "ampl_rec": val })

    # Run setup_bins
    abins, dbins = [2,30,88,210], [3,40,220,780]
    a, d = setup_bins(injrec, flares, ampl_bins=abins,
                      dur_bins=dbins, flares_per_bin=3)

    # Do some checks
    assert len(a) == 4
    assert len(d) == 4

    assert a == abins
    assert d == dbins
