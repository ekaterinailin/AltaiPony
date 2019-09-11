import os
import pytest

import numpy as np
import pandas as pd

from ..injrecanalysis import (characterize_flares,
                              tile_up_injection_recovery,
                              multiindex_into_df_with_nans,
                              percentile,
                              wrap_characterization_of_flares
                             )
                             
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

    with pytest.raises(KeyError):
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
