import numpy as np
import pandas as pd

import pytest

from ..injrec import (characterize_flares,
                      tile_up_injection_recovery,
                      multiindex_into_df_with_nans,
                      percentile,
                     )

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