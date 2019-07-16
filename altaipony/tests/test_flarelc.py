import numpy as np
import pytest
from astropy.io.fits.hdu.hdulist import fitsopen
from inspect import currentframe, getframeinfo

from ..flarelc import FlareLightCurve
from ..lcio import from_K2SC_file

from .. import PACKAGEDIR
from . import test_ids, test_paths

def test_get_saturation():
    flc = mock_flc(detrended=True)
    flc = flc.find_flares()
    r1 = flc.get_saturation()
    assert r1.flares.saturation_f10.iloc[0] == False
    r2 = flc.get_saturation(return_level=True)
    assert r2.flares.saturation_f10.iloc[0] == pytest.approx(0.0495,1e-2)
    r3 = flc.get_saturation(factor=1e-2)
    assert r3.flares['saturation_f0.01'].iloc[0] == True

def test_mark_flagged_flares():
    flc = mock_flc(detrended=True)
    flc = flc.find_flares()
    flc = flc.mark_flagged_flares(explain=True)
    assert flc.flares.quality.iloc[0] == 1152
    s1 = "Sudden sensitivity dropout, Cosmic ray in optimal aperture"
    s2 = "Cosmic ray in optimal aperture, Sudden sensitivity dropout"
    qs = flc.flares.explanation.iloc[0]
    assert ((qs == s1) | (qs == s2))

def test_sample_flare_recovery():
    flc = mock_flc(detrended=True)
    data, fflc = flc.sample_flare_recovery(iterations=2)
    #make sure no flares are injected overlapping true flares
    print(data.istart)
    assert data[(data.istart > 14) & (data.istart < 19)].shape[0] == 0
    #test if all injected event are covered in the merged flares:
    assert data.shape[0] == 2
    assert fflc.gaps == [(0, 1000)]
    assert np.median(fflc.it_med) == pytest.approx(500.005274113832)


def test_characterize_flares():
    # flc = mock_flc(detrended=True)
    # lc = flc.characterize_flares(iterations=1, d=True, fakefreq=1.2, seed=781)
    # assert lc.flares.loc[0, 'rec_prob'] == pytest.approx(0.66666666666)
    # assert lc.flares.loc[0, 'ed_rec'] == pytest.approx(3455.887599271639)
    # assert lc.flares.loc[0, 'ed_rec_corr'] == pytest.approx(6524.739276618502)
    pass

def test_repr():
    pass

def test_getitem():
    pass

def mock_flc(origin='TPF', detrended=False, ampl=1., dur=1):
    """
    Mocks a FlareLightCurve with a sinusoid variation and a single positive outlier.

    Parameter
    -----------
    origin : 'TPF' or str
        Mocks a specific origin, such as 'KLC', 'FLC' etc.
    detrended : False or bool
        If False, a sinusoid signal is added to the mock light curve.

    Return
    -------
    FlareLightCurve
    """
    n = 1000
    time = np.arange(0, n/48, 1./48.)
    pixel_time = np.outer(time,np.full((3,3), 1)).reshape((1000,3,3))
    np.random.seed(13854)

    pipeline_mask = np.array([[False, False, False],
                              [False, True,  False],
                              [False, False, False],])
    quality = np.zeros_like(time)
    np.random.seed(33)
    flux_err = np.random.rand(n)/100.
    if detrended==False:
        flux = np.sin(time/2)*7. + 500. +flux_err
        pixel_flux = np.random.rand(len(time),3,3)/100.+500.+np.sin(pixel_time/2)*7.
        pixel_flux_err = np.random.rand(len(time),3,3)/100.
    else:
        flux = 500. + flux_err
        pixel_flux = np.random.rand(len(time),3,3)/100.+500.
        pixel_flux_err = np.random.rand(len(time),3,3)/100.
    flux[15:15+dur] += 500.*ampl
    flux[15+dur:15+2*dur] += 250.*ampl
    flux[15+2*dur:15+3*dur] += 130.*ampl
    flux[15+3*dur:15+4*dur] += 80.*ampl
    quality[17] = 1024
    quality[18] = 128
    keys = {'flux' : flux, 'flux_err' : flux_err, 'time' : time,
            'pos_corr1' : np.zeros(n), 'pos_corr2' : np.zeros(n),
            'cadenceno' : np.arange(n), 'targetid' : 800000000,
            'origin' : origin, 'it_med' : np.full_like(time,500.005),
            'quality' : quality, 'pipeline_mask' : pipeline_mask,
            'pixel_flux' : pixel_flux, 'campaign' : 5, 'ra' : 22.,
            'dec' : 22., 'mission' : 'K2', 'channel' : 55, 
            'pixel_flux_err' : pixel_flux_err}

    if detrended == False:
        flc = FlareLightCurve(**keys)
    else:
        flc = FlareLightCurve(detrended_flux=flux,
                              detrended_flux_err=flux_err,
                              **keys)
    return flc

def test_invalid_lightcurve():
    """Invalid FlareLightCurves should not be allowed."""
    err_string = ("Input arrays have different lengths."
                  " len(time)=5, len(flux)=4")
    time = np.array([1, 2, 3, 4, 5])
    flux = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError) as err:
        FlareLightCurve(time=time, flux=flux)
    assert err_string == err.value.args[0]

def test_find_gaps():
    lc = from_K2SC_file(test_paths[0])
    lc = lc.find_gaps()
    assert lc.gaps == [(0, 2505), (2505, 3293)] #[(0, 2582), (2582, 3424)]

def test_detrend(**kwargs):
    flc = mock_flc()
    try:
        flc = flc.detrend(de_niter=3,**kwargs)
        assert flc.detrended_flux.shape == flc.flux.shape
        assert flc.pv[0] == pytest.approx(-3.895176160613472, rel=0.1)
    except np.linalg.linalg.LinAlgError:
        warning.warn('Detrending of mock LC failed, this happens.')
        pass

    #test non TPF derived LC fails
    #test the shapes are the same for all
    # test that the necessary attributes are kept

def test_detrend_IO():
    #the mock_flc needs ra, dec, mission, and channel only for k2sc detrending!
    test_detrend(save_k2sc=True, folder='{}/tests/testfiles/'.format(PACKAGEDIR))
    flc = fitsopen('{}/tests/testfiles/pony_k2sc_k2_llc_800000000-c05_kepler_v2_lc.fits'.format(PACKAGEDIR))
    flc = flc[1].data
    mockflc = mock_flc()
    print (flc['TRTIME']-mockflc.flux)
    assert (flc['TIME'] == mockflc.time).all()
    assert (flc['CADENCE'] == mockflc.cadenceno).all()

def test_detrend_fails():
    """If detrend fails, an error is raised with given string."""
    flc =  mock_flc(origin='KLC')
    err_string = ('Only KeplerTargetPixelFile derived FlareLightCurves can be'
              ' passed to detrend().')
    with pytest.raises(ValueError) as err:
        flc.detrend(de_niter=3)
    assert err_string == err.value.args[0]

def test_find_flares():
    """Test that an obvious flare is recovered sufficiently well."""
    flc = mock_flc(detrended=True)
    flc = flc.find_flares()
    assert flc.flares.loc[0,'ed_rec'] == pytest.approx(3455.8875941, rel=1e-4)
    assert flc.flares['ed_rec_err'][0] < flc.flares['ed_rec'][0]
    assert flc.flares['istart'][0] == 15
    assert flc.flares['istop'][0] == 19
    assert flc.flares['cstop'][0] == 19
    assert flc.flares['cstart'][0] == 15
    assert flc.flares['tstart'][0] == pytest.approx(0.3125, rel=1e-4)
    assert flc.flares['tstop'][0] == pytest.approx(0.395833, rel=1e-4)
    
def test_append():
    flc1 = mock_flc(detrended=True)
    flc2 = mock_flc()
    flc = flc1.append(flc2)
    assert flc.flux.shape[0] == 2000
    assert flc.flux_err.shape[0] == 2000
    assert flc.detrended_flux.shape[0] == 2000
    assert flc.detrended_flux.shape[0] == 2000
    assert flc.pixel_flux.shape[0] == 2000    
    assert flc.pixel_flux_err.shape[0] == 2000
    assert flc.it_med.size == 2000

def test_inject_fake_flares():
    flc = mock_flc(detrended=True)
    np.random.seed(84712)
    flc = flc.find_gaps()
    fake_flc = flc.inject_fake_flares()
    # make sure you inject only one flare per LC
    assert len(fake_flc.gaps) == fake_flc.fake_flares.shape[0]
    assert (set(fake_flc.fake_flares.columns.values.tolist()) == 
            {'amplitude', 'duration_d', 'ed_inj', 'peak_time'})
    assert fake_flc.detrended_flux_err.all() >= 1e-10
    assert fake_flc.detrended_flux.all() <= 1.
    assert fake_flc.detrended_flux.shape == flc.detrended_flux.shape
    flc = mock_flc(detrended=False)
    np.random.seed(84712)
    flc = flc.find_gaps()
    fake_flc = flc.inject_fake_flares(inject_before_detrending=True)

    # make sure you inject only one flare per LC
    assert len(fake_flc.gaps) == fake_flc.fake_flares.shape[0]
    assert (set(fake_flc.fake_flares.columns.values.tolist()) == 
            {'amplitude', 'duration_d', 'ed_inj', 'peak_time'})
    assert fake_flc.flux_err.all() >= 1e-10
    assert fake_flc.flux.all() <= 1.
    assert fake_flc.flux.shape == flc.flux.shape


def test_characterize_one_flare():
    flc = mock_flc(detrended=True)
    pass
