"""Tests of the contract of ``custom_detrending``: the detrended flux is
relative flux with a baseline of 1, independent of how the input flux is
scaled, both with and without the GP step."""

import numpy as np
import pytest

from altaipony.detrending import custom_detrending, estimate_detrended_noise

pytestmark = pytest.mark.slow

NOISE = 1e-3
FLARES = [(2.3, 0.02, 0.01), (5.7, 0.04, 0.02)]   # (tpeak, fwhm, ampl), relative flux
SCALES = (1.0, 3400.0)                            # normalised, and e-/s-like
# spot amplitude: 2 % is detected as periodic (GP path), 0.2 % is not
STARS = {"periodic": 0.02, "flat": 0.002}


def _quiet(time, table, pad_before=0.05, pad_after=0.3):
    """Cadences well away from the injected flares."""
    quiet = np.ones(len(time), dtype=bool)
    for a, b in table[["t_start", "t_end"]].values:
        quiet &= ~((time >= a - pad_before) & (time <= b + pad_after))
    return quiet


def _values(x):
    return np.asarray(getattr(x, "value", x), dtype=float)


@pytest.fixture(scope="module")
def detrended(make_lc):
    """Each star (8 days of 2-min data, spots at P = 1.3 d, white noise, two
    flares) detrended at each flux scale."""
    out = {}
    for star, amp in STARS.items():
        for scale in SCALES:
            lc, table = make_lc(modes=[dict(period=1.3, amplitude=amp)], flares=FLARES,
                                noise=NOISE, scale=scale)
            out[star, scale] = (custom_detrending(lc, baseline_method="detrender"), table)
    return out


def test_stars_take_the_intended_path(detrended):
    """The periodic star gets a GP model, the flat one does not."""
    for scale in SCALES:
        assert np.isfinite(_values(detrended["periodic", scale][0].gp_model)).any()
        assert np.isnan(_values(detrended["flat", scale][0].gp_model)).all()


@pytest.mark.parametrize("star", STARS)
@pytest.mark.parametrize("scale", SCALES)
def test_quiet_baseline_is_one(detrended, star, scale):
    """Away from flares, the detrended flux sits at 1 (relative flux)."""
    lc, table = detrended[star, scale]
    det = _values(lc.detrended_flux)
    quiet = _quiet(lc.time.value, table)
    assert abs(np.nanmedian(det[quiet]) - 1.0) < 0.05 * NOISE
    # it_med describes the final detrended flux
    assert abs(np.nanmedian(_values(lc.it_med)) - 1.0) < 0.05 * NOISE


@pytest.mark.parametrize("star", STARS)
def test_detrended_flux_is_scale_invariant(detrended, star):
    """Multiplying the input flux by a constant does not change the relative
    detrended flux."""
    lc1, _ = detrended[star, 1.0]
    lc2, _ = detrended[star, 3400.0]
    np.testing.assert_allclose(_values(lc2.detrended_flux), _values(lc1.detrended_flux),
                               rtol=0, atol=0.05 * NOISE)


@pytest.mark.parametrize("star", STARS)
def test_recovered_amplitude_is_relative(detrended, star):
    """find_flares reports relative amplitudes, the same at both scales and
    close to the injected ones (the template peaks at ~0.95 ampl)."""
    ampl = {}
    for scale in SCALES:
        lc = detrended[star, scale][0].copy()
        lc.detrended_flux = _values(lc.detrended_flux)
        lc = estimate_detrended_noise(lc)
        # merge_sigma=1 joins tail fragments of the longer flare
        flares = lc.find_flares(merge_sigma=1.).flares.sort_values("tstart")
        assert len(flares) == len(FLARES)
        ampl[scale] = flares.ampl_rec.to_numpy(dtype=float)
    expected = np.array([0.95 * a for _, _, a in FLARES])
    np.testing.assert_allclose(ampl[1.0], expected, rtol=0.25)
    np.testing.assert_allclose(ampl[3400.0], ampl[1.0], rtol=0.05)
