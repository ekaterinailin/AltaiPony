# TODO

## Failing tests

`test_customdetrend.py` has been dissolved into `altaipony/tests/detrending/`
(`test_pipeline.py`, `test_noise.py`, `test_spline.py`). 11 of its spline and
knot tests only failed because the spline API had changed (`_build_knot_points`
returns the knot scatter too, `_fit_single_spline` takes `smoothing`,
`best_params` has a `smoothing` key); they were updated and passed. Since
then the spline baseline and its tests have been removed (see Resolved). The
full suite now runs in about a minute: 240 passed, 4 failed.

### `altaipony/tests/detrending/test_pipeline.py`

- **Quantity vs plain number** (4 tests): `custom_detrending` returns
  `detrended_flux` as an astropy Quantity labelled e-/s (but holding relative
  flux), and these tests combine it with plain numbers, which raises
  `UnitConversionError`. Decide what type `detrended_flux` should have (review
  plan, step 1), then fix the tests or the code.
  - `TestCustomDetrending::test_measure_flare_basic`,
    `test_measure_flare_has_required_columns`,
    `test_measure_flare_calculates_duration`: add `500` to
    `flc.detrended_flux`.
  - `TestCustomDetrending::test_custom_detrending_removes_variability`:
    compares `np.nanstd(detrended_flux)` with the std of the plain flux.

## Other bugs

- `FlareLightCurve.find_cont_windows` raises `ValueError: flux and flux_err
  must have the same units` on the light curve built in
  `test_flarelc.py::test_detrend` (around line 245). Since
  `find_iterative_median` no longer calls it by default, that test passes;
  `find_iterative_median(per_window=True)` would still hit the error there.
- `fit_flares.py` passes its fitted "fwhm" to `flare_model_davenport2014`,
  whose third argument is a duration (FWHM = duration / 2). The FWHMs it
  reports are therefore twice the actual FWHM (a 160-min flare came out as
  ~260 min).
- `fakeflares.flare_model_mendoza2022(..., upsample=True)` raises
  `ValueError: The smallest edge difference is numerically 0.` whenever
  `tpeak` coincides with a cadence (the bin edges are built from
  `np.diff(np.abs(t_new))`). `altaipony.synthetic` does not use `upsample`.

## Resolved

- The out-of-memory kills of `test_custom_detrending_handles_nans` and
  `test_full_detrending_pipeline` are gone: the GP no longer computes its
  predictive variance by default, which needed several GB on long light
  curves. Both tests now pass.
- `fakeflares.flare_eqn` returned NaN for cadences long before the flare peak
  (exp overflow times erfc underflow), e.g. 12,650 of 18,000 cadences for a
  6-min flare in a 25-day light curve. `FlareLightCurve.inject_fake_flares`
  (default model `mendoza2022`) added those NaNs to the flux. Fixed: such
  values are now 0; all previously finite values are unchanged.

- `find_iterative_median` now computes one global median by default (the
  intended choice), with `per_window=True` as an option. Before, the result
  depended on whether `find_cont_windows()` had been called first, and
  `find_flares` got per-window medians. This also fixed
  `test_flarelc.py::test_detrend`.
- `custom_detrending` now returns relative flux with a quiet baseline of
  exactly 1 on both paths, independent of the input flux units. Before, the
  GP path returned "1 + residual in input units" (a 1 % flare in e-/s data at
  3400 e-/s came out as an amplitude of ~34), its baseline was off by the GP
  offset (0.1–0.25 sigma), and the non-GP path stayed at the input flux level.
  The GP is fitted on median-normalised flux, and `it_med` is recomputed on
  the final detrended flux. Tested in `tests/detrending/test_pipeline.py`.
  Confirmed end to end: the injection-recovery light curves are now in e-/s
  (baselines 1000–100000 e-/s), and recovery is flare-for-flare identical to
  the normalised run (636/787, 0 false positives).
- After the Savitzky-Golay passes, `lc4.flux_err` was not restored, so the GP
  got flux in input units but errors divided by the Savitzky-Golay trend
  (3400x too small for e-/s data). Fixed.
- The matched-filter flare mask ran on the baseline residual, so rotation
  the baseline could not follow (fast rotators with large, changing
  amplitudes) was masked at every maximum and hidden from the GP, leaving
  periodic false positives. It now runs on the residual of a first GP fit,
  and the GP is refitted with its detections (`_apply_gp_step`).
- A second, unrelated periodic signal (e.g. a 0.69 d mode next to a 3 d
  one) was left unmodelled by the single rotation term and produced periodic
  false positives. `custom_detrending` now adds a second rotation term at a
  period from the flux periodogram when it lowers the residual scatter
  (`gp_second_rotation=True`). See the review plan, step 9, for its
  limitation on short light curves.
- `custom_detrending` had four baselines (`baseline_method`: `auto`,
  `detrender`, `spline`, `multisine`). In injection-recovery all recovered
  about the same (662–668 of 787), but only the polynomial ("detrender") kept
  false positives low (7 vs. 60–115): the others built the GP's starting
  flare mask from their residual with `_identify_flare_mask`, masking the
  variability maxima they could not follow. The polynomial baseline is now
  the only one; `baselines/spline.py`, `baselines/multisine.py`,
  `baseline_method`, the multi-sine options and `_identify_flare_mask` were
  removed.
- The two Savitzky-Golay passes (6 h, 3 h) in `custom_detrending` changed
  nothing for GP stars and slightly lowered recovery for the others (42 vs.
  44 of 57 flares without them, no false positives either way). Removed,
  with `savgol1`, `savgol2`, `use_savgol`, `pad`, `max_sigma`, `longdecay`
  and `break_tolerance`; cadences filled by `interpolate_missing_cadences`
  are now dropped from the output explicitly (the passes used to do that).
- The 6–36 h sinusoid correction in the polynomial baseline changed nothing
  once the GP runs for every star (injection-recovery identical cadence for
  cadence). Removed (`multi_sinusoid_correction` and its settings).
- The periodicity gate: the semi-amplitude threshold was replaced by an S/N
  threshold (S/N > 3 lost 11 flares on the stars it excluded from the GP),
  then by the false-alarm test alone (+5 flares). Flares alone make a
  spotless star "periodic" (FAP ~1e-60, at roughly their spacing), even after
  masking them, but a test with 22 spotless stars showed the GP helps there
  too (133 vs. 119 of 164 flares, no false positives). So the GP now runs for
  every star; the periodogram (computed without the baseline's flare mask)
  only provides its starting period, and the S/N code was removed.

## Tried and dropped

Experiments evaluated with injection-recovery (100 LCs, seed 92) that gave no
clear net gain, and were removed to keep the code focused:

- Matched-filter FWHM grid: starting at 0.01 d, or ending at 0.2 d instead of
  0.15 d, changed at most one flare (0.2 d kept as the long end).
- Tail extension of the polynomial baseline's flare masks (extending each
  flagged stretch while the residual stays above 1σ): +2 flares but 12 false
  positives instead of 4, from leftover rotation on fast rotators.
- Fitting and subtracting flares before the GP (`fit_flares` MCMC: 1–6 min
  per light curve; least squares: fast). With baseline and flares fitted
  together, the polynomial dipped under the flares and they were
  over-subtracted (strong flares lost, more false positives); with a fixed
  baseline the fit under-fitted the tails (+7 / −7 flares).
- Doubling the polynomial windows to 0.8–1.6 d (200 LCs): +14 of 1550
  flares, mostly long ones, but 4 false positives instead of 2. The midpoint,
  0.6–1.2 d, is the default now; it was not compared with 0.4–0.8 d on its
  own.

## Remaining limitation

- Long flares (FWHM > 2 h) are recovered least often (~75 %). The polynomial
  baseline absorbs most of their decay tail (often >80 % of the flux) before
  the GP, because its windows bend into a decay that lasts several hours, and
  its 3σ mask covers only the core.

## Open, related

- `fit_flares.py` fits flares with the Davenport et al. (2014) template
  (`flare_model_davenport2014`). Switch it to the Tovar Mendoza et al. (2022)
  template (`fakeflares.flare_model_mendoza2022`), which the synthetic light
  curves and injection-recovery use (and fix the FWHM convention, see Other
  bugs). The matched filter (`matchedfilter.py`) also still uses Davenport
  (2014).

- `detrended_flux` is relative flux but still labelled e-/s (see review plan
  step 1 and the 4 Quantity test failures).
- The `find_cont_windows` unit error (see Other bugs) may come from
  `flux_err` being multiplied by e-/s twice somewhere, as
  `custom_detrending` did before the Savitzky-Golay passes (removed).
- `notebooks/injection_recovery.py` sleeps 5 s after every light curve, which
  adds ~8 min to a 100-LC run.
