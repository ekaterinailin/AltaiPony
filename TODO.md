# TODO

## Failing tests

`test_customdetrend.py` has been dissolved into `altaipony/tests/detrending/`
(`test_pipeline.py`, `test_noise.py`, `test_spline.py`). 11 of its spline and
knot tests only failed because the spline API had changed (`_build_knot_points`
returns the knot scatter too, `_fit_single_spline` takes `smoothing`,
`best_params` has a `smoothing` key); they were updated and passed. Since
then the spline baseline and its tests have been removed (see Resolved). The
full suite now runs in about a minute: 236 passed, 4 failed.

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

## Open, related

- `detrended_flux` is relative flux but still labelled e-/s (see review plan
  step 1 and the 4 Quantity test failures).
- `custom_detrending` multiplies `lc.flux_err` (already a Quantity) by e-/s
  again before the Savitzky-Golay passes, giving units (e-/s)^2. Harmless so
  far, but the same pattern may be behind the `find_cont_windows` unit error.
