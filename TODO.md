# TODO

## Failing tests

These tests already failed on commit `783c698`, before the
`lightcurve_detrender` cleanup and the restructuring into
`altaipony/detrending/`. The full suite now runs in under a minute without
running out of memory: 245 passed, 16 failed.

### `altaipony/tests/test_customdetrend.py`

The tests import from the new locations: `altaipony.detrending`,
`altaipony.detrending.baselines.spline` and `altaipony.altai` (for
`measure_flare`).

- **Quantity vs plain number** (4 tests): `custom_detrending` returns
  `detrended_flux` as an astropy Quantity in e-/s, but these tests combine it
  with plain numbers, which raises `UnitConversionError`. Either use Quantities
  in the tests or decide whether `detrended_flux` should be a plain array.
  - `test_measure_flare_basic`, `test_measure_flare_has_required_columns`,
    `test_measure_flare_calculates_duration`: add `500` to
    `flc.detrended_flux` (around line 257).
  - `test_custom_detrending_removes_variability`: compares
    `np.nanstd(detrended_flux)` with the std of the plain flux.
- **Spline and knot tests** (12 tests):
  - `test_build_knot_points_basic`
  - `test_build_knot_points_boundary_values`
  - `test_build_knot_points_handles_nans`
  - `test_build_knot_points_percentile_affects_flux`
  - `test_build_knot_points_phase_offset`
  - `test_build_knot_points_short_segment`
  - `test_build_knot_points_sorted`
  - `test_fit_single_spline_handles_multiple_gaps`
  - `test_fit_single_spline_returns_model_and_newflux`
  - `test_fit_single_spline_short_segment_uses_median`
  - `test_fit_spline_best_params_keys`
  - `test_fit_spline_removes_trend`

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

## Open, related

- `detrended_flux` is relative flux but still labelled e-/s (see review plan
  step 1 and the 4 Quantity test failures).
- `custom_detrending` multiplies `lc.flux_err` (already a Quantity) by e-/s
  again before the Savitzky-Golay passes, giving units (e-/s)^2. Harmless so
  far, but the same pattern may be behind the `find_cont_windows` unit error.
