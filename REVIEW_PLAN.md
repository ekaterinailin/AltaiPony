# Review plan: detrending and flare detection

A step-by-step read-through of `altaipony/detrending` and the flare detection
in `altaipony/altai.py`, ordered along the data flow and organised around one
question per step: *is the science sound?* Every step ends with tests that pin
down the behaviour you have convinced yourself of. Detrending tests go into
`altaipony/tests/detrending/`, detection tests into
`altaipony/tests/test_altai.py`.

Line numbers were last refreshed on 2026-10-13 (iterated matched filter,
window detector with local beta, detection extension). Current benchmark
numbers are in `TODO.md`.

**How to use this plan**

- Work through the steps in order. Each step has three parts:
  - **Read**: the code to go through.
  - **Check**: the questions to answer. Items marked ⚠ are suspicions noticed
    along the way: hypotheses to confirm or refute, not bugs.
  - **Tests**: what to write once you understand the behaviour.
- When a check turns up a real problem, first write a test that shows it
  failing, then fix the code, then watch the test pass.
- Tick the boxes as you go.

---

## Step 0: Test setup (done)

- [x] `altaipony/tests/detrending/conftest.py` provides the session fixture
      `make_lc(duration, cadence_min, gaps, modes, flares, noise, scale,
      seed)`, built on `altaipony.synthetic` (`make_time_grid`,
      `sinusoidal_variability` with optional P/2 harmonics, `inject_flares`
      with the Tovar Mendoza et al. 2022 template: peak ~0.95 `ampl`, FWHM
      ~1.09 `fwhm`, ED ~2.0 `ampl * fwhm`), plus `synthetic_lc` (8 days,
      e-/s at 3400, spots, a gap, two flares). `test_make_lc.py` checks it.
- [x] `generate_synthetic_lc` (injection-recovery sample): e-/s flux with a
      quiescent level of 1000–100000 e-/s, log-uniform mode amplitudes
      0.01–5 %, periods 0.25–10 d, P/2 harmonics, 20 % spotless stars.
- [x] A `slow` marker for tests that run the full pipeline.
- [x] Memory-capped runs with `scripts/guarded_run.sh` (cap, `choom`,
      watchdog), one at a time:
  ```bash
  scripts/guarded_run.sh -- python -m pytest altaipony/tests/detrending -m "not slow"
  cd notebooks && ../scripts/guarded_run.sh -w 2000 -l injrec.log -- python -u injection_recovery.py
  ```

## Step 1: The pipeline's contract

**Read:** `pipeline.py`, `custom_detrending` (32–247) top to bottom, without
going into the helpers yet; then `_apply_gp_step` (309–470).

**Check:**
- What exactly does `custom_detrending` promise to return? Write it down in
  one sentence: what `detrended_flux` means, its zero point, its units, and
  which other attributes (`it_med`, `gp_model`) are set.
- ✓ **Zero point and scale** (resolved): both paths return relative flux at
  1.0, the GP path as `1 + (flux − model) / level` (455–461), the non-GP
  path divided by its global iterative median (230–237).
- ✓ **Unit** (resolved 2026-10-15): `detrended_flux`, `detrended_flux_err`
  and `it_med` are dimensionless relative flux, like lightkurve's
  `normalize()`.
- Which steps only produce *masks*, and which produce the *output*? On the
  GP path, the polynomial baseline only feeds the GP's flare mask and the
  periodogram.

**Tests** (`test_pipeline.py`):
- [ ] Output has the same length and time stamps as the input, keeps NaNs
      where the input had them, and sets `it_med` and `gp_model`.
- [x] Zero point at the documented level for the GP path and the non-GP path
      (`use_gp=False`).
- [x] Scale invariance: `scale=1` and `scale=3400` give the same relative
      flare amplitude and equivalent duration.
- [x] Fix the 4 Quantity failures (`detrended_flux` is dimensionless).

## Step 2: Gaps and cadence

**Read:** `pipeline.py` 165–171.

**Check:**
- ⚠ `dt = np.mean(np.diff(time))` (165) includes the gaps, which inflates
  `maxgap * dt`, the gap threshold. Should this be the median?
- Missing cadences are no longer interpolated (2026-10-13): the fill-in was
  needed by the removed Savitzky-Golay passes. The polynomial baseline and the
  GP work on irregular sampling; the matched filter correlates sample by
  sample, so its templates are slightly misaligned next to missing cadences.
  ⚠ Not yet tested with a benchmark that has missing cadences (the synthetic
  light curves have none).

**Tests** (`test_pipeline.py`):
- [ ] A light curve with one long gap is split into exactly two segments.
- [ ] Injection-recovery with randomly missing cadences (1–2 %, plus short
      dropouts) is not worse than without.

## Step 3: Detecting the rotation period

**Read:** `periodicity.py` (`lomb_scargle` 8–57, `detect_strong_periodicity`
60–124); call site `pipeline.py` 183–201.

**Settled by injection-recovery:** the GP runs for every star and only takes
its starting period from the highest peak; the FAP threshold gates the
second-period search and the residual-period term. The Baluev FAP is tiny for
almost everything, including flares alone on a spotless star (~1e-60, ~1e-33
with the baseline's flare mask removed).

**Check:**
- Search range is 0.1 d to half the light curve. What happens near its ends?
- On a spotless star the starting period comes from flares or noise. Is the
  GP's ±5 % period bound ever a problem there?

**Tests** (`test_periodicity.py`):
- [ ] A pure sine with P ∈ {0.3, 2, 8} d is recovered to within 1 %.
- [ ] White noise only → not significant (FAP > 1e-3).
- [ ] Light curve shorter than 2 × `period_min` → not periodic, no error.
- [ ] `lomb_scargle` drops non-finite points and never searches periods
      longer than the data span.

## Step 4: Polynomial baseline (`baselines/polynomial.py`)

The only baseline (spline, multi-sine and `auto` were removed: same recovery,
60–115 false positives vs. 7). It supplies the GP's starting flare mask.

**4a Preliminary flare mask:** `preliminary_flare_mask` (166–220),
`compute_rolling_local_sigma` (126–164).
- Is a 5σ cut on a rough 2-day polynomial residual conservative enough that
  real variability is never masked?

**4b Window fits and blending:** `fit_window_grid` (223–339),
`run_all_windows` (342–401), `score_segment` (440–483),
`assemble_combined_trend` (486–580).
- Degree-4 polynomials in 0.6/0.9/1.2 d windows, normal and half-shifted
  grids. Which timescales can they follow, and which flare durations do they
  eat into?
- Edge points are up-weighted 8× (`n_edge`, `edge_weight`). Why?
- `score_segment` weights (0.15 / 0.30 / 0.05) and the softmax are
  heuristic. Sensible?

**4c Second pass:** in `run_detrending` (766–949), the 3σ mask from the
first-pass residual and the refit.

**4d Final mask:** `build_safe_final_flare_mask`. (The residual centring
before it was removed on 2026-10-15: subtracting a constant does not change
a mask thresholded above the residual's own median.)
- The final mask falls back to the previous one above 5 % flagged cadences.
  Is 5 % right for very active stars?
- Masks wider than 3σ (tail extension, 2σ hysteresis, a 2.5σ threshold) were
  all tried and dropped: on fast rotators the extra cadences are rotation
  leftovers, and masking them creates false positives (`TODO.md`).

**Tests** (`test_polynomial.py`):
- [ ] Smooth trend (P = 3 d) + noise → residual scatter within 10 % of the
      noise.
- [ ] `build_safe_final_flare_mask` falls back to `previous_mask` when the
      candidate exceeds `max_mask_fraction`.
- [ ] `fit_lightcurve_detrender` maps results back onto the full grid with
      NaNs where the input had them.

## Step 5: Gaussian-process model (`gp.py`, `pipeline._apply_gp_step`)

**Read:** `_segment_edge_mask` (31–54), `_bin_training_per_segment`
(57–156), `fit_gp_rotation` (159–488, kernel bounds in `_rotation_bounds`
410–423, clipping 440–472); in `pipeline.py`, `_apply_gp_step` (309–470):
first fit (381), second-period search (394–406, `_second_period_candidates`
250–287), residual-period term (408–424, `_residual_period` 290–306),
matched-filter passes (426–446).

**Check:**
- **Training data.** The GP trains on the original flux minus the flare
  mask: the baseline's mask, then the matched filter's detections on the
  residual of the latest fit, added over up to 3 passes. With a *perfect*
  mask (the injected flare windows) the pipeline recovers as many flares as
  perfect detrending, so the GP does not absorb masked flares: every
  remaining detrending loss is a flare the mask did not cover.
- **One-sided clipping** (3 rounds, 3σ above the model, on binned training
  points). It is essential (without it 40–49 weak long flares are lost: a
  flare at 1σ per cadence stands out at ~3σ in 10-cadence bins). But it
  cannot tell a weak flare from a slightly low model: where the model sags,
  the points above it are clipped and it sags further. This is largest at
  segment edges (−1.4 to −1.9σ over ~0.1 d in LCs 81, 96, 73). The window
  detector's local beta now absorbs the resulting false positives; protecting
  the edges from the clip and clipping relative to each point's error were
  tried and dropped.
- ⚠ **Edge anchors override the flare mask.** A flare at a segment edge is
  forced into the training set; does the GP absorb it?
- **Binning by 10.** Errors propagated as sqrt(Σσ²)/n; the outermost 10
  cadences of each segment stay unbinned.
- **Kernel and bounds:** P within ±5 % of the periodogram's period; Q0 at
  least `2π · τ_min / P − 1/2` (and ≥ 1) with τ_min = 2 d
  (`min_damping_days`); dQ in [1, 200]. Before τ_min, Q0 sat at its lower
  bound in 49 of 100 stars and P at its bound in 42. Which bounds hide a
  misfit? The one remaining false positive (LC 21, a turning point of a fast,
  strong modulation) suggests the GP is slightly too smooth there; τ_min = 1
  d was not better overall.
- **Extra terms** (second period, residual period) are kept if they lower the
  residual scatter, not by a likelihood comparison. Can a term ever absorb
  flares, e.g. with many flares at roughly regular spacing?
- ⚠ **Short light curves.** On 8–12-day two-period stars the second-period
  candidates are often sidelobes just outside the ±10 % exclusion. Should the
  exclusion width scale with 1/T?
- **Offset** (`pipeline.py` 455): `median(flux − gp)` over unmasked points.

**Tests** (`test_gp.py`):
- [x] Two unrelated spot modes (3 d and 0.69 d): one rotation term leaves
      > 3× the white noise, the second term brings it within 1.5×.
- [ ] A fast, strongly modulated rotator (P ≈ 6 h, amplitude rising to
      ~12 %) leaves residual scatter within ~2× the noise and no spurious
      flares at the maxima.
- [ ] `_bin_training_per_segment` keeps both segment edges at full
      resolution and loses no points.
- [ ] `fit_gp_rotation` on a quasi-periodic signal + flares follows the
      modulation, not the flares.
- [ ] A flare at a segment edge: document what happens.
- [ ] Residual-period term: added for a star with an uncovered harmonic, not
      for white noise and flares.

## Step 6: Matched filter (`matchedfilter.py`)

**Read:** `matched_filter_statistic` (167–247), `matched_filter_flare_mask`
(250–384); the passes in `pipeline.py` (426–446).

**Check:**
- Detections above S/N 5 on the residual of the latest GP fit, grown from the
  peak while the whitened residual stays above 1.5σ, padded by 2 cadences,
  capped at 0.6 d per side. Passes continue (up to 3) while they add
  cadences.
- ⚠ The mask often covers only a flare's core (median 8.5 % of the 10 %
  window for lost weak long flares). Lowering the threshold, growing further,
  or masking the template's extent were tried: the latter two recover more
  flares but also mask rotation leftovers on fast rotators and create false
  positives. A better mask would have to tell flares from leftovers.
- The template is Davenport et al. (2014); injected Mendoza flares match best
  at about twice their FWHM.

**Tests** (`test_matchedfilter.py`):
- [ ] A flare at S/N 5 per cadence is flagged; white noise is not.
- [ ] A slow sinusoid of 1σ amplitude does not trigger it.

## Step 7: Noise estimate (`noise.py`)

**Read:** `estimate_detrended_noise` (10–105).

**Check:**
- Rolling std over 100 cadences after a one-sided 2.5σ clip and subtraction
  of the iterative median. ⚠ It is 2.7 % low on flare-free synthetic light
  curves (the clip truncates the distribution), so ~2.2 per mille of noise
  points exceed "3σ" instead of 1.35. The window detector's local beta
  partly corrects for this.
- ✓ Segment starts inside a flare: the rolling std is now filled in both
  directions (98).

**Tests** (`test_noise.py`, which already checks σ = 40 within ±5, with and
without a flare, and with NaNs):
- [ ] White noise σ → estimate within 5 % of σ.
- [ ] White noise + flares → estimate within 10 % of σ.
- [ ] A segment that starts during a flare gets finite noise everywhere.

## Step 8: Flare detection (`altai.py`)

**Read:** `find_flares` (272–389), `find_flares_in_cont_obs_period`
(94–218), `_window_flare_candidates` (40–92), `_local_beta` (14–38),
`_merge_elevated_gaps` (220–241), `_extend_candidates` (243–270).

**Check:**
- **Classic rule** (`max_window=None`): 3 consecutive cadences above 3σ
  (N1 = 3; N2 = 2 amounts to the same with σ = the error). Its index
  arithmetic also flags the cadence after each run.
- **Window detector** (default, `max_window=10`): windows of 3–10 cadences whose
  Fisher-combined noise probability (points capped at 3σ, below the median
  counted as zero excess) is below that of 3 points at 3σ (5.35e-7). At n = 3
  it equals the classic rule. Is Fisher's method the right combination, and
  is a fixed probability per window right given the number of windows and
  lengths tested (1 of 200 white-noise light curves gets a detection)?
- **Local beta** (windows longer than 3): excursions divided by the robust
  scatter of n-cadence block means within `beta_window_days` (1 day) relative
  to white noise, at least 1. It is noisy (95th percentile ~1.26 for white
  noise) and costs ~12 flares, but removes the GP's correlated residuals.
  Could it be estimated with less noise (overlapping blocks, a model of the
  residual's correlation)?
- **Merging** (default `merge_sigma=0.25`): fragments are joined
  when the flux between them averages 0.25σ. ⚠ Can that join two separate
  flares into one detection?
- **Extension** (default `extend_sigma=1.0`): detections grow over adjacent cadences
  above 1σ, so they reach back over the peak of faint flares.

**Tests** (`test_altai.py`):
- [x] `_extend_candidates`: growth, joining, NaN stops it.
- [x] Window detector equals the classic rule at n = 3.
- [x] Detects 10 points at 2.5σ, ignores a single 10σ spike, nothing on
      white noise.
- [x] `_local_beta` ≈ 1 for white noise, > 1.5 with correlated residuals.
- [ ] `find_flares(max_window=..., beta_window_days=...)` converts the beta
      window to cadences per continuous window (20-s and 30-min cadence).

## Step 9: End to end

**Read:** `notebooks/injection_recovery.py` (`flare_finder_altai`,
`detect_flares`, `run_injection_recovery`), `fakeflares.match_flares`,
`notebooks/flare_finding_only_injection_recovery.py`.

**Check:**
- Matching: recovered = injected peak inside a detection; a detection with no
  injected peak is a false positive, even on a flare's tail. Most remaining
  false positives are such partial detections of weak flares; only one is
  outside injected flares.
- Do the recovery rates (`TODO.md`: S/N 2–5 78 %, 5–10 99.5 %, > 10 100 %)
  match what the detection threshold predicts?

**Tests** (`test_end_to_end.py`, marked `slow`):
- [ ] A small injection-recovery run (5 light curves, fixed seeds): ≥ 95 %
      recovery for S/N > 10 and no false positive outside injected flares.
- [ ] One fast rotator (P = 9 h): no false positives.

---

## Afterwards

- [ ] Move the remaining items from `TODO.md` into the matching steps, or
      close them.
- [ ] Update the docstrings to the contract written down in step 1.
- [x] Decide whether the window detector becomes `find_flares`' default
      (yes, 2026-10-13).
- [ ] Regenerate the tutorials' flare tables with the new default.
