"""
UTF-8, Python 3

Injection-recovery test framework for flare finders.

Generates synthetic 25-day light curves with realistic noise, periodic
variability, and data gaps, injects flares using the Mendoza+2022 model
(both from ``altaipony.synthetic``),
runs a user-supplied flare finder, and returns a summary table of
recovered and missed flares plus false positives.

False-positive diagnostic plots
--------------------------------
When ``plot_fp=True`` (the default), a two-panel PNG is saved for every
false-positive detection:

    top panel   – the raw injected light curve in a context window around
                  the detection, with the FP window shaded red and any
                  overlapping injected flares shaded orange.
    bottom panel – the detrended light curve (and iterative-median
                  baseline, if provided) in the same window.

The bottom panel is only drawn when the flare finder returns a dict (see
the ``flare_finder`` parameter of ``run_injection_recovery``).  Returning
a plain list of detections still works and produces a single-panel plot.
"""

import os
import time as time_module

import numpy as np
import pandas as pd
from altaipony.detrending import estimate_detrended_noise, custom_detrending
from altaipony.flarelc import FlareLightCurve
from altaipony.synthetic import generate_synthetic_lc, inject_flares

import matplotlib.pyplot as plt

import astropy.units as u


# ---------------------------------------------------------------------------
# Matching injected → recovered
# ---------------------------------------------------------------------------

def _overlaps(a_start, a_end, b_start, b_end):
    """True if two intervals overlap."""
    return ((a_start >= b_start) and (a_start <= b_end)) | \
           ((a_end   >= b_start) and (a_end   <= b_end)) | \
           ((a_start <= b_start) and (a_end >= b_end))   | \
           ((b_start <= a_start) and (b_end >= a_end))


def match_flares(injected, recovered):
    """Match injected flares to recovered detections by time overlap.

    Parameters
    ----------
    injected : pd.DataFrame
        Output of ``inject_flares``.  Must have columns t_start, t_end.
    recovered : list of (float, float)
        Detected flare intervals (t_start, t_end) from the flare finder.

    Returns
    -------
    matched_injected : pd.DataFrame
        ``injected`` with extra columns: recovered (bool), rec_t_start,
        rec_t_end.
    fp_table : pd.DataFrame
        One row per false-positive detection (rec_t_start, rec_t_end).
    """
    rec = list(recovered)
    matched_rows = []
    used_rec = set()

    for _, inj in injected.iterrows():
        found = False
        for i, (rs, re) in enumerate(rec):
            if _overlaps(rs, re, inj.t_start, inj.t_end):
                matched_rows.append({**inj.to_dict(),
                                     'recovered': True,
                                     'rec_t_start': rs,
                                     'rec_t_end': re})
                used_rec.add(i)
                found = True
                break
        if not found:
            matched_rows.append({**inj.to_dict(),
                                 'recovered': False,
                                 'rec_t_start': np.nan,
                                 'rec_t_end': np.nan})

    # Build FP table from unmatched detections
    fp_rows = [{'rec_t_start': rec[i][0], 'rec_t_end': rec[i][1]}
               for i in range(len(rec)) if i not in used_rec]
    fp_table = pd.DataFrame(fp_rows, columns=['rec_t_start', 'rec_t_end'])

    return pd.DataFrame(matched_rows), fp_table


# ---------------------------------------------------------------------------
# False-positive diagnostic plots
# ---------------------------------------------------------------------------

def _to_array(x):
    """Return a plain numpy array whether x is a Quantity or already an array."""
    return x.value if hasattr(x, 'value') else np.asarray(x)


def _plot_false_positives(
    lc_id, time, flux_injected,
    detrended_time, detrended_flux, it_med,
    fp_df, injected_table, meta, plot_dir,
):
    """Save one diagnostic PNG per false-positive detection.

    Each figure has two panels (one if no detrended flux is available):

    Top panel
        Raw injected light curve in a context window around the detection.
        The FP window is shaded red; any injected flares whose windows
        overlap the context are shaded orange.  This lets you see whether
        the FP is driven by a residual flare wing, a noise spike, or
        genuine stellar variability that the detrender left in.

    Bottom panel
        Detrended light curve (and the iterative-median baseline in blue)
        in the same window.  Comparing the two panels immediately shows
        whether the detrender introduced the artefact or whether it was
        already present in the raw data.

    Parameters
    ----------
    lc_id : int
    time : ndarray
        Full time array of the raw light curve (days).
    flux_injected : ndarray
        Raw flux after flare injection.
    detrended_time : ndarray or None
        Time array of the detrended LC.  May differ in length from ``time``
        if the finder's detrending step interpolated or trimmed cadences.
    detrended_flux : ndarray or None
        Detrended flux values.  Pass None for a single-panel plot.
    it_med : ndarray or None
        Iterative-median baseline aligned to ``detrended_time``.
        Overlaid in blue on the bottom panel.  Pass None to omit.
    fp_df : pd.DataFrame
        False-positive table with columns rec_t_start and rec_t_end.
    injected_table : pd.DataFrame
        Injected flare table (t_start, t_end) for context annotation.
    meta : dict
        Light-curve metadata (noise_ppm, n_modes).
    plot_dir : str
        Directory in which to save PNGs (created if absent).
    """
    os.makedirs(plot_dir, exist_ok=True)

    has_detrended = (detrended_time is not None) and (detrended_flux is not None)
    if has_detrended:
        det_t = _to_array(detrended_time)
        det_f = _to_array(detrended_flux)
        med_f = _to_array(it_med) if it_med is not None else None

    for fp_idx, fp_row in fp_df.iterrows():
        fp_start = fp_row['rec_t_start']
        fp_end   = fp_row['rec_t_end']
        fp_mid   = 0.5 * (fp_start + fp_end)
        fp_dur   = max(fp_end - fp_start, 2.0 / 1440.0)  # minimum 2 min

        # Context window: at least 2 hours on each side, capped at 6 hours.
        # 10× the FP duration gives generous context for narrow spikes while
        # keeping the plot useful for longer, variability-driven FPs.
        half_win = np.clip(10.0 * fp_dur, 2.0 / 24.0, 6.0 / 24.0)
        t_lo = fp_mid - half_win
        t_hi = fp_mid + half_win

        # Injected flares whose windows overlap the context window
        nearby_inj = injected_table[
            (injected_table['t_end']   >= t_lo) &
            (injected_table['t_start'] <= t_hi)
        ]

        n_panels = 2 if has_detrended else 1
        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(12, 3.5 * n_panels),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]   # flatten to 1-D for uniform indexing

        # ── top panel: raw injected light curve ──────────────────────────
        ax0 = axes[0]
        win_raw  = (time >= t_lo) & (time <= t_hi)
        ax0.plot(time[win_raw], flux_injected[win_raw],
                 'k.', markersize=2, label='injected LC')

        # Shade injected flare windows for context — orange so they are
        # visually distinct from the red FP even when they overlap.
        legend_inj_done = False
        for _, inj in nearby_inj.iterrows():
            label = 'injected flare' if not legend_inj_done else None
            ax0.axvspan(inj['t_start'], inj['t_end'],
                        color='orange', alpha=0.35, label=label)
            legend_inj_done = True

        ax0.axvspan(fp_start, fp_end,
                    color='red', alpha=0.25, label='false positive')
        ax0.set_ylabel('Flux (injected)')
        ax0.set_title(
            f"LC {lc_id} | FP {fp_idx + 1} of {len(fp_df)} | "
            f"noise = {meta['noise_ppm']:.0f} ppm | modes = {meta['n_modes']} | "
            f"window [{fp_start:.4f}, {fp_end:.4f}] d"
        )
        ax0.legend(loc='upper right', fontsize=8, framealpha=0.7)

        # ── bottom panel: detrended light curve ───────────────────────────
        if has_detrended:
            ax1 = axes[1]
            win_det = (det_t >= t_lo) & (det_t <= t_hi)

            ax1.plot(det_t[win_det], det_f[win_det],
                     'k.', markersize=2, label='detrended')

            if med_f is not None:
                ax1.plot(det_t[win_det], med_f[win_det],
                         'b-', linewidth=1.2, label='iterative median')

            for _, inj in nearby_inj.iterrows():
                ax1.axvspan(inj['t_start'], inj['t_end'],
                            color='orange', alpha=0.35)

            ax1.axvspan(fp_start, fp_end, color='red', alpha=0.25)
            ax1.set_ylabel('Detrended flux')
            ax1.legend(loc='upper right', fontsize=8, framealpha=0.7)

        axes[-1].set_xlabel('Time (days)')
        plt.tight_layout()

        fname = (
            f"{plot_dir}/fp_lc{lc_id:03d}"
            f"_fp{fp_idx:02d}"
            f"_t{fp_mid:.4f}"
            f"_noise{meta['noise_ppm']:.0f}ppm.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main injection-recovery runner
# ---------------------------------------------------------------------------

def run_injection_recovery(
    flare_finder,
    n_lcs=50,
    duration_days=25.0,
    cadence_min=2.0,
    seed=70,
    plot_fp=True,
    fp_plot_dir='diag_plots',
):
    """Run a full injection-recovery experiment.

    Parameters
    ----------
    flare_finder : callable
        Function with signature::

            flare_finder(time, flux, meta, flare_table)
                -> list of (t_start, t_end)
                   OR
                   dict with keys:
                     'detections'     – list of (t_start, t_end)         [required]
                     'detrended_time' – ndarray aligned to detrended LC  [optional]
                     'detrended_flux' – ndarray of detrended flux values  [optional]
                     'it_med'         – ndarray of iterative-median      [optional]

        Returning a dict enables the two-panel false-positive diagnostic
        plot.  Returning a plain list still works and produces a
        single-panel plot showing only the raw injected light curve.
    n_lcs : int
        Number of synthetic light curves to generate and test.
    duration_days : float
        Length of each light curve in days.
    cadence_min : float
        Nominal cadence in minutes.
    seed : int
        Master random seed.
    plot_fp : bool
        If True (default), save a diagnostic PNG for every false-positive
        detection into ``fp_plot_dir``.
    fp_plot_dir : str
        Directory for false-positive diagnostic plots.  Created if absent.
        Defaults to ``'diag_plots'``.

    Returns
    -------
    results : pd.DataFrame
        One row per injected flare across all light curves, with columns:

        lc_id, noise_ppm, n_modes, tpeak, fwhm, ampl, snr,
        t_start, t_end, recovered, rec_t_start, rec_t_end, false_positives
    fp_results : pd.DataFrame
        One row per false-positive detection, with columns:
        lc_id, noise_ppm, n_modes, rec_t_start, rec_t_end
    """
    all_rows = []
    fp_rows  = []

    for lc_id in range(n_lcs):
        lc_seed = seed + lc_id
        rng = np.random.default_rng(lc_seed)

        # --- generate light curve
        time, flux, meta = generate_synthetic_lc(
            duration_days=duration_days,
            cadence_min=cadence_min,
            seed=lc_seed,
        )

        # --- inject flares
        flux_injected, flare_table = inject_flares(time, flux, rng=rng, baseline=meta['baseline'])

        plt.figure(figsize=(10, 5))
        plt.plot(time, flux_injected, 'k.', markersize=1)
        plt.title(
            f"Synthetic LC | noise={meta['noise_ppm']:.0f} ppm | "
            f"modes={meta['n_modes']} | injected flares={len(flare_table)}"
        )
        plt.savefig(
            f"diag_plots/synthetic_lc_{meta['noise_ppm']:.0f}_ppm"
            f"_modes_{meta['n_modes']}_flares_{len(flare_table)}.png",
            dpi=150,
        )
        plt.close()

        if flare_table.empty:
            continue

        # --- run flare finder
        try:
            raw_result = flare_finder(time, flux_injected, meta, flare_table)
        except Exception as e:
            print(f"LC {lc_id}: flare_finder raised {e!r} — skipping.")
            continue

        # Support two return conventions:
        #   plain list → detections only, no detrended arrays available
        #   dict       → detections + optional diagnostic arrays for FP plots
        if isinstance(raw_result, dict):
            detections     = raw_result['detections']
            detrended_time = raw_result.get('detrended_time', None)
            detrended_flux = raw_result.get('detrended_flux', None)
            it_med         = raw_result.get('it_med', None)
        else:
            detections     = raw_result
            detrended_time = None
            detrended_flux = None
            it_med         = None

        # --- match injected → recovered
        matched, fp = match_flares(flare_table, detections)
        n_fp = len(fp)

        matched['lc_id']           = lc_id
        matched['noise_ppm']       = meta['noise_ppm']
        matched['n_modes']         = meta['n_modes']
        matched['min_period_hr'] = meta['min_period_hr']
        matched['false_positives'] = n_fp
        all_rows.append(matched)

        if not fp.empty:
            fp['lc_id']     = lc_id
            fp['noise_ppm'] = meta['noise_ppm']
            fp['n_modes']   = meta['n_modes']
            fp['min_period_hr'] = meta['min_period_hr']
            fp_rows.append(fp)

        # --- false-positive diagnostic plots
        if plot_fp and n_fp > 0:
            _plot_false_positives(
                lc_id=lc_id,
                time=time,
                flux_injected=flux_injected,
                detrended_time=detrended_time,
                detrended_flux=detrended_flux,
                it_med=it_med,
                fp_df=fp,
                injected_table=flare_table,
                meta=meta,
                plot_dir=fp_plot_dir,
            )

        print(
            f"LC {lc_id:3d} | noise={meta['noise_ppm']:6.0f} ppm | "
            f"modes={meta['n_modes']} | "
            + f"P_min={meta['min_period_hr']:.1f} h | "
            + f"injected={len(flare_table)} | "
            f"recovered={matched['recovered'].sum()} | "
            f"FP={n_fp}"
        )

        time_module.sleep(5)  # slight delay to avoid overwhelming the output

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    results = pd.concat(all_rows, ignore_index=True)
    results['snr'] = results['ampl'] / (results['noise_ppm'] * 1e-6)

    col_order = [
        'lc_id', 'noise_ppm', 'n_modes', 'min_period_hr',
        'tpeak', 'fwhm', 'ampl', 'snr', 't_start', 't_end',
        'recovered', 'rec_t_start', 'rec_t_end',
        'false_positives',
    ]
    results = results[[c for c in col_order if c in results.columns]]

    fp_col_order = ['lc_id', 'noise_ppm', 'n_modes', 'min_period_hr', 'rec_t_start', 'rec_t_end']
    if fp_rows:
        fp_results = pd.concat(fp_rows, ignore_index=True)
        fp_results = fp_results[[c for c in fp_col_order if c in fp_results.columns]]
    else:
        fp_results = pd.DataFrame(columns=fp_col_order)

    return results, fp_results


# ---------------------------------------------------------------------------
# Quick summary stats
# ---------------------------------------------------------------------------

def summarise(results, fp_results=None):
    """Print a brief recovery summary from the results table."""
    if results.empty:
        print("No results.")
        return

    n_inj  = len(results)
    n_rec  = results['recovered'].sum()
    n_miss = n_inj - n_rec
    n_fp   = results.groupby('lc_id')['false_positives'].first().sum()

    print(f"Injected : {n_inj}")
    print(f"Recovered: {n_rec}  ({100 * n_rec / n_inj:.1f} %)")
    print(f"Missed   : {n_miss}  ({100 * n_miss / n_inj:.1f} %)")
    print(f"False pos: {n_fp}")

    results = results.copy()
    results['snr'] = results['ampl'] / (results['noise_ppm'] * 1e-6)
    bins   = [0, 2, 5, 10, 25, np.inf]
    labels = ['<2', '2–5', '5–10', '10–25', '>25']
    results['snr_bin'] = pd.cut(results['snr'], bins=bins, labels=labels)
    tbl = (results.groupby('snr_bin', observed=True)['recovered']
                  .agg(['sum', 'count'])
                  .rename(columns={'sum': 'recovered', 'count': 'injected'}))
    tbl['rate_%'] = (100 * tbl['recovered'] / tbl['injected']).round(1)
    print("\nRecovery by S/N:")
    print(tbl.to_string())

    if 'min_period_hr' in results:
        # fastest mode within reach of the 6-36 h sinusoid correction
        results['fast_rotator'] = results['min_period_hr'] < 36.0
        by_fast = (results.groupby(['fast_rotator', 'snr_bin'], observed=True)['recovered']
                          .agg(['sum', 'count'])
                          .rename(columns={'sum': 'recovered', 'count': 'injected'}))
        by_fast['rate_%'] = (100 * by_fast['recovered'] / by_fast['injected']).round(1)
        n_fast = results.loc[results['fast_rotator'], 'lc_id'].nunique()
        print(f"\nRecovery by S/N, fast rotators (P_min < 36 h, {n_fast} LCs) vs. others:")
        print(by_fast.to_string())

    if fp_results is not None and not fp_results.empty:
        fp_per_lc = fp_results.groupby('lc_id').size().rename('n_fp')
        print(
            f"\nFalse-positive detections: {len(fp_results)} total across "
            f"{fp_results['lc_id'].nunique()} LCs "
            f"(mean {fp_per_lc.mean():.1f}/LC, max {fp_per_lc.max()})"
        )
        n_unique = fp_results['noise_ppm'].nunique()
        if n_unique >= 2:
            print("\nFalse positives by noise level (ppm quartiles):")
            fp_results = fp_results.copy()
            fp_results['noise_bin'] = pd.qcut(
                fp_results['noise_ppm'], q=min(4, n_unique),
                precision=0, duplicates='drop',
            )
            print(fp_results.groupby('noise_bin', observed=True)
                            .size().rename('n_fp').to_string())
        else:
            print(f"\n  (only {n_unique} distinct noise level in FP table — "
                  "skipping noise breakdown)")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def flare_finder_altai(time, flux, meta, flare_table):
        """Run altaipony detrending + flare detection.

        Returns a dict so that ``run_injection_recovery`` can pass the
        detrended arrays through to the false-positive diagnostic plots.
        Keys
        ----
        detections     – list of (tstart, tstop) tuples
        detrended_time – time array of the detrended LC
        detrended_flux – detrended flux values
        it_med         – iterative-median baseline aligned to detrended_time
        """
        lc = FlareLightCurve(
            time * u.d,
            flux * u.electron / u.s,
            flux_err=np.full_like(flux, 0.001 * meta['baseline']) * u.electron / u.s,
        )
        lcd = custom_detrending(
            lc,
            periodicity_amplitude_threshold=0.001,
            periodicity_fap_threshold=0.001,
            n_per=3,
            maxgap=50,
            baseline_method = 'detrender',
        )
        lcd.detrended_flux = lcd.detrended_flux.value
        lcd = estimate_detrended_noise(lcd)
        print(
            f"Estimated detrended noise: "
            f"{np.median(lcd.detrended_flux_err)*1e6:.0f} ppm "
            f"vs. true {meta['noise_ppm']:.0f} ppm"
        )
        lcd = lcd.find_cont_windows()
        lcd = lcd.find_iterative_median()
        flares = lcd.find_flares(merge_sigma=1.0).flares
        flares = flares[["tstart", "tstop"]].values.tolist()

        # Overview plot (raw + detrended + annotations)
        plt.figure(figsize=(10, 5))
        plt.plot(time, flux, 'k.', markersize=1)
        plt.plot(lcd.time.value, lcd.gp_model, c="c", linestyle="--")
        plt.plot(lcd.time.value, lcd.detrended_flux, 'r.', markersize=1)
        plt.plot(lcd.time.value, lcd.it_med, 'b-', linewidth=1)
        for t_start, t_stop in flare_table[["t_start", "t_end"]].values:
            plt.axvspan(t_start, t_stop, color='orange', alpha=0.5)
        for tstart, tstop in flares:
            plt.axvspan(tstart, tstop, color='cyan', alpha=0.5)
        plt.title(
            f"Flare finder example | noise={meta['noise_ppm']:.0f} ppm | "
            f"modes={meta['n_modes']} | "
            + f"P_min={meta['min_period_hr']:.1f} h | "
            + f"injected={len(flare_table)} | detected={len(flares)}"
        )
        plt.savefig(
            f"diag_plots/flare_finder_example"
            f"_noise_{meta['noise_ppm']:.0f}_ppm"
            f"_modes_{meta['n_modes']}"
            + f"_Pmin_{meta['min_period_hr']:.1f}h"
            + f"_injected_{len(flare_table)}"
            f"_detected_{len(flares)}.png",
            dpi=150,
        )
        plt.close()

        return {
            'detections'    : flares,
            'detrended_time': lcd.time.value,
            'detrended_flux': lcd.detrended_flux,
            'it_med'        : lcd.it_med,
        }

    results, fp_table = run_injection_recovery(
        flare_finder=flare_finder_altai,
        n_lcs=100,
        seed=92,
        plot_fp=True,
        fp_plot_dir='diag_plots',
    )

    # save results to CSV for later analysis
    results.to_csv("diag_plots/injection_recovery_results.csv", index=False)

    print("\n--- Summary ---")
    summarise(results, fp_table)
    print("\n--- Results table (first 150 rows) ---")
    print(results.head(150).to_string(index=False))
    print("\n--- False-positive table (first 120 rows) ---")
    print(fp_table.head(120).to_string(index=False))

    # Recovery scatter: FWHM vs S/N, coloured by recovered/missed
    plt.figure(figsize=(6, 5))
    for group, df in results.groupby('recovered'):
        plt.scatter(df['fwhm'], df['snr'], label=f'recovered={group}', alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('FWHM (days)')
    plt.ylabel('S/N')
    plt.legend()
    plt.savefig("diag_plots/recovery_scatter.png", dpi=150)
    plt.close()