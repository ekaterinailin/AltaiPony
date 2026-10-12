"""
UTF-8, Python 3

Flare-finding-only injection-recovery: the same benchmark as
``injection_recovery.py``, but with perfect detrending.

The light curves, noise and injected flares are exactly those of
``injection_recovery.py`` (same generator, seeds and flare draws).  Instead
of ``custom_detrending``, each light curve is detrended by subtracting the
known synthetic variability, so the detrended flux is 1 + noise + flares.
Flare detection is the same (``injection_recovery.detect_flares``).

The result is the best recovery the flare-finding steps can achieve; the
difference to ``injection_recovery.py`` is what the detrending loses.  If
``diag_plots/injection_recovery_results.csv`` exists, the script prints that
difference by S/N, FWHM and type of star.

Run from the ``notebooks`` directory.  Output goes to
``diag_plots_flare_finding_only/``.
"""

import os

import numpy as np
import pandas as pd
import astropy.units as u

from altaipony.flarelc import FlareLightCurve
from altaipony.synthetic import generate_synthetic_lc, sinusoidal_variability

import injection_recovery as ir

OUT = "diag_plots_flare_finding_only"
PIPELINE_RESULTS = "diag_plots/injection_recovery_results.csv"


def flare_finder_perfect_detrending(time, flux, meta, flare_table):
    """Detrend by subtracting the known variability, then detect flares as in
    ``flare_finder_altai``."""
    detrended = flux / meta["baseline"] - sinusoidal_variability(time, meta["modes"])
    lc = FlareLightCurve(time * u.d, flux * u.electron / u.s,
                         flux_err=np.full_like(flux, 0.001 * meta["baseline"]) * u.electron / u.s)
    lc.detrended_flux = detrended
    lc, flares = ir.detect_flares(lc)
    return {
        "detections": flares,
        "detrended_time": lc.time.value,
        "detrended_flux": lc.detrended_flux,
        "it_med": lc.it_med,
    }


def compare(perfect, pipeline):
    """Recovery with perfect detrending vs. the pipeline, by S/N, FWHM and
    type of star (spotless, fast rotator with P_min < 36 h, other)."""
    assert np.allclose(perfect.tpeak, pipeline.tpeak), "different injected flares"
    spotless = {i: generate_synthetic_lc(seed=SEED + i)[2]["spotless"]
                for i in perfect.lc_id.unique()}
    star = np.where(perfect.lc_id.map(spotless), "spotless",
                    np.where(perfect.min_period_hr < 36, "fast rotator", "other spotted"))
    groups = {
        "S/N": pd.cut(perfect.snr, [0, 5, 10, 25, np.inf], labels=["2–5", "5–10", "10–25", ">25"]),
        "FWHM [min]": pd.cut(perfect.fwhm * 1440, [0, 15, 30, 60, 120, 200],
                             labels=["≤15", "15–30", "30–60", "60–120", "120–200"]),
        "star": pd.Series(star),
    }
    for name, g in groups.items():
        table = pd.DataFrame({
            "injected": perfect.groupby(g, observed=True).size(),
            "perfect detrending": perfect.groupby(g, observed=True).recovered.sum(),
            "pipeline": pipeline.groupby(g, observed=True).recovered.sum(),
        })
        table["gap"] = table["perfect detrending"] - table["pipeline"]
        print(f"\nRecovered by {name}:")
        print(table.to_string())
    print(f"\nTotal: perfect detrending {perfect.recovered.sum()}, pipeline "
          f"{pipeline.recovered.sum()} of {len(perfect)} injected flares")


N_LCS, SEED = 100, 92      # as in injection_recovery.py

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    results, fp_table = ir.run_injection_recovery(
        flare_finder=flare_finder_perfect_detrending,
        n_lcs=N_LCS,
        seed=SEED,
        plot_fp=True,
        fp_plot_dir=OUT,
    )
    results.to_csv(f"{OUT}/flare_finding_only_results.csv", index=False)
    fp_table.to_csv(f"{OUT}/flare_finding_only_false_positives.csv", index=False)

    print("\n--- Summary (perfect detrending) ---")
    ir.summarise(results, fp_table)

    if os.path.exists(PIPELINE_RESULTS):
        print("\n--- Perfect detrending vs. pipeline (diag_plots/) ---")
        compare(results, pd.read_csv(PIPELINE_RESULTS))
