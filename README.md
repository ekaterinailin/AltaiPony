# AltaiPony
An improved and lean version of Appaloosa w/o extensive I/O, but with de-trending but using K2SC and lightkurve.

## Structure 

```lightcurve.py```
All of the IO stuff goes in here. A lightcurve class with its constructor.

```detrend.py```
K2SC detrending is stitched in here.

```findflares.py```
Split LC into continuous observation chunks.
Apply thresholds to detect candidates.

```fakeflares.py```
Includes:
- semi-empirical flare model
- injection/recovery procedure for synthetic flares.
```analysis.py```
- calculates ED, duration, amplitude, uncertainties, observation times
- possibly other stats about the original flares 
- (correlations with other astrophysical photometric varibility)
- flare energy correction factor
- flare recovery probability

```altai.py```
Main wrapper that
1. takes a K2 (or TESS) ID or a path to a .fits or TPF.gz file
2. creates a light curve using lightkurve
3. de-trends light curve using K2SC
4. find flare candidates
(4b. runs fake flare injection/recovery)
(4c. Calculates flare parameters, corrects ED and returns recovery probability)

