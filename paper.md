---
title: 'AltaiPony - Flare science in Kepler, K2 and TESS light curves'
tags:
  - Python
  - astronomy
  - stellar activity
  - Kepler
  - TESS
  - K2
  - stellar flares
authors:
  - name: Ekaterina Ilin^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0002-6299-7542
    affiliation: "1, 2" # (Multiple affiliations must be quoted)

affiliations:
 - name: Leibniz-Institute for Astrophysics Potsdam (AIP), An der Sternwarte 16, 14482 Potsdam, Germany
   index: 1
 - name: Institute for Physics and Astronomy, University of Potsdam, Karl-Liebknecht-Str. 24/25, 14476 Potsdam, Germany
   index: 2

date: 03 November 2020
bibliography: paper.bib
---

# Summary

Why do we want to study flares in optical light curves?

# Statement of need


There are many light curves, and flare statistics is a growing field, so...
Kepler, K2, and TESS

`lightkurve` [@lightkurve2018] offer versatile light curve handling including visualization, tools for light curve de-trending, transit detection, and asteroseismology, and are the most widely used software for handling Kepler, K2, and TESS data. `AltaiPony` can access most its features, which makes it an accessible tool for new users who are already familiar with `lightkurve`.

`AltaiPony` is a toolbox for statistical flares studies on photometric time series 
from Kepler, K2, and TESS, including flare search and characterization, injection/recovery diagnostics, 
and statistical analysis of flare frequency distributions along with extensive documentation and tutorials.

The software includes functionality inherited from a flare finding solution for Kepler light curves (`Appaloosa`, `@davenport2016`).

`AltaiPony` was designed to be used by astronomical researchers as a one stop shop 
solution that includes adaptations of common de-trending tools like the Savitzky-Golay filter and K2SC [@aigrain2016] that are suited to preserve flare signal and remove astrophysical and instrumental variability. 

The design allows users to add custom de-trending functions to `AltaiPony`.

It has already been used in a number of scientific publications [@ilin2019, @ilin2020, @ramsay2020, more...]

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

# Acknowledgements

I acknowledge contributions from Michael Gully-Santiago and Geert Barentsen,
who offered invaluable advice and hands-on support in th early development
stages of the software.

# References
