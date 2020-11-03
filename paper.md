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

Flares are unmistakable signs of stellar magnetic activity and a key to our understanding
of stellar properties and evolution. Driven by magnetic instability, they are violent ex-
plosions that penetrate all layers of a star’s atmosphere, and enhance the overall stellar
brightness by up to orders of magnitude within minutes. We observe them as distinct
signatures in the light curves − repeated measurements of stellar brightness over un-
interrupted periods of time − of most stars. Their rates and energies provide
unique clues to the nature of the stars that produce them. Space missions like Kepler and TESS have already
collected light curve of hundreds of thousands of stars, some of which have been 
monitored for multiple years, and continue to gather high-cadence high-precision
data. To address the sheer size of this growing treasury and to aid the growing number of 
researchers who study the flares observed by Kepler and TESS, we developed `AltaiPony`. 
`AltaiPony` is a toolbox for statistical flares studies on photometric time series 
from these missions, including flare search and characterization, injection-recovery diagnostics, 
and statistical analysis of flare frequency distributions along with extensive 
documentation and Jupyter-based tutorials.


`AltaiPony` can access most features that are implemented in `lightkurve` [@lightkurve2018], 
which makes it an accessible tool for new users who are already familiar with 
the software. `lightkurve` offers versatile light curve handling including 
visualization, basic tools for light curve de-trending, transit detection, and 
asteroseismology, and is the most widely used software for handling Kepler, 
K2, and TESS data.


`AltaiPony` was designed to be used by astronomical researchers as a one stop shop 
solution with adaptations of common de-trending tools like the Savitzky-Golay filter
from `lightkurve.flatten()`, and K2SC [@aigrain2016; @k2sc2016] that are suited 
to preserve flare signal and remove astrophysical and instrumental variability. 
The design also allows users to add custom de-trending functions to `AltaiPony`. 
An example can be found in `@ilin2020`.

As measured flare amplitudes and durations are different from their intrinsic properties 
due to various astrophysical and instrumental effects, `AltaiPony` features an 
injection-recovery pipeline for synthetic flares with visualization aids to quantify 
the cumulated effects  introduced by light curve specific noise levels, time sampling,
 and the de-trending and flare finding procedure of choice. 

The frequencies $f$ of flares have been shown to follow a power law distribution in energy $E$:

\begin{equation}
f(>E) = \dfrac{\beta}{\alpha - 1}E^{-\alpha + 1},
\end{equation}

To estimate $\alpha$ and $\beta$ and their respective uncertainties for a given sample
 of flares, `AltaiPony` includes a modified maximum likelihood estimator method 
[@maschberger2009] for $\alpha$ with a subsequent least-squares fit to $\beta$ with 
bootstrapped uncertainties, as well as a class based fully Bayesian framework that 
incorporates both the power law nature of $f$, and the exponential flare waiting times 
to analyse flare frequency distributions adapted from a solar model [@wheatland2004] 
that uses emcee [@emcee2013] to sample from the posterior distribution using 
the Markov Chain Monte Carlo method.

`AltaiPony` has already been used in peer-reviewed publications [@ilin2019; @ilin2020; @ramsay2020], 
and remains under active development.

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

# Acknowledgements

I acknowledge valuable contributions from Michael Gully-Santiago and Geert Barentsen,
who offered advice and hands-on support in the early development
stages of the project.

# References
