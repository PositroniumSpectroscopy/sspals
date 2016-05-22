# sspals
python tools for analysing single-shot positron annihilation lifetime spectra

## Prerequisites

Tested using Anaconda (Continuum Analytics) with Python 2.7 and 3.5.

Package dependencies:

* scipy, numpy, pandas

## Installation

via pip (recommended):

```
pip install sspals
```

alternatively, try the development version

```
git clone https://github.com/PositroniumSpectroscopy/sspals
```

and then run

```
python setup.py install
```

## About

Single-shot positron annihilation lifetime spectroscopy (SSPALS) [Ref. 1] 
essentially consists of studying the number of annihilation gamma-rays 
measured as a function of time following implantation of a time-focused
(~5 ns) positron bunch into a solid target material.

For certain materials a significant fraction of the positrons (~ 30%) will
bind to electrons to form positronium (Ps), which can then be re-emitted 
to vacuum. Ps has a characteristic mean lifetime of 142 ns in vacuum, which 
makes it relatively easy to identify in SSPALS spectra.

This package includes a handful of useful tools for working with SSPALS data. 
The two main functions are used to: (i) combine data split across hi/ low 
gain channels of a digital oscilloscope, and (ii) to estimate the amount of
Ps formed using the so-called delayed fraction.

*sspals.chmx(hi, low)*
>	Remove zero offset from hi and low gain data, invert and splice 
    together by swapping saturated values from the hi-gain channel 
    for those from the low-gain channel.  Apply along rows of a 2D array.

*sspals.sspals(arr, dt, limits=[A, B, C])*
>	Calculate the trigger time t0 (using a cfd) and the delayed fraction (DF)
	(integral B->C / integral A->C) for each row of a 2D array. Return a pandas 
	DataFrame [(t0, AC, BC, DF)].

Raw data (hi, low) is expected to be 2D arrays of repeat measurements, where each
row contains a single SSPALS waveform.

For examples see the IPython/ Jupter notebooks,

https://github.com/PositroniumSpectroscopy/sspals/tree/master/examples

**Refs**.

1.	D. B. Cassidy et al. (2006), Appl. Phys. Lett., 88, 194105. http://dx.doi.org/10.1063/1.2203336
