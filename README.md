# positronium
python tools pertaining to positronium

## Prerequisites

Tested using Anaconda (Continuum Analytics) with Python 2.7 and 3.5.

Package dependencies:

* scipy

## Installation

via pip (recommended):

```
pip install positronium
```

## About

This package is designed to collate useful bits of code relating to the positronium atom
(i.e., an electron bound to its antiparticle, the positron).

The package currently only contains two very simple modules.

*constants* is intended to collect useful constants.

*bohr* uses an adaption of the Rydberg formula (sim. to hydrogen) to calculate the principle
energy levels of positronium, or the interval between two levels.  The default unit is 'eV',
however, the kwarg 'unit' can be configured to return, for instance, the corresponding vacuum
wavelength.

```python
>>> from positronium import bohr
>>> bohr.En(1, 2, unit='nm')
243.00454681357735
```

For further examples see the IPython/ Jupter notebooks,

https://github.com/PositroniumSpectroscopy/positronium/tree/master/examples
