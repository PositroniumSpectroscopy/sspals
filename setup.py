#! python
''' This package contains tools for combining high- and
    low-resolution SSPALS data, and for calculating the so-called
    delayed fraction, which can be used to estimate the amount
    of positrons that are converted into long-lived states of
    positronium.
'''
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Natural Language :: English",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Topic :: Scientific/Engineering :: Physics", 
]

setup(name='sspals',
      version='0.0.4',
      description='process single-shot positron annihlation lifetime spectra',
      url='https://github.com/PositroniumSpectroscopy/sspals',
      author='Adam Deller',
      author_email='a.deller@ucl.ac.uk',
      license='BSD',
      packages=['sspals'],
      install_requires=[
          'scipy>0.12', 'numpy>1.9', 'pandas>=0.16',
      ],
      include_package_data=True,
      classifiers=CLASSIFIERS,
      zip_safe=False)
