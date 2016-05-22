#! python
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='sspals',
      version='0.0.2',
      description='process single-shot positron annihlation lifetime spectra',
      long_description='This package contains tools for combining high- and \
                       low-resolution SSPALS data, and for calculating the so-called \
                       delayed fraction, which can be used to estimate the amount \
                       of positrons that are converted into long-lived states of \
                       positronium.',
      url='https://github.com/PositroniumSpectroscopy/sspals',
      author='Adam Deller',
      author_email='a.deller@ucl.ac.uk',
      license='BSD',
      packages=['sspals'],
      install_requires=[
          'scipy>0.14', 'numpy>1.10', 'pandas>0.17 ',
      ],
      include_package_data=True,
      zip_safe=False)
