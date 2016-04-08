from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='sspals',
      version='0.0.1',
      description='for processing single-shot positron annihlation lifetime spectrscopy',
      url='https://github.com/PositroniumSpectroscopy/sspals',
      author='Adam Deller',
      author_email='a.deller@ucl.ac.uk',
      license='BSD',
      packages=['sspals'],
      install_requires=[
          'scipy>0.14',
      ],
      include_package_data=True,
      zip_safe=False)
