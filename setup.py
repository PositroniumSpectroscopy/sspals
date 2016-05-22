from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='sspals',
      version='0.1.0',
      description='process single-shot positron annihlation lifetime spectra',
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
