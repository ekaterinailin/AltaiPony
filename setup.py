import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="altaipony",
    version="2.1.1",
    author="Ekaterina Ilin",
    author_email="eilin@aip.de",
    description="Flare science in Kepler and TESS light curves",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/ekaterinailin/AltaiPony",
    packages=setuptools.find_packages(),
    install_requires = ['numpy>=1.15.1', 'pybind11','lightkurve>=2.4','pandas>=2.0', 'george>=0.3',
                        'progressbar2>=3.51.4','seaborn',  'astropy>=5', 'k2sc>=1.0.2', 'emcee','corner', 'scipy>1.5'],

    include_package_data=True,
    package_data={
      'altaipony': ['static/*csv', 'examples/kplr010002792-2010174085026_llc.fits']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
