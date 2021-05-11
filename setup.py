import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="altaipony",
    version="1.0.1",
    author="Ekaterina Ilin",
    author_email="eilin@aip.de",
    description="Flare science in Kepler, K2 and TESS light curves",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/ekaterinailin/AltaiPony",
    packages=setuptools.find_packages(),
    install_requires = ['numpy>=1.15.1', 'pybind11','lightkurve>=2','pandas==1.1.4, !=1.1.5', 'george>=0.3',
                        'progressbar2>=3.51.4','seaborn',  'astropy>=4.1', 'k2sc==1.0.1.4', 'emcee','corner', 'scipy>1.5'],

    include_package_data=True,
    package_data={
      'altaipony': ['static/*csv', 'examples/kplr010002792-2010174085026_llc.fits']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
