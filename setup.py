import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="altaipony",
    version="1.0.0",
    author="Ekaterina Ilin",
    author_email="eilin@aip.de",
    description="Flare science in Kepler, K2 and TESS light curves",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ekaterinailin/AltaiPony",
    packages=setuptools.find_packages(),
    install_requires = ['numpy>=1.15.1', 'pybind11','lightkurve>=1.9.1','pandas==1.1.4, !=1.1.5',
                        'progressbar2>=3.51.4','seaborn',  'astropy>=4.1', 'k2sc==1.0.1.4', 'emcee','corner', 'scipy>1.5'],

    include_package_data=True,
    package_data={
      'altaipony': ['static/*csv']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
