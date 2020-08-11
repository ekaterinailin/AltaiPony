import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="altaipony",
    version="0.0.1",
    author="Ekaterina Ilin",
    author_email="eilin@aip.de",
    description="A flare finding and analysis package for K2",
    long_description=long_description,
    long_description_content_type="text/restructuredtext",
    url="https://github.com/ekaterinailin/AltaiPony",
    packages=setuptools.find_packages(),
    install_requires = ['numpy>=1.15.1', 'pybind11','lightkurve==1.9.1','pandas>=0.23.4',
                        'progressbar2','seaborn',  'k2sc==1.0.1.4'],
    include_package_data=True,
    package_data={
      'altaipony': ['static/*csv']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
