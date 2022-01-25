import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

# This call to setup() does all the work
setuptools.setup(
    name="MetPyQC",
    version="0.0.1",

    description="Quality control and reconstruction of meteorological data",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/silve92/MetPyQC",
    author="Lorenzo Silvestri",
    author_email="lorenzo.silvestri.eng@gmail.com",
    license="MIT license",

    # Metadata
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    # Packaging
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=["numpy>=1.20.1", "scipy>=1.6.0", "pandas>=1.1.3",
                      "scikit-learn>=0.23.2", "tqdm>=4.42.1"],
)
