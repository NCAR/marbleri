from setuptools import setup

classifiers = ['Development Status :: 4 - Beta',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.6',
               ]

requires = ["numpy>=1.14",
            "pandas>=0.2",
            "scipy>=1.0",
            "matplotlib>=2.0",
            "xarray",
            "netcdf4",
            "tensorflow>=1.8",
            "keras",
            "scikit-learn>=0.2"]

if __name__ == "__main__":
    setup(name="marbleri",
          version="0.1",
          description="Deep Learning for Hurricane Intensity Prediction",
          author="David John Gagne",
          author_email="dgagne@ucar.edu",
          license="MIT",
          url="https://github.com/NCAR/marbleri",
          packages=["marbleri"],
          scripts=[],
          data_files=[],
          keywords=["hurricane", "deep learning"],
          include_package_data=True,
          zip_safe=False,
          install_requires=requires
          )
