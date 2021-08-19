# marbleri: Prediction of Rapid Intensification with Deep Learning

## Requirements
* python 3.6 
* numpy
* scipy
* matplotlib
* tensorflow
* pandas
* xarray
* netcdf4
* keras
* scikit-learn
* jupyter
* ipython

## Installation with conda
First install [miniconda](https://docs.conda.io/en/latest/miniconda.html). If you are running on the
NCAR supercomputer, I recommend installing miniconda in your work directory.

Clone marbleri from github: `git clone https://github.com/NCAR/marbleri.git`

Once miniconda is installed, you can create a dedicated environment for marbleri with the following command:
```bash
cd marbleri
conda env create -f environment.yml
conda activate marbleri
```

If you already have created an environment for marbleri dependencies and need to update it:
```bash
conda activate marbleri
conda env update -f environment.yml
```

To install marbleri in the environment:
```bash
pip install . # Install a copy into the python environment
pip install -e . # Install a soft-linked copy to enable development without reinstalling.
```

Use pytest to validate that the installation was successful:
```bash
pytest .
```

## Data Locations
The data for the project are located in `/glade/p/ral/nsap/rozoff/hfip` on
Cheyenne. The input data consist of HWRF runs for each TC.

## How to run
```bash
python train_hwrf_ml.py config/hwrf_train_2020_realtime.yml -t
```