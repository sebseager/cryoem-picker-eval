# SPHIRE-crYOLO

## Summary

crYOLO is available for download as a Python package from PyPi.

The crYOLO paper can be found [here](https://doi.org/10.1038/s42003-019-0437-z). 
Refer also to the [wiki](https://sphire.mpg.de/wiki/doku.php?id=pipeline:window:cryolo) (potentially outdated) and 
[readthedocs user guide](https://cryolo.readthedocs.io/en/latest/) for more information.

## crYOLO Installation

*Note: modified from https://cryolo.readthedocs.io/en/latest/installation.html*

Create and activate a new conda environment with the required dependencies. It is possible to use 
`conda activate cryolo` instead of `source activate cryolo` for conda versions ≥ 4.4.

```shell script
conda create -n cryolo -c conda-forge -c anaconda python=3.6 pyqt=5 cudnn=7.1.2 numpy==1.14.5 cython wxPython==4.0.4 intel-openmp==2019.4
source activate cryolo
```

Verify that `which pip` points to an executable inside the `bin/` directory of your conda installation, and that 
`which python` points to an executable inside your `conda_envs/bin` directory.

To install crYOLO with GPU support (recommended, if you have a GPU available), run 

```shell script
pip install 'cryolo[gpu]'
```

or, to install crYOLO with CPU support only, run 

```shell script
pip install 'cryolo[cpu]'
```

crYOLO provides three general models, each of which has been trained on a variety of data sets. They are available in 
[this section](https://cryolo.readthedocs.io/en/latest/installation.html#download-the-general-models) of the user 
guide. The following commands can also be used to download the most current versions (as of October 1, 2020) to your 
current working directory.

Low-pass filtered:

```shell script
wget ftp://ftp.gwdg.de/pub/misc/sphire/crYOLO-GENERAL-MODELS/gmodel_phosnet_202005_N63_c17.h5
```

Neural-network (JANNI) denoised:

```shell script
wget ftp://ftp.gwdg.de/pub/misc/sphire/crYOLO-GENERAL-MODELS/gmodel_phosnet_202005_nn_N63_c17.h5
```

Negative stain:

```shell script
wget ftp://ftp.gwdg.de/pub/misc/sphire/crYOLO-GENERAL-MODELS/gmodel_phosnet_negstain_20190226.h5
```
