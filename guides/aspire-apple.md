# ASPIRE APPLEpicker

## Summary

There are several versions of APPLEpicker—a 
[standalone Python version](https://github.com/PrincetonUniversity/APPLEpicker-python), a 
[standalone MATLAB version](https://github.com/PrincetonUniversity/APPLEpicker), and an implementation within 
[the ASPIRE project for Python](https://github.com/ComputationalCryoEM/ASPIRE-Python).

The standalone version(s) seem to have been deprecated, according to one of the collaborators 
[here](https://github.com/PrincetonUniversity/APPLEpicker/issues/1#issuecomment-525574243). Therefore, the following 
document refers only to APPLEpicker as included in the ASPIRE-Python project.

The APPLEpicker paper can be found [here](https://doi.org/10.1016/j.jsb.2018.08.012). Refer also to the ASPIRE 
[wiki](https://computationalcryoem.github.io/ASPIRE-Python/) for more information.

## ASPIRE Installation

*Note: modified from https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/master/README.md*

ASPIRE must be installed into its own conda environment. It can be obtained either as a pip-installable package or via 
its git repository. Because this guide includes several patches, it will follow the latter method.

First, clone the ASPIRE repo using 

```shell script
git clone https://github.com/ComputationalCryoEM/ASPIRE-Python.git
```

During the course of our experiments, we have created a set of patches for APPLEpicker. To apply these changes, run the 
patch script included in `cryo-docs/patches/aspire`. This script will replace several of APPLEpicker's files with our 
patched versions.

```shell script
sh ./cryo-docs/patches/aspire/patch-aspire.sh ASPIRE-Python/
```

Create a new conda environment containing the packages in the included `environment.yml` file.

```shell script
cd /path/to/ASPIRE/clone
conda env create -f environment.yml
conda activate aspire
```

The ASPIRE README recommends running their provided unit tests using

```shell script
cd /path/to/ASPIRE/clone
python setup.py test
```

A second method is also provided, which can be used if the above becomes deprecated in future.

```shell script
cd /path/to/ASPIRE/clone
PYTHONPATH=./src pytest tests
```

If unit tests pass, install ASPIRE to the active conda environment (make sure you run `conda activate aspire` if you 
have not already done so).

```shell script
cd /path/to/ASPIRE/clone
python setup.py install
```
