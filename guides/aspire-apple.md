# ASPIRE APPLEpicker

## Summary

There are several versions of APPLEpicker—a [standalone Python version](https://github.com/PrincetonUniversity/APPLEpicker-python), a [standalone MATLAB version](https://github.com/PrincetonUniversity/APPLEpicker), and an implementation within [the ASPIRE project for Python](https://github.com/ComputationalCryoEM/ASPIRE-Python).

The standalone version(s) seem to have been deprecated, according to one of the collaborators [here](https://github.com/PrincetonUniversity/APPLEpicker/issues/1#issuecomment-525574243). Therefore, the following document refers only to APPLEpicker as included in the ASPIRE-Python project.

The APPLEpicker paper can be found [here](https://doi.org/10.1016/j.jsb.2018.08.012). Refer also to the ASPIRE [wiki](https://computationalcryoem.github.io/ASPIRE-Python/) for more information. The following guide is drawn in part from these resources.

## Installation

ASPIRE must be installed into its own conda environment. It can be obtained either as a pip-installable package or via its git repository. Because this guide includes several patches, it will follow the latter method.

First, clone the ASPIRE repo using 

```shell script
git clone https://github.com/ComputationalCryoEM/ASPIRE-Python.git
```

During the course of our experiments, we have created a set of patches for APPLEpicker. If you would like to apply these changes, run the patch script included in `cryo-docs/patches/aspire`. This script will replace `apple.py`, `helper.py`, and `picking.py` in the `ASPIRE-Python/src/aspire/apple/` directory with our patched versions.

```shell script
sh cryo-docs/patches/aspire/patch-aspire.sh ASPIRE-Python/
```

Create a new conda environment containing the packages in the included `environment.yml` file.

```shell script
conda env create -f ASPIRE-Python/environment.yml
conda activate aspire
```

The ASPIRE README recommends running their provided unit tests using

```shell script
python ASPIRE-Python/setup.py test
```

A second method is also provided, which can be used if the above becomes deprecated in future.

```shell script
PYTHONPATH=./ASPIRE-Python/src pytest tests
```

If unit tests pass, install ASPIRE to the active conda environment (make sure you run `conda activate aspire` first, if you have not already done so).

```shell script
python ASPIRE-Python/setup.py install
```

## Usage

### Overview

Inputs
- micrographs for which to pick particles
- particle size, tau1, tau2, and other configurable parameters

Outputs
- directory containing `*.star` coordinate files

### Setup

### Configuration

[TODO: SORT OUT ASPIRE CONFIG CMD, and/or whether config.ini is read from conda package (doesn't seem like it)]

There are several configurable parameters hard-coded in `ASPIRE-Python/src/aspire/config.ini`, including the following. [TODO: FIGURE OUT DEFINITIONS FOR EACH IF POSSIBLE]
- `particle_size` (default: 78)
- `query_image_size` (default: 52)
- `max_particle_size` (default: 156)
- `min_particle_size` (default: 19)
- `tau1` (default: 710)
- `tau2` (default: 7100)

### Picking

Start by collecting the micrograph files (`*.mrc`) to be picked in a directory (assuming they are not already available in their own directory). If you would like to use an existing public data set, [our guide to the EMPIAR database](empiar.md) may be helpful.

```shell script
cd /path/to/dataset
mkdir mrc
mv path/to/your_mrc_files/*.mrc mrc/
```

Create another directory, in which crYOLO configurations, temporary files, and predicted coordinates will be saved.

```shell script
mkdir apple_output
```

To pick particles for every micrograph in `mrc/`, run

```shell script
python -m aspire apple --mrc_dir mrc/ --output_dir apple_output/
```

or, to pick a single micrograph only, run

```shell script
python -m aspire apple --mrc_file mrc/your_micrograph.mrc --output_dir apple_output/
```

The `--create_jpg` flag can be appended to either of the `python -m aspire apple` commands above to generate and save images of the micrograph(s) with particle detections overlayed.
