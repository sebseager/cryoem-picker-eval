---
layout: default
parent: Particle Pickers
---

# ASPIRE APPLEpicker

## Summary

ASPIRE (which includes APPLEpicker) is available as a set of conda-installable Python modules. We used Python 3.7 for this installation. 

There are several versions of APPLEpicker—a [standalone Python version](https://github.com/PrincetonUniversity/APPLEpicker-python), a [standalone MATLAB version](https://github.com/PrincetonUniversity/APPLEpicker), and an implementation within [the ASPIRE project for Python](https://github.com/ComputationalCryoEM/ASPIRE-Python).

The standalone version(s) seem to have been deprecated, according to one of the collaborators [here](https://github.com/PrincetonUniversity/APPLEpicker/issues/1#issuecomment-525574243). Therefore, the following document refers only to APPLEpicker as included in the ASPIRE-Python project.

The APPLEpicker paper can be found [here](https://doi.org/10.1016/j.jsb.2018.08.012). Refer also to the ASPIRE [wiki](https://computationalcryoem.github.io/ASPIRE-Python/) for more information. The following guide is drawn in part from these resources.

## Installation

ASPIRE must be installed into its own conda environment. It can be obtained either as a pip-installable package or via its git repository. Because this guide includes several patches, it will follow the latter method.

First, clone the ASPIRE repo into `pickers/aspire/` using 

```shell script
git clone https://github.com/ComputationalCryoEM/ASPIRE-Python.git pickers/aspire
```

During the course of our experiments, we have created a set of patches for APPLEpicker. If you would like to apply these changes, run the patch script included in `cryo-docs/patches/aspire`. This script will replace `apple.py`, `helper.py`, and `picking.py` in the `pickers/aspire/src/aspire/apple/` directory with our patched versions.

```shell script
sh patches/aspire/patch-aspire.sh pickers/aspire/
```

Create a new conda environment containing the packages in the included `environment.yml` file.

```shell script
conda env create -f pickers/aspire/environment.yml
conda activate aspire
```

The new environment will be named `aspire` by default (as specified in the environment file), but `-n your_name_here` can be added to the `conda env create` command to change this name.

The ASPIRE README recommends running their provided unit tests before installing. (Note that due to modifications made to the APPLEpicker source by our patch script above, the `testPickCenters` unit test may fail. The rest of the tests should pass with a handful of `FutureWarning`s, `DeprecationWarning`s, etc.)

```shell script
(cd pickers/aspire && PYTHONPATH=./src pytest tests)
```

In case the above does not work, a second unit test method is also provided, though it appears that it may become deprecated in future.

```shell script
(cd pickers/aspire && python setup.py test)
```

Install ASPIRE to the active conda environment (make sure you run `conda activate aspire` first, if you have not already done so).

```shell script
(cd pickers/aspire && python setup.py install)
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

There are several configurable parameters hard-coded in `pickers/aspire/src/aspire/config.ini`, including the following. [TODO: FIGURE OUT DEFINITIONS FOR EACH IF POSSIBLE]
- `particle_size` (default: 78)
- `query_image_size` (default: 52)
- `max_particle_size` (default: 156)
- `min_particle_size` (default: 19)
- `tau1` (default: 710): percentage of training images believed to contain a particle
- `tau2` (default: 7100): percentage of training images believed may *potentially* contain a particle

### Picking

Start by collecting the micrograph files (`*.mrc`) to be picked in a directory (assuming they are not already available in their own directory). If you would like to use an existing public data set, [our guide to the EMPIAR database](empiar.md) may be helpful.

```shell script
mkdir -p name_of_data_set/mrc
mv path/to/your_mrc_files/*.mrc name_of_data_set/mrc
```

Here we will use the micrographs located in `demo_data/` as an example. Create another directory, in which any output, temporary, or configuration files will be saved by the picker.

```shell script
mkdir demo_data/apple_out
```

To pick particles for every micrograph in `demo_data/mrc/`, run

```shell script
python -m aspire apple --mrc_dir demo_data/mrc/ --output_dir demo_data/apple_out/
```

or, to pick a single micrograph only, run

```shell script
python -m aspire apple --mrc_file demo_data/mrc/your_micrograph.mrc --output_dir demo_data/apple_out/
```

The `--create_jpg` flag can be appended to either of the `python -m aspire apple` commands above to generate and save images of the micrograph(s) with particle detections overlayed.
