# PARSED

## Summary

PARSED is available as a collection of Python programs.

The PARSED paper can be found [here](https://doi.org/10.1093/bioinformatics/btz728). Both the PARSED [package files](/patches/parsed/original) and a [user manual](/patches/parsed/original/PARSED_Manual_V1.pdf) can be downloaded in the `parsed_v1.zip` archive available with the supplementary materials of the PARSED paper. However, since this software does not appear to have been actively maintained since its publication, we include all relevant files in `patches/parsed/original` of this repo for ease of access. In doing so, we adopt for these files the same usage license identified in the paper, namely:

> the PARSED package and user manual [are available] for noncommercial use

## Installation

First, make a new directory for the PARSED package. 

```shell script
mkdir parsed
```

To install PARSED with our patches and bug fixes, copy the relevant files from `patches/parsed`:

```shell script
cp cryo-docs/patches/{mic_preprocess.py,parsed_main.py,particle_mass.py,pre_train_model.h5} parsed/
```

or, to install PARSED as it was originally published, copy the relevant files from `patches/parsed/original`:

```shell script
cp cryo-docs/patches/original/{mic_preprocess.py,parsed_main.py,particle_mass.py,pre_train_model.h5} parsed/
```

Create and activate a new conda environment with the required dependencies. It may take conda several tries to solve this environment due to the highly specific package dependencies. Note that the versions specified for each package correspond with those listed on the first page of the [PARSED user manual](/patches/parsed/original/PARSED_Manual_V1.pdf). It may be possible to use the latest versions of some packages (e.g., `mrcfile` and `trackpy`) without adverse effects.

```shell script
conda create -n parsed -c conda-forge h5py=2.7.1 keras=2.0.8 numba=0.37.0 pandas=0.20.3 matplotlib=2.1.0 mrcfile=1.0.1 trackpy=0.4.1 python=3.6
conda activate parsed
```

Install any additional dependencies (not directly available through conda) with `pip`. In doing so, however, it is important that we use the `pip` executable located in the conda environment created above. To verify this, check that running `which pip` outputs something like `/path/to/conda_envs/cryolo/bin/pip`. If not, try restarting your shell session or running `conda deactivate parsed` followed by `conda activate parsed` before proceeding.

```shell script
pip install opencv-python
```

If your system does not already have an installation of `tensorflow` and/or `cudatoolkit`, run the following to install both (and their dependencies). Note that `tensorflow-gpu` 1.3.0 is no longer available for `python` versions higher than 3.6 (however, since we specified `python=3.6` when creating the `parsed` environment, this should work).

```shell script
conda install tensorflow-gpu=1.3.0
```

## Usage

### Overview

Inputs
- micrographs for which to pick particles
- picking aperture size, micrograph pixel resolution, and other configurable parameters
- general model `*.h5` file

Outputs
- directory containing `*.star` coordinate files

### Pick using pretrained model

Start by collecting the micrograph files (`*.mrc`) to be picked in a directory (assuming they are not already available in their own directory). If you would like to use an existing public data set, [our guide to the EMPIAR database](empiar.md) may be helpful.

```shell script
cd /path/to/dataset
mkdir mrc
mv path/to/your_mrc_files/*.mrc mrc/
```

Create another directory, in which crYOLO configurations, temporary files, and predicted coordinates will be saved.

```shell script
mkdir parsed_output
```

Use the following command to pick all micrographs in `mrc/`. A description of parameters (modified from the [PARSED user manual](/patches/parsed/original/PARSED_Manual_V1.pdf)) is given below.

```shell script
python -W ignore parsed_main.py --model=parsed/pre_train_model.h5 --data_path=mrc/ --output_path=parsed_output/ --file_pattern=*.mrc --job_suffix=autopick --angpixel=1.34 --img_size=4096 --edge_cut=0  --core_num=4 --aperture=160 --mass_min=4
```

Parameters
- `--model`: pre-trained segmentation model
- `--data_path`: path to directory of input micrographs
- `--output_path`: path to output directory
- `--file_pattern`: RegEx (Regular Expression)-like pattern that describes the filenames of the micrographs to be picked (e.g., `*.mrc` includes only files ending in `.mrc`)
- `--job_suffix`: suffix to be appended to the filenames of the output coordinate files (e.g., the input micrograph `Falcon_2012_06_12-15_27_22_0.mrc` would correspond with an output coordinate file named `Falcon_2012_06_12-15_27_22_0_autopick.star`)
- `--angpixel`: sample rate (pixel resolution) for specified micrographs (in Å/pixel)
- `--img_size`: either the width or height of the micrograph in pixels—whichever is larger
- `--edge_cut`: number of pixels to "crop" from each edge of the micrograph
- `--core_num`: number of processes to run in parallel
- `--aperture`: diameter of the particle-picking aperture (in Å)
- `--mass_min`: minimal picking mass of detected blobs (particle candidates)

Note: a GPU ID to use for picking can also be specified by appending the `--gpu_id #` flag, where `#` is the GPU ID. On an NVIDIA-based system, use the `nvidia-smi` command to get a list of available GPUs and their IDs.
