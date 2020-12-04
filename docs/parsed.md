---
layout: default
parent: pickers
---

# PARSED

## Summary

PARSED is available as a collection of Python programs. We used Python 3.6 for this installation.

The PARSED paper can be found [here](https://doi.org/10.1093/bioinformatics/btz728). Both the PARSED [package files](/pickers/parsed/original) and a [user manual](/pickers/parsed/PARSED_Manual_V1.pdf) can be downloaded in the `parsed_v1.zip` archive available with the supplementary materials of the PARSED paper. We include all relevant files in `pickers/parsed/` of this repo for ease of access. In doing so, we adopt for these files the same usage license identified in the paper, namely:

> the PARSED package and user manual [are available] for noncommercial use

## Installation

First, make a new directory for the PARSED package. 

```shell script
mkdir parsed
```

PARSED is available as it was originally published in `pickers/parsed/original/`. Modified versions of these files with our patches and bug fixes are available in in `pickers/parsed/`. We will assume the patched files are being used, but the original software can be used at any time by replacing references to `pickers/parsed/` with `pickers/parsed/original` where applicable.

Create and activate a new conda environment with the required dependencies. It may take conda several tries to solve this environment due to the highly specific package dependencies. The versions specified for each package correspond with those listed on the first page of the [PARSED user manual](/pickers/parsed/PARSED_Manual_V1.pdf). It may be possible to use later versions of some packages (e.g., `mrcfile`) without adverse effects.

```shell script
conda create -n parsed -c conda-forge h5py=2.7.1 keras=2.0.8 numba=0.37.0 pandas=0.20.3 matplotlib=2.1.0 mrcfile=1.1.2 trackpy=0.4.1 tensorflow-gpu=1.7.0 pip python=3.6
conda activate parsed
```

Install any additional dependencies (not directly available through conda) with `pip`. In doing so, however, it is important that we use the `pip` executable that was just installed in the conda environment created above. To verify this, check that running `which pip` outputs something like `/path/to/conda_envs/parsed/bin/pip`. If not, try restarting your shell session or running `conda deactivate parsed` followed by `conda activate parsed` before proceeding.

```shell script
pip install opencv-python==3.1.*
```

## Usage

### Overview

Inputs
- micrographs for which to pick particles
- picking aperture size, micrograph pixel resolution, and other configurable parameters
- general model `*.h5` file

Outputs (in a single directory)
- `*.star` coordinate files (for context, `.star` files are text-based file formats for storing data, essentially containing the titles and collections of data in text)
- `*.star` extended parameter files (containing coordinates, but also blob mass and other data)
- `*.h5` data files (not used)

### Pick using pretrained model

Start by collecting the micrograph files (`*.mrc`) to be picked in a directory (assuming they are not already available in their own directory). If you would like to use an existing public data set, [our guide to the EMPIAR database](empiar.md) may be helpful.

```shell script
mkdir -p name_of_data_set/mrc
mv path/to/your_mrc_files/*.mrc name_of_data_set/mrc
```

Here we will use the micrographs located in `demo_data/` as an example. Create another directory, in which any output, temporary, or configuration files will be saved by the picker.

```shell script
mkdir demo_data/parsed_out
```

Use the following command to pick all micrographs in `demo_data/mrc/`. A description of parameters (modified from the [PARSED user manual](/pickers/parsed/PARSED_Manual_V1.pdf)) is given below.

```shell script
python -W ignore parsed_main.py --model=pickers/parsed/pre_train_model.h5 --data_path=demo_data/mrc/ --output_path=demo_data/parsed_out/ --file_pattern=*.mrc --job_suffix=autopick --angpixel=1.34 --img_size=4096 --edge_cut=0  --core_num=4 --aperture=160 --mass_min=4
```

When providing arguments for the command above, relative directories can be specified with `./` and `../`, but do not use `~`, **as it does not appear to expand to the user's home folder**. Note that if PARSED does not recognize a path, it may exit with the usual `No such file or directory`, or it may also attempt to pick anyway, returning `Coordinates extraction finished 0 TotalPickNum found`.

Parameters
- `--model`: pre-trained segmentation model
- `--data_path`: path to directory of input micrographs
- `--output_path`: path to output directory
- `--file_pattern`: regex (regular expression)-like pattern that describes the filenames of the micrographs to be picked (e.g., `*.mrc` includes only files ending in `.mrc`)
- `--job_suffix`: suffix to be appended to the filenames of the output coordinate files (e.g., the input micrograph `Falcon_2012_06_12-15_27_22_0.mrc` would correspond with an output coordinate file named `Falcon_2012_06_12-15_27_22_0_autopick.star`)
- `--angpixel`: sample rate (pixel resolution) for specified micrographs (in Å/pixel)
- `--img_size`: either the width or height of the micrograph in pixels—whichever is larger
- `--edge_cut`: number of pixels to "crop" from each edge of the micrograph
- `--core_num`: number of processes to run in parallel
- `--aperture`: diameter of the particle-picking aperture (in Å)
- `--mass_min`: minimal picking mass of detected blobs (particle candidates)

Note: a GPU ID to use for picking can also be specified by appending the `--gpu_id #` flag, where `#` is the GPU ID. On an NVIDIA-based system, use the `nvidia-smi` command to get a list of available GPUs and their IDs.
