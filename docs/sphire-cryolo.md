---
layout: default
parent: Particle Pickers
---

# SPHIRE-crYOLO

## Summary

crYOLO is available as a Python package from PyPI. We used Python 3.6 for this installation.

The crYOLO paper can be found [here](https://doi.org/10.1038/s42003-019-0437-z). Refer also to the [wiki](https://sphire.mpg.de/wiki/doku.php?id=pipeline:window:cryolo) (potentially outdated) and the [readthedocs user guide](https://cryolo.readthedocs.io/en/latest/) for more information. The following guide is drawn in part from these resources.

## Installation

Create and activate a new conda environment with the required dependencies.

```shell script
conda create -n cryolo -c conda-forge -c anaconda python=3.6 pyqt=5 cudnn=7.1.2 numpy==1.14.5 cython wxPython==4.0.4 intel-openmp==2019.4 pip
conda activate cryolo
```

Since crYOLO is available through PyPI, it can be installed using the package manager `pip`. In doing so, however, it is important that we use the `pip` executable that was just installed in the conda environment created above. To verify this, check that running `which pip` outputs something like `/path/to/conda_envs/cryolo/bin/pip`, and that `which python` outputs something like `/path/to/conda_envs/cryolo/bin/python`.

To install crYOLO with GPU support (recommended, if you have a GPU available), run 

```shell script
pip install 'cryolo[gpu]'
```

or, to install crYOLO with CPU support only, run 

```shell script
pip install 'cryolo[cpu]'
```

### General models

crYOLO provides three general models, each of which has been trained on a variety of data sets. They are available in [this section](https://cryolo.readthedocs.io/en/latest/installation.html#download-the-general-models) of the user guide. The following commands can also be used to download the most current versions (as of October 1, 2020) to your current working directory.

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

## Usage

### Overview

Inputs
- micrographs for which to pick particles
- particle box size and other configurable parameters
- manually picked particle coordinate files for training set (optional)
- general model `*.h5` file (optional unless using JANNI denoising or performing model training)

Outputs
- directory containing `*.box` coordinate files
- directory containing `*.star` coordinate files
- directory containing `*.cbox` coordinate files
- directory containing confidence distributions and other statistical data

### Setup

Start by collecting the micrograph files (`*.mrc`) to be picked in a directory (assuming they are not already available in their own directory). If you would like to use an existing public data set, [our guide to the EMPIAR database](empiar.md) may be helpful.

```shell script
mkdir -p name_of_data_set/mrc
mv path/to/your_mrc_files/*.mrc name_of_data_set/mrc
```

Here we will use the micrographs located in `demo_data/` as an example. Create another directory, in which any output, temporary, or configuration files will be saved by the picker.

```shell script
mkdir demo_data/cryolo_out
```

This guide covers usage of crYOLO from the command line. A crYOLO GUI (see [this tutorial](https://cryolo.readthedocs.io/en/stable/tutorials/tutorial_overview.html#start-cryolo)) is also available, which provides approximately the same functionality as the command line interface. The GUI requires either a physical monitor connected to the machine running crYOLO or an X11 display forwarding configuration (which sends the GUI to your machine over SSH). The GUI can be accessed by running `cryolo_gui.py` with no arguments.

The remainder of this guide will refer to the command line interface, but users of the GUI may be able to follow along (e.g., the `cryolo_predict.py` command is represented graphically by the `predict` action in the sidebar).

### Box size determination

According to [EMAN2 box size standards](https://blake.bcm.edu/emanwiki/EMAN2/BoxSize), 

> the particle box-size must be 1.5-2x the size of the largest axis of your particle

and, for optimal performance, should be selected from the following list:

> 16, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 84, 96, 100, 104, 112, 120, 128, 132, 140, 168, 180, 192, 196, 208, 216, 220, 224, 240, 256, 260, 288, 300, 320, 352, 360, 384, 416, 440, 448, 480, 512, 540, 560, 576, 588, 600, 630, 640, 648, 672, 686, 700, 720, 750, 756, 768, 784, 800, 810, 840, 864, 882, 896, 900, 960, 972, 980, 1000, 1008, 1024

### Creating training data

*Note: crYOLO can pick particles using its general model alone. If you would like to do this, feel free to skip this section and continue to [configuration method 1](#method-1-use-pretrained-general-model-as-is).*

 However, it is also possible to refine that model or train a new one from scratch to fit your data (see configuration methods [2](#method-2-refine-the-general-model-to-your-data) and [3](#method-3-train-your-own-model-from-scratch) below). This section describes how training data can be created.

In order to isolate training data, a subset of the micrographs (here, some `train_1.mrc`, `train_2.mrc`, etc. in `demo_data/mrc/`) should be placed into a separate directory (`demo_data/train_mrc/`). These images, along with some known particle coordinates for each (in `demo_data/train_coord/`), will be used to train the model. crYOLO matches an image to its corresponding coordinate file by comparing the filenames (e.g. `demo_data/train_mrc/Falcon_2012_06_12-14_57_34_0.mrc` and `demo_data/train_coord/Falcon_2012_06_12-14_57_34_0.box` would be paired).

```shell script
mkdir demo_data/train_mrc demo_data/train_coord
mv demo_data/mrc/{train_1.mrc,train_2.mrc,train_3.mrc} demo_data/train_mrc/
```

To populate `train_coord/`, the provided `cryolo_boxmanager.py` or other tools like `e2boxer` (see EMAN2 wiki [here](https://blake.bcm.edu/emanwiki/EMAN2/Programs/e2boxer)) can be used. They provide a graphical interface for manually selecting particle coordinates. Note that currently `cryolo_boxmanager.py` does not support filaments, so `e2helixboxer.py` is recommended in that case (more on that [here](https://cryolo.readthedocs.io/en/stable/tutorials/tutorial_overview.html#id6)).

To open a micrograph for manual picking in the crYOLO box manager, run `cryolo_boxmanager.py`, click `File` → `Open image folder`, and select the `train_mrc/` directory created and populated above. From here, there are options to apply a temporary low-pass filter or change the box size (see the [Box size determination](#box-size-determination) section above).

Within a manual picking window,
- Left-click a particle to select it
- Left-click-and-drag to move a box
- Use `Ctrl` + left-click to delete a box

To save your picks, select `File` → `Write box files`. 

### Configuration

Configure crYOLO by choosing a configuration file location, box size (220 in this example), and image filter. Use one of the following commands to generate picking configuration files, depending on the data set you are working with, the kind of model you would like to use, and whether or not you would like to train your own model.

Note that for each of the `config` commands given below, other options are usually available (run `cryolo_gui.py config --help` for a good description of each). Most will probably not need to be changed.

#### Method 1. Use pretrained general model as-is

The crYOLO [general model](https://cryolo.readthedocs.io/en/latest/other/other.html#general-model-data-sets) is trained on the following EMPIAR data sets:

> 10023, 10004, 10017, 10025, 10028, 10050, 10072, 10081, 10154, 10181, 10186, 10160, 10033, 10097

as well as simulated data sets based on the following PDB entries:

> 1sa0, 5lnk, 5xnl, 6b7n, 6bhu, 6dmr, 6ds5, 6gdg, 6h3n, 6mpu

Use one of these commands to generate a configuration file.

| Model               | Sample command                                                                                                          |
|---------------------|-------------------------------------------------------------------------------------------------------------------------|
| Phosaurus low-pass* | `cryolo_gui.py config demo_data/cryolo_out/cryolo_config.json 220 --filter LOWPASS --low_pass_cutoff 0.1`               |
| JANNI-denoised**    | `cryolo_gui.py config demo_data/cryolo_out/cryolo_config.json 220 --filter JANNI --janni_model /path/to/janni_model.h5` |
| Negative stain      | `cryolo_gui.py config demo_data/cryolo_out/cryolo_config.json 220 --filter NONE`                                        |

*\* The* `low_pass_cutoff` *can be changed to anything between 0 and 0.5 inclusive, but 0.1 is the default*

*\*\* The* `janni_model.h5` *referenced here can be downloaded at the end of the [Installation](#installation) section above.*

#### Method 2. Refine the general model to your data

*We assume that you have a directory* `demo_data/train_mrc/` *containing a set of training micrographs, and another* `demo_data/train_coord/` *containing corresponding coordinate files. If not, please take a look at the [Creating training data](#creating-training-data) section above.*

To allow crYOLO to separate automatically a random 20% of the training set for use as a validation set (default), run

```shell script
cryolo_gui.py config demo_data/cryolo_out/cryolo_config.json 220 --train_image_folder demo_data/train_mrc/ --train_annot_folder demo_data/train_coord/ --pretrained_weights /path/to/one_of_the_gmodels.h5
```

Otherwise, to specify validation images and their corresponding coordinate files, make new directories `demo_data/valid_mrc/` and `demo_data/valid_coord/`, populate them with micrographs and box files accordingly, and run the configuration command.

```shell script
mkdir demo_data/valid_mrc demo_data/valid_coord
mv demo_data/train_mrc/{train_1.mrc,train_2.mrc} demo_data/valid_mrc/
mv demo_data/train_coord/{train_1.box,train_2.box} demo_data/valid_coord/
cryolo_gui.py config demo_data/cryolo_out/cryolo_config.json 220 --train_image_folder demo_data/train_mrc/ --train_annot_folder demo_data/train_coord/ --pretrained_weights /path/to/one_of_the_gmodels.h5 --valid_image_folder demo_data/valid_mrc/ --valid_annot_folder demo_data/valid_coord/
```

#### Method 3. Train your own model from scratch

*We assume that you have a directory* `demo_data/train_mrc/` *containing a set of training micrographs, and another* `demo_data/train_coord/` *containing corresponding coordinate files. If not, please take a look at the [Creating training data](#creating-training-data) section above.*

To allow crYOLO to separate automatically a random 20% of the training set for use as a validation set (default), run

```shell script
cryolo_gui.py config demo_data/cryolo_out/cryolo_config.json 220 --train_image_folder demo_data/train_mrc/ --train_annot_folder demo_data/train_coord/
```

Otherwise, to specify validation images and their corresponding coordinate files, make new directories `demo_data/valid_mrc/` and `demo_data/valid_coord/`, populate them with micrographs and box files accordingly, and run the configuration command.

```shell script
mkdir demo_data/valid_mrc demo_data/valid_coord
mv demo_data/train_mrc/{train_1.mrc,train_2.mrc} demo_data/valid_mrc/
mv demo_data/train_coord/{train_1.box,train_2.box} demo_data/valid_coord/
cryolo_gui.py config demo_data/cryolo_out/cryolo_config.json 220 --train_image_folder demo_data/train_mrc/ --train_annot_folder demo_data/train_coord/ --valid_image_folder demo_data/valid_mrc/ --valid_annot_folder demo_data/valid_coord/
```

### Training

*Note: this section only applies to configuration methods [2](#method-2-refine-the-general-model-to-your-data) and [3](#method-3-train-your-own-model-from-scratch) above.*

Use a command below according to your intended training method. These commands save their output in the current working directory, so to keep everything organized, they can be run as follows:

```shell script
(cd demo_data/cryolo_out/ && INSERT_COMMAND_HERE)
```

Note that it is possible to run method 3 below (training from scratch) using configuration method [2](#method-2-refine-the-general-model-to-your-data) above. This allows the new model's weights to be initialized closer (potentially) to the values they ought to end up at, while still performing "from scratch" training (*not* refinement).

| Method                      | Sample command                                                               |
|-----------------------------|------------------------------------------------------------------------------|
| 1. General model as-is      | N/A (no training required)                                                   |
| 2. General model refinement | `cryolo_train.py -c path/to/cryolo_config.json -w 0 -g 0 --fine_tune -lft 2` |
| 3. Training from scratch    | `cryolo_train.py -c path/to/cryolo_config.json -w 5 -g 0`                    |

The `-w` flag sets the number of warmup epochs, and must be zero when using `--fine_tune` (for model refinement). The `-lft` flag sets the number of layers to fine tune, for which the authors recommend a default of 2. The `-g` flag indicates the GPU ID(s) to be used in training. On an NVIDIA-based system, use the `nvidia-smi` command to get a list of available GPUs and their IDs. Specify multiple GPUs with something like `-g '0 1 2'`.

You might also consider adding the `--cleanup` flag, which deletes filtered images after training, if you would like to conserve storage space on your system.

### Picking

To pick particles (not filaments) for every micrograph in `mrc/` using either one of the general models (if following configuration/training method 1) or a model you refined or trained in the previous section, run the following:

```shell script
cryolo_predict.py -c demo_data/cryolo_out/cryolo_config.json -w path/to/model.h5 -i mrc/ -g 0 -o demo_data/cryolo_out/ -t 0.3
```

The flag `-t` sets the confidence threshold (i.e., how "sure" crYOLO is that a particular detection is actually a particle) below which picks will not be included in the output `*.star` and `*.box` coordinate files. Regardless of this value, however, all picks—along with their confidence values—will be recorded in `demo_data/cryolo_out/CBOX/*.cbox` files.

If picking filaments, the following command can be used:

```shell script
cryolo_predict.py -c demo_data/cryolo_out/cryolo_config.json -w path/to/model.h5 -i mrc/ -g 0 -o demo_data/cryolo_out/ -t 0.3 --filament -fw 100 -bd 20 -mn 6
```

where the `-fw` indicates filament width in pixels, `-bd` indicates the distance between adjacent boxes on the filament, and `-mn` indicates the smallest number of boxes that are allowed to constitute a filament.
