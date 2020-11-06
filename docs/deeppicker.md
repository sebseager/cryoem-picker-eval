# DeepPicker

## Summary

DeepPicker is available as a [GitHub repo](https://github.com/nejyeah/DeepPicker-python). We used Python 3.6 for this installation.

The DeepPicker paper can be found [here](https://arxiv.org/abs/1605.01838). Refer also to the [README](https://github.com/nejyeah/DeepPicker-python/blob/master/README.md) for more information. The following guide is drawn in part from these resources.

## Compatibility

We have tested DeepPicker successfully on RHEL (Red Hat Enterprise Linux) 7.7. According to the maintainers,

> it only supports Ubuntu 12.0+, centOS 7.0+, and RHEL 7.0+

## Installation [TODO: TEST THIS SECTION]

First, clone the ASPIRE repo into `pickers/deeppicker/` using 

```shell script
git clone https://github.com/nejyeah/DeepPicker-python.git pickers/deeppicker
```

Create and activate a new conda environment using Python version 3.6 (since `tensorflow` versions 1.12.0 and earlier do not support Python 3.7).

```shell script
conda create -n deeppicker matplotlib scipy python=3.6
conda activate deeppicker
```

DeepPicker requires the `tensorflow` machine learning package, which in turn requires `cudatoolkit` and `cudnn` in order to support CUDA-compatible GPUs (if your system already has global installations of CUDA and cuDNN, feel free to skip the following discussion). These can be installed manually, but it is important to use the correct version of each to avoid errors. See [this compatibility chart](https://www.tensorflow.org/install/source#gpu) for more information. The [DeepPicker GitHub](https://github.com/nejyeah/DeepPicker-python#1-install-tensorflow) indicates that `cudatoolkit` 7.5 and `cudnn` 4 should be used, but these versions are quite old and may not be compatible with modern GPU hardware (i.e., CUDA 7.5 does not support Pascal GPUs or later).

To install the latest versions of `tensorflow-gpu` and all its compatible dependencies (including `cudatoolkit` and `cudnn`), use

```shell script
conda install tensorflow-gpu
```

or, to specify a `tensorflow-gpu` version (which in turn will pull the correct versions of `cudatoolkit` and `cudnn`), use 

```shell script
conda install tensorflow-gpu=#.##.##
```

Note that `conda search tensorflow-gpu` can be used to see which versions of `tensorflow-gpu` are available with conda. If the version you would like to install is not available in your conda channels, you can either specify a different channel or install everything manually with `pip`—see [release history](https://pypi.org/project/tensorflow-gpu/#history) for `tensorflow-gpu`.

## Usage

### Overview

DeepPicker is a *trainable* particle picker, that when presented coordinate files of particles along with the micrographs of origin, can produce new `.h5` files that DeepPicker can then operate by to make refined picks. 

Thus, the dfesired application is important:
- If wanting to pick by the general model that DeepPicker comes with, follow heading 'Pick using pretrained model'
- If wanting to train a new model and then pick with that model, follow heading 'Training a new model' and then 'Pick using pretrained model' (*but substitute the `pre_trained_model` parameter with the name of the newly created `.h5` file*)

### Training a new model

#### Specifics

**File formatting:** DeepPicker has a very strict format of `.star` files that it can comprehend. Our script [INSERT SOMETHING], which converts `.box` ground truth coordinate files from EMPIAR into readable `.star` files, may be of help. 

**File location:** The `.star` files *in proper format* have to be in the same folder as the micrographs they correspond to.

**File naming:** The `.star` files also have to have the same name as the corresponding `.mrc` file, except with an identifying suffix at the end (e.g., the micrograph `Falcon_2012_06_12-15_27_22_0.mrc` would correspond with the coordinate file `Falcon_2012_06_12-15_27_22_0_cnnPick.star`, as there is a suffix at the end; the suffix string will be identified in the script command for training for DeepPicker)

#### Running training

Assuming the aforementioned specifics are met, use this command, with tailoring of parameters. Parameters outlined below as well.

```shell script
python train.py --train_type 1 --train_inputDir "input_dir" --particle_size ### --mrc_number -1 --particle_number -1 --coordinate_symbol 'some_string' --model_save_dir 'output_dir' --model_save_file 'output_model_name'
```

Parameters
- `--train_inputDir`: input directory of `.star` and corresponding `.mrc` files
- `--train_type`: options are 1, 2, 3, or 4. 1 is recommended, for specimen-specific new models; 2 for multiple molecules, 3 for iterative training.
- `--mrc_number`: number of `.mrc` files to pick from the directory specified; default=-1 refers to all
- `--particle_size`: the size of the particle
- `--coordinate_symbol`: suffix that identifies `.star` file for each `.mrc` file; refer [Specifics]
- `--model_save_dir`: the directory to save the model `.h5` file to
- `--model_save_file`: the name of the model `.h5` file


### Pick using pretrained model

Start by collecting the micrograph files (`*.mrc`) to be picked in a directory (assuming they are not already available in their own directory). If you would like to use an existing public data set, [our guide to the EMPIAR database](empiar.md) may be helpful.

```shell script
mkdir -p name_of_data_set/mrc
mv path/to/your_mrc_files/*.mrc name_of_data_set/mrc
```

Here we will use the micrographs located in `demo_data/` as an example. Create another directory, in which any output, temporary, or configuration files will be saved by the picker.

```shell script
mkdir demo_data/deeppicker_out
```

Use the following command to pick all micrographs in `demo_data/mrc/`. A description of parameters is given below.

```shell script
python autoPick.py --inputDir 'demo_data/mrc/' --pre_trained_model 'pretrained_or_created_model' --particle_size ### --mrc_number -1 --outputDir 'demo_data/deepicker_out' --coordinate_symbol 'text_indicator' --threshold 0.5
```

Parameters
- `--inputDir`: input directory of `.mrc` files
- `--pre_trained_model`: the `.h5` model file
- `--mrc_number`: number of `.mrc` files to pick from the directory specified; default=-1 refers to all
- `--particle_size`: the size of the particle
- `--outputDir`: output directory to save the coordinate `.star` files
- `--coordinate_symbol`: suffix to be appended to the filenames of the output coordinate files (e.g., the input micrograph `Falcon_2012_06_12-15_27_22_0.mrc` would correspond with an output coordinate file named `Falcon_2012_06_12-15_27_22_0_cnnPick.star`)





