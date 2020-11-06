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

### Inputs

