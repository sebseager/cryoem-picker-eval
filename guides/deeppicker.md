# DeepPicker

## Summary

DeepPicker is available as a [GitHub repo](https://github.com/nejyeah/DeepPicker-python).

The DeepPicker paper can be found [here](https://arxiv.org/abs/1605.01838). Refer also to the [README](https://github.com/nejyeah/DeepPicker-python/blob/master/README.md) for more information. The following guide is drawn in part from these resources.

## Compatibility

We have tested DeepPicker successfully on RHEL (Red Hat Enterprise Linux) 7.7. According to the maintainers,

> it only supports Ubuntu 12.0+, centOS 7.0+, and RHEL 7.0+

## Installation - *****NOT DONE*****

Create and activate a new conda environment with the required dependencies.

```shell script
conda create -n deeppicker python=3.6
conda activate topaz
```

With the `topaz` environment active, install the Topaz package with its dependencies.

```shell script
conda install topaz -c tbepler -c pytorch
```

If your machine/computing cluster does not already have a global installation of `cudatoolkit` available, it can be installed to the `topaz` environment, as follows.

```shell script
conda install cudatoolkit=7.5 -c pytorch
```
