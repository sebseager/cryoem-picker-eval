# Topaz

## Summary

Topaz is available as a conda package.

The Topaz paper can be found [here](https://doi.org/10.1038/s41592-019-0575-8). Refer also to the [GitHub repo](https://github.com/tbepler/topaz), the [web-based GUI](https://emgweb.nysbc.org/topaz.html) for generating Topaz commands, and the [tutorials](https://github.com/tbepler/topaz/tree/master/tutorial) included in the Topaz repo for more information.

## Installation

*Note: modified from https://github.com/tbepler/topaz/blob/master/README.md*

Create and activate a new conda environment with the required dependencies. It is possible to use `conda activate topaz` instead of `source activate topaz` for conda versions ≥ 4.4.

```shell script
conda create -n topaz python=3.6
source activate topaz
```

With the `topaz` environment active, install the Topaz package with its dependencies.

```shell script
conda install topaz -c tbepler -c pytorch
```

If your machine/computing cluster does not already have a global installation of `cudatoolkit` available, it can be installed to the `topaz` environment, as follows.

```shell script
conda install cudatoolkit=9.0 -c pytorch
```

Verify that the Topaz installation works by running `topaz --help`. If it does, you should see a help menu. If your shell returns something like `topaz: command not found`, you may need to deactivate and reactivate the environment (as above, `source` can be replaced with `conda` depending on your Anaconda version).

```shell script
source deactivate topaz
source activate topaz
```
 
If you see a traceback (error), however, you may also need to install the `future` package. Make sure you're in the `topaz` environment before doing so.

```shell script
conda install future
```

## Usage

