---
layout: default
parent: pickers
---

# Scipion

## Summary

Scipion is a cryo-electron microscopy image processing framework. It was presented in the Journal of Structural Biology in 2016: [Scipion: A software framework toward integration, reproducibility and validation in 3D electron microscopy](https://doi.org/10.1016/j.jsb.2016.04.010).

The [Scipion official documentation](https://scipion-em.github.io/docs/index.html) may be a helpful resource.

## Installation

The following installation instructions are optimized for the Farnam compute cluster in the Yale High Performance Computing center. Thanks to Michael Strickler for his help in getting this procedure to work. For further information, see the [install notes](https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html) in the documentation.

CUDA is not located at `/usr/local/cuda` on the clusters, so you do not need to set the PATH. CUDA 8.0 support is pulled in when you load the IMOD module, but you might instead select CUDA 10.1 as suggested by the instructions, along with a specific GCC compiler.

```shell script
module load GCCcore/7.3.0 CUDA/10.1.243 OpenMPI/3.1.1-GCC-7.3.0-2.30
```

Create a separate Conda environment for running the installer.

```shell script
conda create -y --name scipion-installer pip
​conda activate scipion-installer
```

Run the installer module, pointing it to the directory created above.

```shell script
mkdir path/to/new/scipion/directory
python -m scipioninstaller path/to/new/scipion/directory -j 4
```

A new conda environment called `scipion3` should have been created. Activate it.

```shell script
conda deactivate  # deactivate scipion-installer
# the installer environment can optionally be deleted
conda activate scipion3
```

​Add missing dependencies to the `scipion3` environment. The instructions do not specify a NumPy version, but the code appears to require NumPy 1.18.4 or lower.

```shell script
conda install -c anaconda sqlite hdf5 libopencv numpy=1.18.4 scipy mpi4py libtiff
conda install -c eumetsat fftw3
```

Then from inside this environment, run `scipion3` located in the directory you specified above.

```shell script
./path/to/new/scipion/directory/scipion3
```

Note that this installation procedure has not yet been tested fully. In particular, we rely on the cluster's OpenMPI instead of installing it via conda, to avoid unexpected behavior with multiple OpenMPI instances running.
