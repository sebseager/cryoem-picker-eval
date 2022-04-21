---
layout: default
title: Home
nav_order: 1
---

# Cryo-EM Particle Picker Comparison Analysis

## Table of Contents

- [Introduction](#introduction) (this document)
- [Particle picker guides](pickers)
- Miscellaneous
    - [EMPIAR usage guide](empiar.md)

## Introduction

This repository (repo) contains installation guides, procedures, and patches for cryo-EM particle pickers, as followed during our experimentation for [TODO: FILL PAPER CITATION].

These guides are primarily written for use on Linux systems; it may be possible to adapt some procedures to other *NIX systems like macOS with relatively minor changes, but we have not tested these use cases. Installations on Windows will likely be more complicated, and may require various supplementary procedures or compatibility layers.

We recommend something like the following directory hierarchy for ease of housekeeping, and will assume this general structure from now on.

```text
cryo-docs/              <-- this repo
├── docs/               <-- picker installation/usage guides
├── patches/
├── pickers/
│   ├── picker1/        ⎤
│   ├── picker2/        ⎥ pickers to be installed here
│   └── picker3/        ⎦
├── demo_data/
│   ├── mrc/
│   ├── train_mrc/
│   ├── train_coord/
│   ├── picker1_out/    ⎤
│   ├── picker2_out/    ⎥ picker output, created in guides
│   └── picker3_out/    ⎦
... etc.
```

If you haven't already, please clone the most recent version of this repo (which follows the above structure) and change directory into the clone. **All installation guides will begin by assuming that your current working directory is the root of this repository.** Note that here and in all guides, `/path/to/something` should be replaced by the path to the indicated resource on your system.

```shell script
git clone --depth 1 https://github.com/sebseager/cryo-docs.git && cd cryo-docs
```

This repo's `demo_data/` directory contains some sample data (five micrographs from the [EMPIAR-10017](https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10017/) beta-galactosidase data set) that can be used to test picker installations. All guides will refer to these data, but any references to `demo_data/` can be replaced with applicable paths to your own micrograph data.

To get started, take a look at our particle picker [usage guides](/docs) (or navigate using the table of contents above).

## Additional remarks

### Anaconda/Miniconda

Most of these guides assume that the Anaconda package manager (referred to as conda) is available on your system. If not, it can be installed by following [these instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). We recommend using the lighter Miniconda distribution for most applications. A PDF with a list of useful conda commands can be found [here](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

When switching between different conda environments, we will use `conda activate my_environment_name` and `conda deactivate my_environment_name`. This is the preferred method to use with conda versions 4.4.0 and above. A possible exception to this may be shell scripts or cluster job queue scripts, where you may find that the following method works better:

```shell script
source /path/to/miniconda#/bin/activate /path/to/conda_envs/my_environment_name
``` 

If the `conda activate` or `conda deactivate` commands do not work, you will need to add the conda executable to your `PATH` environment variable. To allow conda to run its built-in setup automatically, use

```shell script
/path/to/miniconda#/bin/conda init
```

or, to add conda to your `PATH` temporarily (for your current shell session only), use

```shell script
export PATH=/path/to/miniconda#/bin:$PATH
```

or, to add conda to your `PATH` such that it persists across shell sessions, add it to your `~/.bashrc` (or `~/.zshrc`, etc. depending on your shell) like so

```shell script
echo 'export PATH=/path/to/miniconda#/bin:$PATH' >> ~/.bashrc
```
