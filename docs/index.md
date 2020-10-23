---
layout: default
title: Cryo-EM Particle Picker Docs
---

# Cryo-EM Particle Picker Docs

Hello!

To get started, take a look at our particle picker [usage guides](guides).

## Additional remarks

### Anaconda/Miniconda

Most of these guides assume that the Anaconda package manager (referred to as conda) is available on your system. If not, it can be installed by following [these instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). We recommend using the lighter Miniconda distribution for most applications. A PDF with a list of useful conda commands can be found [here](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

When switching between different conda environments, we will use `conda activate my_environment_name` and `conda deactivate my_environment_name`. This is the preferred method to use with conda versions 4.4.0 and above. A possible exception to this may be shell scripts or cluster job queue scripts, where you may find that the following method works better:

```shell script
source /path/to/miniconda#/bin/activate /path/to/conda_envs/my_environment_name
``` 

If the `conda activate` or `conda deactive` commands do not work, you will need to add the conda executable to your `PATH` environment variable. To allow conda to run its built-in setup automatically, use

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
