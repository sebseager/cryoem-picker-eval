# cryo-docs

Hello!

This repository (repo) contains installation guides, procedures, and patches for cryo-EM particle pickers, as followed during our experimentation for [TODO: FILL PAPER CITATION].

We recommend something like the following directory hierarchy for ease of housekeeping, and will assume this general structure from now on.

```text
main/              <-- default current working directory
├── cryo-docs/     <-- this repo
│   ├── guides/
│   ├── patches/
│   └── etc.
├── picker1/
├── picker2/
├── picker3/
├── mrc/
├── train_image/
├── train_annot/
... etc.
```

First, clone this repo into its own directory (creating directories as needed), as per the above structure. **All installation guides will begin by assuming that your current working directory is** `main/` **and that this repo is available at** `main/cryo-docs/`.

Replace `/path/to/main` with the appropriate path in which to install particle pickers (if following the above folder hierarchy). Note that here and in all guides, `/path/to/something` should be replaced by the path to the indicated resource on your system.

```shell script
mkdir -p /path/to/main
cd /path/to/main
git clone https://github.com/seb-seager/cryo-docs.git
```

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
