# cryo-docs

Hello!

This repository (repo) contains installation guides and patches for cryo-EM particle pickers.

Most of these guides assume that the Anaconda package manager (referred to as conda) is available on your system. If 
not, it can be installed by following 
[these instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). We recommend using the 
lighter Miniconda distribution for most applications. A PDF with a list of useful conda commands can be found 
[here](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

We also recommend something like the following directory hierarchy for ease of housekeeping, and will assume this 
general structure from now on.

```text
pickers/            <-- default current working directory
├── cryo-docs/      <-- this repo
│   ├── guides/
│   ├── patches/
│   └── etc.
├── picker1/
├── picker2/
└── picker3/
```

First, clone this repo into its own directory (creating directories as needed), as per the above structure. 
**All installation guides will begin by assuming that your current working directory is** `pickers/` **and that this 
repo is available at** `pickers/cryo-docs/`.

Replace `/path/to/pickers` with the appropriate path in which to install particle pickers (if following the above 
folder hierarchy).

```shell script
mkdir -p /path/to/pickers
cd /path/to/pickers
git clone https://github.com/seb-seager/cryo-docs.git
```

To get started, take a look at our particle picker [installation guides](guides).
