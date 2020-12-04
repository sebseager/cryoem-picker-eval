---
layout: default
parent: Particle Pickers
---

# AutoCryoPicker

## Summary

AutoCryoPicker is available as a [GitHub Repo](https://github.com/jianlin-cheng/AutoCryoPicker), as a matlab application. We used Matlab 2017b for running all commands.

## Installation

First, clone the AutoCryoPicker repo into `pickers/autocryopicker/` using 

```shell script
git clone https://github.com/jianlin-cheng/AutoCryoPicker.git pickers/autocryopicker
```

In some cases, AutoCryoPicker is very slow in running, due to loading many picture files during the particle picking process. If you would like to speed it up, we have
a patch under `cryo-docs/patches/deeppicker` to remove these pictures. This script will replace several AutoCryoPicker source files with our patched versions (keeping 
the others as cloned in the previous step).
