---
layout: default
parent: Particle Pickers
---

# AutoCryoPicker

## Summary

AutoCryoPicker is built for MATLAB and is available as a [GitHub repo](https://github.com/jianlin-cheng/AutoCryoPicker). We ran it in MATLAB 2020b.

## Installation

First, clone the AutoCryoPicker repo into `pickers/autocryopicker/` using 

```shell script
git clone https://github.com/jianlin-cheng/AutoCryoPicker.git pickers/autocryopicker
```

As is, AutoCryoPicker produces detections for its own intensity-based clustering (IBC) approach, as well as for k-means and fuzzy c-means clustering algorithms (for comparison purposes). It also does not output particle centers, radii, or confidence metrics by default.

To apply our patch, replace the `AutoPicker_Final_Demo.m` file in the `Signle Particle Detection_Demo` [sic] directory with our version as follows:

```shell script
sh patches/autocryopicker/patch_autocryopicker.sh pickers/autocryopicker/
```

## Usage

Before running AutoCryoPicker, `.mrc` micrograph files must be converted to a `.png` format. [TODO: add link to script to do this (message me for this for now)]

Assuming the MATLAB executable is available at the `matlab` command, use the following to run AutoCryoPicker on a single micrograph:

```shell script
png_file=path/to/mrc.png
out_name=$(basename $png_file); matlab -nosplash -nodisplay -r "mrc='$png_file';AutoPicker_Final_Demo" -logfile "${out_name%.png}.box"
```

[TODO: in shell above get rid of anything before AUTOCRYOPICKER_DETECTIONS_START]
