---
layout: default
parent: Particle Pickers
---

# Topaz

## Summary

Topaz is available as a conda package. We used Python 3.6 for this installation.

The Topaz paper can be found [here](https://doi.org/10.1038/s41592-019-0575-8). Refer also to the [GitHub repo](https://github.com/tbepler/topaz), the [web-based GUI](https://emgweb.nysbc.org/topaz.html) for generating Topaz commands, and the [tutorials](https://github.com/tbepler/topaz/tree/master/tutorial) included in the Topaz repo for more information. The following guide is drawn in part from these resources.

## Installation

Create and activate a new conda environment (named `topaz`) with the required dependencies.

```shell script
conda create -n topaz python=3.6
conda activate topaz
```

With the `topaz` environment active, install the Topaz package with its dependencies.

```shell script
conda install topaz -c tbepler -c pytorch
```

If your machine/computing cluster does not already have a global installation of `cudatoolkit` available, it can be installed to the `topaz` environment, as follows.

```shell script
conda install cudatoolkit=9.0 -c pytorch
```

Verify that the Topaz installation works by running `topaz --help`. If it does, you should see a help menu. If your shell returns something like `topaz: command not found`, you may need to deactivate and reactivate the environment.

```shell script
conda deactivate topaz
conda activate topaz
```
 
If you see a traceback (error), however, you may also need to install the `future` package. Make sure you're in the `topaz` environment before doing so.

```shell script
conda install future
```

## Usage

The authors of Topaz provide [several detailed tutorials](https://github.com/tbepler/topaz/tree/master/tutorial) in their repository—in particular, in their [quick start guide](https://github.com/tbepler/topaz/blob/master/tutorial/01_quick_start_guide.ipynb) and [more detailed walkthrough](https://github.com/tbepler/topaz/blob/master/tutorial/02_walkthrough.ipynb). The following outlines broader usage examples and practices.

Start by collecting the micrograph files (`*.mrc`) to be picked in a directory (assuming they are not already available in their own directory). If you would like to use an existing public data set, [our guide to the EMPIAR database](empiar.md) may be helpful.

```shell script
mkdir -p name_of_data_set/mrc
mv path/to/your_mrc_files/*.mrc name_of_data_set/mrc
```

Here we will use the micrographs located in `demo_data/` as an example.

### Preprocessing

Create a Topaz output directory with subdirectories for micrograph preprocessing.

```shell script
mkdir -p demo_data/topaz_out/processed/micrographs
```

We will now process the micrographs with the `preprocess` command (which combines `downsample` and `normalize` operations). This will downscale the micrographs by a specified multiplier, which is intended to help the neural network learn and converge more quickly. It is suggested in the [detailed walkthrough](https://github.com/tbepler/topaz/blob/master/tutorial/02_walkthrough.ipynb) that a downsampling multiplier should be chosen as follows:

> We recommend downsampling your data enough that the diameter of your particle fits within the receptive field of the CNN architecture you are using ... as a rule of thumb, downsampling to about 4-8 Å per pixel generally works well, but this may need to be adjusted for very large or very small particles to fit the classifier

For reference, the training tab of the [Topaz GUI](https://emgweb.nysbc.org/topaz.html) provides the following classifier specifications:

> Your particle must have a diameter (longest dimension) after downsampling of:
> - 70 pixels or less for resnet8
> - 30 pixels or less for conv31
> - 62 pixels or less for conv63
> - 126 pixels or less for conv127

For example, if the original micrograph resolution was 1.2 Å/pix, a downsampling factor of 5 would bring the preprocessed micrograph's resolution to 6 Å/pix. The preprocessing command would be as follows.

```shell script
topaz preprocess -s 5 -o demo_data/topaz_out/processed/micrographs/ demo_data/mrc/*.mrc
```

Topaz has two primary picking strategies: using the [pretrained general model](#method-1-use-pretrained-general-model-as-is), or [training a new model](#method-2-train-a-model-from-scratch) (optionally initialized with pretrained weights).

### Method 1: Use pretrained general model as-is

The `extract` command takes input and output paths, as well two numerical parameters. The `-r` parameter should be set to the radius of the particle you would like to pick. It is recommended that this be kept relatively small (as appropriate for your particle), as Topaz will not pick particles any closer than this to prevent multiple detections per particle. The `-x` parameter will upscale the resulting picks to the original micrograph, and should be the same as `-s` from `topaz preprocess`.

```shell script
topaz extract -r 14 -x 5 -o demo_data/topaz_out/predicted_particles_all_upsampled.txt demo_data/topaz_out/processed/micrographs/*.mrc
```

### Method 2: Train a model from scratch

In order to isolate training data, a subset of the micrographs (here, some `train_1.mrc`, `train_2.mrc`, etc. in `demo_data/mrc/`) should be placed into a separate directory (`demo_data/train_mrc/`). These images, along with some known particle coordinates for each (in `demo_data/train_coord/`), will be used to train the model. crYOLO matches an image to its corresponding coordinate file by comparing the filenames (e.g. `demo_data/train_mrc/Falcon_2012_06_12-14_57_34_0.mrc` and `demo_data/train_coord/Falcon_2012_06_12-14_57_34_0.star` would be paired).

```shell script
mkdir demo_data/train_mrc demo_data/train_coord
mv demo_data/mrc/{train_1.mrc,train_2.mrc,train_3.mrc} demo_data/train_mrc/
```

To populate `train_coord/`, software like EMAN2's `e2boxer` may be used to generate coordinate files for the training micrographs. Topaz also provides an online graphical interface (located [here](https://emgweb.nysbc.org/topaz.html), in the `Pick | Analyze` tab) which can be used to generate training coordinates. For the sake of example, the `.star` files located in `demo_data/star` can be used. Note that Topaz takes training coordinates as a single file of the following format (columns are separated by single `\t` tab characters):
 
 ```text
image_name  x_coord y_coord
Falcon_2012_06_12-14_57_34_0    3822    3477
Falcon_2012_06_12-14_57_34_0    3810	3402
...
```
 
 Topaz also supports conversion from other file formats using its `topaz convert` utility. We also provide a conversion utility at `tools/coord_converter.py` that may be helpful.
 
 Before training can proceed, the training coordinate files must be downscaled by the same factor used to downscale the micrographs. Assuming a downscaling factor of 6, and that your training data are available in `demo_data/train_mrc/` and `demo_data/train_coord/`:
 
```shell script
topaz convert -s 6 -o demo_data/topaz_out/processed/particles.txt demo_data/train_coord/particles.txt
```

We then make new directories for our new model.

```shell script
mkdir -p demo_data/topaz_out/saved_models
```

Topaz training can then be run as follows, where `-n` represents the approximate number of particles expected per micrograph.

```shell script
topaz train -n 400 --num-workers 8 \
            --train-images demo_data/topaz_out/processed/micrographs/ \
            --train-targets demo_data/topaz_out/processed/particles.txt \
            --save-prefix demo_data/topaz_out/saved_models/model \
            -o demo_data/topaz_out/saved_models/model_training.txt
```

See `topaz train --help` for more detailed explanations of the arguments.

This trained model can now be used to extract particles. In the following command, the `-r` parameter should be set to the radius of the particle you would like to pick. It is recommended that this be kept relatively small (as appropriate for your particle), as Topaz will not pick particles any closer than this to prevent multiple detections per particle. The `-x` parameter will upscale the resulting picks to the original micrograph, and should be the same as `-s` from `topaz preprocess`. The `-m` parameter should point to the last epoch of the model trained above.

```shell script
topaz extract -r 14 -x 5 -m demo_data/topaz_out/saved_models/model_epoch10.sav \
              -o demo_data/topaz_out/predicted_particles_all_upsampled.txt \
              demo_data/topaz_out/processed/micrographs/*.mrc
```

### Detection format conversion

Topaz provides a utility to convert the format of the `extract` output file, which can be used like so if needed (e.g. to convert from `.txt` to `.star`):

```shell script
topaz convert -o demo_data/topaz_out/predicted_particles_all_upsampled.star demo_data/topaz_out/predicted_particles_all_upsampled.txt
```

We also provide a script at `tools/coord_converter.py` that may be useful for coordinate file conversion.
