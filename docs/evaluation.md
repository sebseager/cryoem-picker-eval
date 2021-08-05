---
layout: default
parent: pickers
---

# Evaluation Procedures

Only parameters relevant to picker tuning are included here; others can be kept at their defaults or supplied on a case-by-case basis.

## ASPIRE APPLEpicker

### Training parameters

N/A

### Picking parameters

* `particle_size` (default: 78)
* `query_image_size` (default: 52)
* `max_particle_size` (default: 156)
* `min_particle_size` (default: 19)
* `tau1` (default: 710): percentage of training images believed to contain a particle
* `tau2` (default: 7100): percentage of training images believed may potentially contain a particle

## AutoCryoPicker

[MATLAB - this was Mihir's so not a clue, yet]

## DeepPicker

### Training parameters

* `train_type` (1, 2, 3, or 4)
    1. single-molecule CNN training
    2. multiple-molecule CNN training
    3. iterative training (pick with pre-trained first)
    4. training via RELION Class2D
* `particle_size`: the size of the particle
* `particle_number` (only for train types 2, 3, or 4)
    2. "The default value is -1, so all the particles in the data file will be used for training. If it is set to 10000, and there are two kinds of molecules, then each one contributes only 5,000 positive samples."
    3. "If the value is ranging (0,1), then it means the prediction threshold. If the value is ranging (1,100), then it means the proportion of the top sorted ranking particles. If the value is larger than 100, then it means the number of top sorted ranking particles."
    4. "The default value is -1, so all the particles in the `classification2D.star` file will be used as training samples."

### Picking parameters

* `particle_size`: the size of the particle
* `threshold`: confidence threshold to pick particle, the default is 0.5.

## PARSED

### Training parameters

N/A

### Picking parameters

* `aperture`: diameter of the particle-picking aperture (in Å)
* `mass_min`: minimal picking mass of detected blobs (particle candidates)

## SPHIRE-crYOLO

### Training parameters

(For `cryolo_gui.py config`)

* `box_size`
* `pretrained_weights` (if refining general model)

(For `cryolo_train.py`)

* `w`: number of warmup epochs (e.g., 0 for refinement, 5 for from-scratch)
* `fine_tune` (include for refinement, leave out for from-scratch)
* `lft`: layers to fine-tune (e.g., 2 for refinement, leave out for from-scratch)

### Picking parameters

* `t`: picking threshold (default 0.3)

(See sphire-cryolo page of these docs for parameters to use to pick filaments.)

## Topaz

### Preprocessing parameters

* `s`: downsampling multiplier (should be constant and based on particle size and model architecture; see the [Topaz training walkthrough](https://github.com/tbepler/topaz/blob/master/tutorial/02_walkthrough.ipynb) for more on how this works)

### Training parameters

* `n`: approx. particles expected per micrograph
* `k`: perform k-fold validation (as an alternative to manually specifying train/validation sets)
* `model`: model architecture to use (`s` in preprocessing must agree with max particle diameter indicated below)
    1. resnet8 (longest particle diameter <= 70 pix after downsampling)
    2. conv31 (longest particle diameter <= 30 pix after downsampling)
    3. conv63 (longest particle diameter <= 62 pix after downsampling)
    4. conv127 (longest particle diameter <= 126 pix after downsampling)
* `radius`: "This sets how many micrograph regions around each particle coordinate are considered positive by assigning all regions with centers within radius pixels to be positive regions. By default, this is set to 3. It is recommended to make this smaller than your particle radius, but not so small that there are very few positive regions per particle (unless you have a lot of labeled coordinates (1000+))."
* `no_pretrained`: set to train from scratch

### Picking parameters

* `r`: particle radius
* `x`: upscale particle coordinates (should match `s` from preprocessing)
