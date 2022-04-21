#!/bin/bash

CONDA_ACTIVATE="/gpfs/slayman/pi/gerstein/cjc255/tools/miniconda3/bin/activate"
CONDA_ENVS="/gpfs/slayman/pi/gerstein/cjc255/conda_envs"

# paths
USER_ID="sjs264"
DATASET_ID="10470"
DATASET_HOME="/gpfs/gibbs/pi/gerstein/${USER_ID}/imppel/${DATASET_ID}"
UTIL_SCRIPT_DIR="/gpfs/gibbs/pi/gerstein/sjs264/imppel/docs/scripts/"
PICKER_INSTALL_DIR="/gpfs/slayman/pi/gerstein/cjc255/tools/"

# ground-truth file suffix, including the dot
GT_SUFFIX="star"

# pixel resolution
ANG_PIX_RES=1.06

RLN_MRC_VOLTAGE=500 
RLN_FRAME_DOSE=2.5  
RLN_PATCHES_X=5     
RLN_FIRST_FRAME=1
RLN_LAST_FRAME=-1
RLN_PATCHES_Y=5     
RLN_GROUP_FRAMES=3  
RLN_BINNING_FACTOR=1

# micrograph (longest) side length in pixels
MRC_LONGSIDE_PIX=4096

# box size taken from EMDB map information: https://www.ebi.ac.uk/emdb/EMD-2824?tab=experiment
# follows EMAN2 box size conventions: https://blake.bcm.edu/emanwiki/EMAN2/BoxSize
# box size must be an even integer!
EMAN_BOXSIZE_PIX=320

# ** double check the downsampling factor s for your data
# 10017 has a pixel reoslution of 1.7 and Topaz expects between 4-8
TOPAZ_SCALE=8

# ${EMAN_BOXSIZE_PIX} / (${TOPAZ_SCALE} ** 2) == 180/(4**2) ~ 12
TOPAZ_PARTICLE_RAD=20

TOPAZ_MODEL="resnet8"

# 11599 GT particles/20 mrc ~ 580
NUM_PARTICLES_PER_MRC=24

# src: https://github.com/sebseager/cryo-docs/blob/main/patches/aspire/apple.py#L17
ASPIRE_SCALE=2

# try either 24,48,72 from EMAN2 box sizes
ASPIRE_BOXSIZE_PIX=24

# cassper_util currently contains
# - labels/: labels generated by David
# - patches/: cassper patches, currently not uploaded to Seb's github
# - TSaved/: cassper general model source code
CASSPER_UTIL="/gpfs/gibbs/pi/gerstein/dp823/cassper_util/"