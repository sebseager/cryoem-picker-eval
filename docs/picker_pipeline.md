# Picker Pipeline

Author: Christopher JF Cameron & Sebastian JH Seager

## User-set globals

We recommend preparing a separate file containing each dataset's required parameters, then sourcing that file in your shell to set them all at once. Some example files are provided in `dataset_params/`. Please note that if you change compute devices (i.e., from a CPU node to a GPU node on a cluster), you may have to re-source these parameters.

```bash
CONDA_ACTIVATE          # path to conda activate binary
CONDA_ENVS              # path to conda environments directory
USER_ID                 # for constructing paths
DATASET_ID              # for constructing paths
DATASET_HOME            # main directory for this dataset
UTIL_SCRIPT_DIR         # path to scripts/ directory in this repo
PICKER_INSTALL_DIR      # path to picker installation directory
                        # (e.g., for pickers installed directly from git)

GT_SUFFIX               # ground truth file suffix, excluding the dot
ANG_PIX_RES             # pixel resolution, in Angstroms (A/pix)

RLN_MRC_VOLTAGE         # micrograph voltage (usu. 300 kV)
RLN_FRAME_DOSE          # frame dose in (e/A^2)
RLN_FIRST_FRAME         # 1 to take first (check paper)
RLN_LAST_FRAME          # -1 to take last (check paper)
RLN_PATCHES_X           # e.g., 5
RLN_PATCHES_Y           # e.g., 5
RLN_GROUP_FRAMES        # e.g., 1, 2, or 3; must be divisible by group
                        # (depends on num. frames in stack)
RLN_BINNING_FACTOR      # e.g., 1

MRC_LONGSIDE_PIX        # longest micrograph side length in pixels
NUM_PARTICLES_PER_MRC   # total num. ground truth particles / number of micrographs
EMAN_BOXSIZE_PIX        # even integer, following EMAN2 box size conventions
                        # (https://blake.bcm.edu/emanwiki/EMAN2/BoxSize),
                        # taken from a dataset's voxel size on the EMD experiment tab

TOPAZ_SCALE             # Topaz expects scaled pixel resolution between 4-8 A/pix
TOPAZ_PARTICLE_RAD      # int. approx. of ${EMAN_BOXSIZE_PIX} / (${TOPAZ_SCALE} ** 2)
TOPAZ_MODEL             # depends on particle diameter after downsampling:
                        # 70 pixels or less for resnet8
                        # 30 pixels or less for conv31
                        # 62 pixels or less for conv63
                        # 126 pixels or less for conv127

ASPIRE_SCALE            # see https://github.com/sebseager/cryo-docs/blob/main/patches/aspire/apple.py#L17
ASPIRE_BOXSIZE_PIX      # try either 24,48,72 from EMAN2 box sizes

CASSPER_UTIL            # cassper_util/ currently contains
                        #   labels/     manually generated labels
                        #   patches/    cassper patches
                        #   TSaved/     cassper general model source code
```

## Derived or constant globals

```bash
EMAN_BOXSIZE_A=$(echo "$EMAN_BOXSIZE_PIX * $ANG_PIX_RES / 1" | bc)
QUERY_IMAGE_SIZE=$ASPIRE_BOXSIZE_PIX
QBOX=$(((((4000 / ASPIRE_SCALE) ** 2) / (QUERY_IMAGE_SIZE ** 2)) * 4))
TAU1=$((QBOX / 33))
TAU1=${TAU1%.*}
TAU2=$((QBOX / 3))
TAU2=${TAU2%.*}
CONTAINER=$((450 / ASPIRE_SCALE))
```

## Download micrographs and particle coordinates from Globus

1. Log into [Globus](https://app.globus.org/)
2. Go to File Manager
3. Set source collection and filepath to "Shared EMBL-EBI public endpoint" and "/gridftp/empiar/world_availability/`${DATASET_ID}`/data/"
4. Set destination collection and filepath to "yale#farnam" and "/gpfs/gibbs/pi/gerstein/`${USER_ID}`/imppel/`${DATASET_ID}`/relion"
5. Select files to be downloaded from source to destination
6. Click "Start" and the following message should appear: "Transfer request submitted successfully"
7. You will recieve an email from Globus notification when transfer has completed/failed

## Data preprocessing

**This should be done on a CPU node.**

```bash
# load RELION
module load RELION/3.1.2-fosscuda-2020b

# make directories
mkdir -p ${DATASET_HOME}/relion
mkdir -p ${DATASET_HOME}/ground_truth
mkdir -p ${DATASET_HOME}/pngs

# move ground truth coordinate files to their own directory
cd ${DATASET_HOME}/relion
mv *.${GT_SUFFIX} ${DATASET_HOME}/ground_truth/
```

## Motion correction (only if necessary)

Skip this section if the micrographs downloaded from EMPIAR were already motion corrected (i.e., each `*.mrc` file is a single image, as opposed to an image stack or "movie").

**Note: this must be run on a GPU node!**

```bash
# create RELION directory
cd ${DATASET_HOME}/relion

# import job
mkdir -p Import/job001
relion_import  --do_movies  --optics_group_name "opticsGroup1" --angpix ${ANG_PIX_RES} --kV 300 --Cs 2.7 --Q0 0.1 --beamtilt_x 0 --beamtilt_y 0 --i "*.mrc" --odir Import/job001/ --ofile movies.star --pipeline_control Import/job001/

# motioncorr job
mkdir -p MotionCorr/job002
$(which relion_run_motioncorr) --i Import/job001/movies.star --o MotionCorr/job002/ --first_frame_sum ${RLN_FIRST_FRAME} --last_frame_sum ${RLN_LAST_FRAME} --use_own --j 1 --bin_factor ${RLN_BINNING_FACTOR} --bfactor 150 --dose_per_frame ${RLN_FRAME_DOSE} --preexposure 0 --patch_x ${RLN_PATCHES_X} --patch_y ${RLN_PATCHES_Y} --eer_grouping 32 --group_frames ${RLN_GROUP_FRAMES} --dose_weighting --pipeline_control MotionCorr/job002/

# if motioncorr gives the error `Patch size must be even`, then it's possible that
# one or both micrograph dimensions are not even; if this is the case, run the
# following to trim one pixel from the offending dimensions (the --sw flag slices
# the width, the --sh flag slices the height, and :-1 removes the last pixel)
mkdir orig_mrc && mv *.mrc orig_mrc
python3 scripts/slice_mrcs.py --sw :-1 --sh :-1 -o . *.mrc

# ctf estimation
mkdir -p CtfFind/job003
$(which relion_run_ctffind_mpi) --i MotionCorr/job002/corrected_micrographs.star --o CtfFind/job003/ --Box 512 --ResMin 30 --ResMax 5 --dFMin 5000 --dFMax 50000 --FStep 500 --dAst 100 --ctffind_exe /ysm-gpfs/apps/software/CTFFIND/4.1.14/ctffind --ctfWin -1 --is_ctffind4  --fast_search --pipeline_control CtfFind/job003/
```

## Create train/test/validation splits

This assumes 20 micrographs and creates a split of: 10 test, 8 train, 2 validation.

```bash
# globs pointing to all original ground truth and micrograph files
GT_FILES="${DATASET_HOME}/relion/*.${GT_SUFFIX}"
MRC_FILES="${DATASET_HOME}/relion/*.mrc"

# make all output directories
mkdir -p ${DATASET_HOME}/relion/{train_img,train_annot,val_img,val_annot,test_img,test_annot}/

# train set
eval ls ${MRC_FILES} | head -8 | xargs -n 1 -I {} cp -s --target-directory=${DATASET_HOME}/relion/train_img/ {}
eval ls ${GT_FILES} | head -8 | xargs -n 1 -I {} python ${UTIL_SCRIPT_DIR}/coord_converter.py {} ${DATASET_HOME}/relion/train_annot/ -f ${GT_SUFFIX} -t box -b ${EMAN_BOXSIZE_PIX} -s ""

# validation set
eval ls ${MRC_FILES} | head -10 | tail -2 | xargs -n 1 -I {} cp -s --target-directory=${DATASET_HOME}/relion/val_img/ {}
eval ls ${GT_FILES} | head -10 | tail -2 | xargs -n 1 -I {} python ${UTIL_SCRIPT_DIR}/coord_converter.py {} ${DATASET_HOME}/relion/val_annot/ -f ${GT_SUFFIX} -t box -b ${EMAN_BOXSIZE_PIX} -s ""

# test set
eval ls ${MRC_FILES} | tail -10 | xargs -n 1 -I {} cp -s --target-directory=${DATASET_HOME}/relion/test_img/ {}
eval ls ${GT_FILES} | tail -10 | xargs -n 1 -I {} python ${UTIL_SCRIPT_DIR}/coord_converter.py {} ${DATASET_HOME}/relion/test_annot/ -f ${GT_SUFFIX} -t box -b ${EMAN_BOXSIZE_PIX} -s ""
```

## crYOLO

**Note: this must be run on a GPU!**

```bash
# activate the cryolo environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/cryolo
```

### Predict test set with general model

```bash
# download general model
mkdir -p ${DATASET_HOME}/relion/cryolo/general
cd ${DATASET_HOME}/relion/cryolo/general
wget ftp://ftp.gwdg.de/pub/misc/sphire/crYOLO-GENERAL-MODELS/gmodel_phosnet_202005_N63_c17.h5

# configure run
cryolo_gui.py config ${DATASET_HOME}/relion/cryolo/general/config_cryolo.json ${EMAN_BOXSIZE_PIX} --filter LOWPASS --low_pass_cutoff 0.1

# predict
cryolo_predict.py -c ${DATASET_HOME}/relion/cryolo/general/config_cryolo.json -w ${DATASET_HOME}/relion/cryolo/general gmodel_phosnet_202005_N63_c17.h5 -i ${DATASET_HOME}/relion/test_img/ -g 0 -o ${DATASET_HOME}/relion/cryolo/general/ -t 0.3
```

### Refine general model weights

```bash
# make output directories
mkdir -p ${DATASET_HOME}/relion/cryolo/refined
cd ${DATASET_HOME}/relion/cryolo/refined

# configure train
cryolo_gui.py config config_cryolo.json ${EMAN_BOXSIZE_PIX} --train_image_folder ${DATASET_HOME}/relion/train_img/ --train_annot_folder ${DATASET_HOME}/relion/train_annot/ --valid_image_folder ${DATASET_HOME}/relion/val_img/ --valid_annot_folder ${DATASET_HOME}/relion/val_annot/ --pretrained_weights ${DATASET_HOME}/relion/cryolo/general/gmodel_phosnet_202005_N63_c17.h5 --saved_weights_name ${DATASET_HOME}/relion/cryolo/refined/refined_weights.h5

# train
cryolo_train.py -c ${DATASET_HOME}/relion/cryolo/refined/config_cryolo.json -w 0 -g 0 --fine_tune -lft 2
```

### Predict with refined model

```bash
# configure run
cryolo_gui.py config ${DATASET_HOME}/relion/cryolo/refined/config_cryolo.json ${EMAN_BOXSIZE_PIX} --filter LOWPASS --low_pass_cutoff 0.1

# predict
cryolo_predict.py -c ${DATASET_HOME}/relion/cryolo/refined/config_cryolo.json -w ${DATASET_HOME}/relion/cryolo/refined/refined_weights.h5 -i ${DATASET_HOME}/relion/test_img/ -g 0 -o ${DATASET_HOME}/relion/cryolo/refined/ -t 0.3
```

### Score general and refined models

```bash
# create evaluation environment (if needed)
conda create -n imppel
conda install -c anaconda networkx pandas scikit-learn tqdm yaml pyyaml
conda install -c conda-forge matplotlib-base mrcfile
conda update --all
conda clean --all

# activate evaluation environment
conda deactivate
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/imppel/

# convert general particles to box
mkdir -p ${DATASET_HOME}/relion/cryolo/general/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/cryolo/general/STAR/*.star -o ${DATASET_HOME}/relion/cryolo/general/BOX -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force

# score general model
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/cryolo/general/BOX/*.box &> ${DATASET_HOME}/relion/cryolo/general/particle_set_comp.txt

# convert refined particles to box
mkdir -p ${DATASET_HOME}/relion/cryolo/refined/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/cryolo/refined/STAR/*.star -o ${DATASET_HOME}/relion/cryolo/refined/BOX -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force

# score refined model
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/cryolo/refined/BOX/*.box &> ${DATASET_HOME}/relion/cryolo/refined/particle_set_comp.txt

# print scores
echo "general" $(tail -1 ${DATASET_HOME}/relion/cryolo/general/particle_set_comp.txt)
echo "refined" $(tail -1 ${DATASET_HOME}/relion/cryolo/refined/particle_set_comp.txt)
```

## Topaz

**Note: this must be run on a GPU!**

```bash
# activate the topaz environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/topaz
```

### Predict test set with general model

```bash
# downsample and normalize micrographs
mkdir -p ${DATASET_HOME}/relion/topaz/general
cd ${DATASET_HOME}/relion/topaz/general

# apply particle scaling
topaz preprocess -s ${TOPAZ_SCALE} -o ${DATASET_HOME}/relion/topaz/general/test_img_downsampled/ ${DATASET_HOME}/relion/test_img/*.mrc

# pick particles
topaz extract -r ${TOPAZ_PARTICLE_RAD} -x ${TOPAZ_SCALE} -o ${DATASET_HOME}/relion/topaz/general/predicted_particles_all_upsampled.txt ${DATASET_HOME}/relion/topaz/general/test_img_downsampled/*.mrc
```

### Train model from scratch

```bash
# preprocess train micrographs
topaz preprocess -s ${TOPAZ_SCALE} -o ${DATASET_HOME}/relion/topaz/refined/train_img_downsampled/ ${DATASET_HOME}/relion/train_img/*.mrc

# convert train particles to topaz input format
mkdir -p ${DATASET_HOME}/relion/topaz/refined/train_annot/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/train_annot/*.box -o ${DATASET_HOME}/relion/topaz/refined/train_annot/ -f box -t box -b ${EMAN_BOXSIZE_PIX} --round 0 -s ""
topaz convert -s ${TOPAZ_SCALE} ${DATASET_HOME}/relion/topaz/refined/train_annot/*.box -o ${DATASET_HOME}/relion/topaz/refined/train_particles.txt

# preprocess validation micrographs
topaz preprocess -s ${TOPAZ_SCALE} -o ${DATASET_HOME}/relion/topaz/refined/val_img_downsampled/ ${DATASET_HOME}/relion/val_img/*.mrc

# convert validation particles to topaz input format
mkdir -p ${DATASET_HOME}/relion/topaz/refined/val_annot/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/val_annot/*.box -o ${DATASET_HOME}/relion/topaz/refined/val_annot/ -f box -t box -b ${EMAN_BOXSIZE_PIX} --round 0 -s ""
topaz convert -s ${TOPAZ_SCALE} ${DATASET_HOME}/relion/topaz/refined/val_annot/*.box -o ${DATASET_HOME}/relion/topaz/refined/val_particles.txt

# run training
topaz train -r ${TOPAZ_PARTICLE_RAD} -n ${NUM_PARTICLES_PER_MRC} --num-workers 8 \
 --train-images ${DATASET_HOME}/relion/topaz/refined/train_img_downsampled/ \
 --train-targets ${DATASET_HOME}/relion/topaz/refined/train_particles.txt \
 --test-images ${DATASET_HOME}/relion/topaz/refined/val_img_downsampled/ \
 --test-targets ${DATASET_HOME}/relion/topaz/refined/val_particles.txt \
 --save-prefix ${DATASET_HOME}/relion/topaz/refined/model \
 --model ${TOPAZ_MODEL} \
 -o ${DATASET_HOME}/relion/topaz/refined/model_training.txt
```

### Predict test set with refined model

```bash
# pick particles using model
topaz extract -r ${TOPAZ_PARTICLE_RAD} -x ${TOPAZ_SCALE} -m ${DATASET_HOME}/relion/topaz/refined/model_epoch10.sav -o ${DATASET_HOME}/relion/topaz/refined/predicted_particles_all_upsampled.txt ${DATASET_HOME}/relion/topaz/general/test_img_downsampled/*.mrc
```

### Score general and refined models

```bash
# convert general model picks to individual box files
mkdir -p ${DATASET_HOME}/relion/topaz/general/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/topaz/general/predicted_particles_all_upsampled.txt ${DATASET_HOME}/relion/topaz/general/BOX/ -f tsv -t box -b ${EMAN_BOXSIZE_PIX} -c 1 2 none none 3 0 --header --multi_out

# score general model
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/topaz/general/BOX/*.box &> ${DATASET_HOME}/relion/topaz/general/particle_set_comp.txt

# convert refined model picks to individual box files
mkdir -p ${DATASET_HOME}/relion/topaz/refined/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/topaz/refined/predicted_particles_all_upsampled.txt ${DATASET_HOME}/relion/topaz/refined/BOX/ -f tsv -t box -b ${EMAN_BOXSIZE_PIX} -c 1 2 none none 3 0 --header --multi_out

# score refined model
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/topaz/refined/BOX/*.box &> ${DATASET_HOME}/relion/topaz/refined/particle_set_comp.txt

# print scores
echo "general" $(tail -1 ${DATASET_HOME}/relion/topaz/general/particle_set_comp.txt)
echo "refined" $(tail -1 ${DATASET_HOME}/relion/topaz/refined/particle_set_comp.txt)
```

## AutoCryoPicker

### Installation and patching

```bash
# get latest version from github
git clone https://github.com/jianlin-cheng/AutoCryoPicker.git ${PICKER_INSTALL_DIR}/AutoCryoPicker/

# apply our patch
cp YOUR/PATH/TO/patches/autocryopicker/AutoPicker_Final_Demo.m ${PICKER_INSTALL_DIR}/AutoCryoPicker//Signle\ Particle\ Detection_Demo/AutoPicker_Final_Demo.m
```

### Convert micrographs to PNG images

```bash
python ${UTIL_SCRIPT_DIR}/mrc_to_img.py ${DATASET_HOME}/relion/test_img/*.mrc -f png -o ${DATASET_HOME}/pngs
```

### Pick particles (using batch job)

```bash
# make output directory
mkdir -p ${DATASET_HOME}/relion/autocryopicker/BOX/

# write slurm batch job
cat << END > ${DATASET_HOME}/relion/autocryopicker/run_submit.script
#!/bin/bash
#SBATCH --job-name=autocryopicker
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/autocryopicker/slurm-%j.out

module load MATLAB/2020b
sleep 5s

cd ${PICKER_INSTALL_DIR}/AutoCryoPicker//Signle\ Particle\ Detection_Demo/
out_dir=${DATASET_HOME}/relion/autocryopicker/BOX/
for f in ${DATASET_HOME}/pngs/\*.png; do
    out_name=\$(basename \$f)
    label_file="\${out_dir}/\${out_name%.png}.box"
    matlab -nosplash -nodisplay -r "mrc='\$f';out_dir='\$out_dir';AutoPicker_Final_Demo" -logfile "\$label_file"
    awk '/AUTOCRYOPICKER_DETECTIONS_START/ ? c++ : c' \${label_file} > \${label_file/.box/.tmp} && mv \${label_file/.box/.tmp} \${label_file}
done
END

# submit the script
sbatch ${DATASET_HOME}/relion/autocryopicker/run_submit.script
```

### Score model

```bash
# calculate score
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/autocryopicker/BOX/*.box &> ${DATASET_HOME}/relion/autocryopicker/particle_set_comp.txt

# print score
echo "general" $(tail -1 ${DATASET_HOME}/relion/autocryopicker/general/particle_set_comp.txt)
```

## DeepPicker

### Installation and patching

```bash
# get latest version from github
git clone https://github.com/nejyeah/DeepPicker-python.git ${PICKER_INSTALL_DIR}/deeppicker

# apply our patch
cp YOUR/PATH/TO/patches/deeppicker/*.py ${PICKER_INSTALL_DIR}/deeppicker
```

### Pick particles with general model

```bash
# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/deeppicker


```

# ----------------------------------------------------------------------------------------------------------------------

# pick particles with general model

mkdir -p ${DATASET_HOME}/relion/deeppicker/general/STAR/
python /gpfs/slayman/pi/gerstein/cjc255/tools/deeppicker/autoPick.py --inputDir ${DATASET_HOME}/relion/test_img/ --pre_trained_model /gpfs/slayman/pi/gerstein/cjc255/tools/deeppicker/trained_model/model_demo_type3 --particle_size ${EMAN_BOXSIZE_PIX} --mrc_number -1 --outputDir ${DATASET_HOME}/relion/deeppicker/general/STAR/ --coordinate_symbol \_deeppicker --threshold 0.5
mkdir -p ${DATASET_HOME}/relion/deeppicker/general/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/deeppicker/general/STAR/\*.star ${DATASET_HOME}/relion/deeppicker/general/BOX/ -f star -t box -b ${EMAN_BOXSIZE_PIX} --header
rm -rf ${DATASET_HOME}/relion/deeppicker/general/STAR/

# create collection of train mrcs and particles

mkdir -p ${DATASET*HOME}/relion/deeppicker/train/
cp -s ${DATASET_HOME}/relion/train_img/*.mrc ${DATASET*HOME}/relion/deeppicker/train/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/train_annot/*.box ${DATASET_HOME}/relion/deeppicker/train/ -f box -t star -b ${EMAN_BOXSIZE_PIX} --header --force

# train model from scratch and pick particles

mkdir -p ${DATASET_HOME}/relion/deeppicker/refined/STAR/

# particle number = wc -l ${DATASET_HOME}/relion/deeppicker/train/\*.box

python /gpfs/slayman/pi/gerstein/cjc255/tools/deeppicker/train.py --train_type 1 --train_inputDir ${DATASET_HOME}/relion/deeppicker/train/ --particle_size ${EMAN_BOXSIZE_PIX} --mrc_number -1 --particle_number 4374 --coordinate_symbol '' --model_retrain --model_load_file /gpfs/slayman/pi/gerstein/cjc255/tools/deeppicker/trained_model/model_demo_type3 --model_save_dir ${DATASET_HOME}/relion/deeppicker/refined/ --model_save_file model_demo_type3_refined

# pick with trained model

python /gpfs/slayman/pi/gerstein/cjc255/tools/deeppicker/autoPick.py --inputDir ${DATASET_HOME}/relion/test_img/ --pre_trained_model ${DATASET_HOME}/relion/deeppicker/refined/model_demo_type3_refined --particle_size ${EMAN_BOXSIZE_PIX} --mrc_number -1 --outputDir ${DATASET_HOME}/relion/deeppicker/refined/STAR/ --coordinate_symbol \_deeppicker --threshold 0.5
mkdir -p ${DATASET_HOME}/relion/deeppicker/refined/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/deeppicker/refined/STAR/\*.star ${DATASET_HOME}/relion/deeppicker/refined/BOX/ -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force
rm -rf ${DATASET_HOME}/relion/deeppicker/refined/STAR/

# general -> 62.628%

# refined -> 84.181%

python ${UTIL*SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET*HOME}/relion/deeppicker/general/BOX/*.box &> ${DATASET*HOME}/relion/deeppicker/general/particle_set_comp.txt
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET*HOME}/relion/deeppicker/refined/BOX/*.box &> ${DATASET_HOME}/relion/deeppicker/refined/particle_set_comp.txt

# visual comparison

mkdir -p ${DATASET_HOME}/relion/deeppicker/vis_cmp/

# cp -s ${DATASET_HOME}/relion/test_annot/Falcon_2012_06_12-15_43_48_0.box ${DATASET_HOME}/relion/deeppicker/vis_cmp/ground_truth.box

cp -s ${DATASET_HOME}/relion/deeppicker/refined/BOX/Falcon_2012_06_12-15_43_48_0_deeppicker.box ${DATASET_HOME}/relion/deeppicker/vis_cmp/refined.box
cp -s ${DATASET_HOME}/relion/deeppicker/general/BOX/Falcon_2012_06_12-15_43_48_0_deeppicker.box ${DATASET_HOME}/relion/deeppicker/vis_cmp/general.box

python ${UTIL_SCRIPT_DIR}/plot_boxfile.py -m ${DATASET_HOME}/relion/test_img/Falcon_2012_06_12-15_43_48_0.mrc -g ${DATASET_HOME}/relion/test_annot/Falcon_2012_06_12-15_43_48_0.box -p ${DATASET_HOME}/relion/deeppicker/vis_cmp/\*.box --num_gt 32 -o ${DATASET_HOME}/relion/deeppicker/vis_cmp/ --force &> ${DATASET_HOME}/relion/deeppicker/vis_cmp/stdout.txt

##

# identify particles with ASPIRE APPLE-picker

##

# make output dirs

mkdir -p ${DATASET_HOME}/relion/aspire/STAR/

# rescale micrographs

mkdir -p ${DATASET_HOME}/relion/aspire/test_img_downsampled/
for input in ${DATASET_HOME}/relion/test_img/*.mrc
    do basename=${input##\*/}
singularity exec /gpfs/ysm/datasets/cryoem/eman2.3_ubuntu18.04.sif e2proc2d.py --meanshrink=${ASPIRE_SCALE} ${input} ${DATASET_HOME}/relion/aspire/test_img_downsampled/${basename}
done

# process micrographs in batch job

cat << END > ${DATASET_HOME}/relion/aspire/run_submit.script
#!/bin/bash
#SBATCH --job-name=aspire
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/aspire/slurm-%j.out

source ${CONDA_ACTIVATE} ${CONDA_ENVS}/aspire

# this script is a wrapper for APPLE picker and bypasses the ASPIRE config system

python ${UTIL*SCRIPT_DIR}/../pickers/aspire/apple_cli.py ${DATASET_HOME}/relion/aspire/test_img_downsampled/*.mrc -o ${DATASET*HOME}/relion/aspire/STAR/ --particle_size ${ASPIRE_BOXSIZE_PIX} --max_particle_size $((ASPIRE_BOXSIZE_PIX * 2)) --min_particle_size $((ASPIRE_BOXSIZE_PIX / 4)) --minimum_overlap_amount $((ASPIRE_BOXSIZE_PIX / 10)) --query_image_size ${QUERY_IMAGE_SIZE} --tau1 ${TAU1} --tau2 ${TAU2} --container_size ${CONTAINER}
END

sbatch ${DATASET_HOME}/relion/aspire/run_submit.script

# convert from STAR to BOX format

python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/aspire/STAR/\*.star ${DATASET_HOME}/relion/aspire/tmp/ -f star -t box -b ${ASPIRE_BOXSIZE_PIX} --force --round 0

# rm -rf ${DATASET_HOME}/relion/aspire/STAR/

# upsample particle coordinates

source ${CONDA_ACTIVATE} ${CONDA_ENVS}/topaz
mkdir -p ${DATASET_HOME}/relion/aspire/BOX/
for input in ${DATASET_HOME}/relion/aspire/tmp/\*.box
do topaz convert ${input} -o ${DATASET_HOME}/relion/aspire/BOX/ --up-scale ${ASPIRE_SCALE} --to box --boxsize ${EMAN_BOXSIZE_PIX}
done
rm -rf ${DATASET_HOME}/relion/aspire/tmp/

# compare particle picks against ground ground truth

# general model -> 81.015%

python ${UTIL*SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET*HOME}/relion/aspire/BOX/*.box &> ${DATASET_HOME}/relion/aspire/particle_set_comp.txt

##

##

# identify particle with PARSED

##

# install

#

# note: the source files in the cp command are already patched;

# original software is a static .zip in the paper's supplementary

# and is available extracted at imppel/docs/pickers/parsed/original/

#

# note: PARSED manual is available at /gpfs/gibbs/pi/gerstein/sjs264/imppel/docs/pickers/parsed/PARSED_Manual_V1.pdf

#

# mkdir -p /gpfs/slayman/pi/gerstein/cjc255/tools/PARSED

# cp /gpfs/gibbs/pi/gerstein/sjs264/imppel/docs/pickers/parsed/\*.{py,h5} /gpfs/slayman/pi/gerstein/cjc255/tools/PARSED

# opencv2 needs to be v3.4.3: https://github.com/facebookresearch/maskrcnn-benchmark/issues/339

# conda create -n parsed -c conda-forge python=3.6 h5py=2.7.1 keras=2.0.8 numba=0.37.0 pandas=0.20.3 matplotlib=2.1.0 mrcfile=1.1.2 trackpy=0.4.1 tensorflow-gpu opencv=3.4.3 tqdm

# conda activate parsed

# make output dirs

mkdir -p ${DATASET_HOME}/relion/parsed/STAR/

# pick particles with general model WITHOUT blob thresholding

# (i.e. no data-specific optimization)

cat << END > ${DATASET_HOME}/relion/parsed/run_submit.script
#!/bin/bash
#SBATCH --job-name=parsed
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/parsed/slurm-%j.out

module load CUDA/8.0.61
sleep 5s

module load cuDNN/8.0.5.39-CUDA-11.1.1
sleep 5s

source ${CONDA_ACTIVATE} ${CONDA_ENVS}/parsed
sleep 5s

cd /gpfs/slayman/pi/gerstein/cjc255/tools/PARSED
python -W ignore parsed_main.py --model=pre_train_model.h5 --data_path=${DATASET_HOME}/relion/test_img/ --output_path=${DATASET_HOME}/relion/parsed/STAR/ --file_pattern=\*.mrc --job_suffix=parsed --angpixel=${ANG_PIX_RES} --img_size=${MRC_LONGSIDE_PIX} --edge_cut=0 --core_num=4 --aperture=${EMAN_BOXSIZE_A}
END

sbatch ${DATASET_HOME}/relion/parsed/run_submit.script

source ${CONDA_ACTIVATE} ${CONDA_ENVS}/parsed/

# plot distribution of detected particles

cd /gpfs/slayman/pi/gerstein/cjc255/tools/PARSED
python particle_mass.py drawmass --pick_output=${DATASET_HOME}/relion/parsed/STAR/ --job_suffix=parsed --tmp_hist=${DATASET_HOME}/relion/parsed/hist.png

# threshold particles - NOT NEEDED FOR 10017

# -> threshold needs to be manually determined

# -> open plot in previous step to determine threshold

# -> only perform cutoff if mass distribution is multimodal

# PARSED_THRES=100

# python particle_mass.py cutoff --pick_output=${DATASET_HOME}/relion/parsed/STAR/ --job_suffix=parsed --output_suffix=parsed_thres --thres=$PARSED_THRES

source ${CONDA_ACTIVATE} ${CONDA_ENVS}/imppel/

# convert particles from STAR to BOX format

mkdir -p ${DATASET_HOME}/relion/parsed/BOX/

# \_parsed_thres

python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/parsed/STAR/\*\_parsed.star ${DATASET_HOME}/relion/parsed/BOX/ -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force
rm -rf ${DATASET_HOME}/relion/parsed/STAR/

# score particles

# trained -> 77.24%

python ${UTIL*SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET*HOME}/relion/parsed/BOX/*.box &> ${DATASET_HOME}/relion/parsed/particle_set_comp.txt

##

# identy particle with RELION LoG auto-picking

##

# removed for now

# DIAMETER_BG_CIRCLE_PIX=$(( EMAN_BOXSIZE_PIX*0.75 + ((EMAN_BOXSIZE_PIX*0.75)%2) ))

# DOWNSCALING_FACTOR=4

# DOWNSCALED_BOXSIZE_PIX=$(( EMAN_BOXSIZE_PIX/DOWNSCALING_FACTOR + ((EMAN_BOXSIZE_PIX/DOWNSCALING_FACTOR)%2) ))

# -> Rescale particles? Yes

# -> Re-scaled sized (pixels)? $DOWNSCALED_BOXSIZE_PIX

# note: things starting with '$' are variables and should be evaluated, e.g. with `echo`, before entering into RELION

# note: this assumes we have already done a CtfFind job; if this is not the case, please add that job before step 1

# 0) Import (x3) train, test, and validation

# repeat the following for each coordinate set path (COORD_DIR)

# train: ${DATASET_HOME}/relion/train_annot/

# test: ${DATASET_HOME}/relion/test_annot/

# val: ${DATASET_HOME}/relion/val_annot/

COORD_DIR=# train/test/val path from above
cp ${COORD_DIR}/\*.box ${DATASET_HOME}/relion/

# Import job

# -> Movies/mics / Import raw movies/micrographs? No

# -> Others / Input file: \*.box

# -> Node type: Particle coordinates (_.box, _\_pick.star)

# -> (may want to name the job so you know which is train/test/val)

# after running this, do:

rm ${DATASET_HOME}/relion/\*.box

# then repeat for all three coordinate sets

# TEMPLATE-FREE STEP

# 1) Auto-picking

# -> Input micrographs for autopick: Import/jobXXX/micrographs.star

# -> OR: use Laplacian-of-Gaussian? Yes

# -> Min. diameter for loG filter (A): 319 ~= $EMAN_BOXSIZE_A

# -> Max. diameter for loG filter (A): 319 ~= $EMAN_BOXSIZE_A

# -> 1 MPI procs, Yes, pi_gerstein, sbatch

# 2D CLASS GENERATION

# 2) Particle extraction

# -> micrograph STAR file: Import/jobXXX/micrographs_ctf.star

# -> Coordinate-filesuffix: AutoPick/jobXXX/coordssuffixautopick.star

# -> Particle box size (pix): $EMAN_BOXSIZE_PIX

# -> Rescale particles? Yes

# -> Re-scaled sized (pixels - must be even number)? $((EMAN_BOXSIZE_PIX/2))

# -> 1 MPI procs, Yes, pi_gerstein, sbatch

# 3) 2D classification

# -> Extract/jobXXX/particles.star

# -> Have data been phase-flipped? No

# -> Number of classes: 32

# -> Use fast subsets for large data sets? No (except for 10470=Yes)

# -> Mask diameter (A): $((EMAN_BOXSIZE_A \* 0.75))

# -> Number of pooled particles: 32

# -> Use GPU acceleration? Yes

# -> Which GPUs to use: <leave_blank>

# -> 5 MPI procs, Yes, gpu, sbatch, No. of GPUs: 1

# 4) Subset select

# -> Select classes from model.star: Class2D/jobXXX/run_it025_model.star

# -> Regroup the particles? No (change to Yes if RELION warns about small particle groups)

# -> Run -> order by reverse sort on rlnClassDistribution

# -> Select good class averages and try not to include many similar views

# -> right click and "Save selected classes" option

# TEMPLATE-BASED STEP

# 5) Auto-picking

# -> Input micrographs for autopick: Import/jobXXX/micrographs.star

# -> 2D references: Select/jobXXX/class_averages.star

# -> Or: using Laplacian-of-Gaussian: No

# -> Pixel size in references (A): 3.54 ~= $((ANG_PIX_RES \* 2))

# -> Adjust default threshold: 0.8

# -> Minimum inter-particle distance (A): 159.3 == 90 \* 1.77 == $((EMAN_BOXSIZE_PIX/2))

# -> Write FOM maps? Yes

# -> Use GPU acceleration? Yes

# -> Which GPUs to use: 0

# -> 1 MPI procs, Yes, pi_gerstein_gpu, sbatch, No. of GPUs: 1

# convert particles from STAR to BOX format

mkdir -p ${DATASET_HOME}/relion/log/BOX/

# update job ID and wildcard/suffix for your RELION instance/filenames

python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/AutoPick/job007/\*0_autopick.star ${DATASET_HOME}/relion/log/BOX/ -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force

# score particles

# trained -> 27.58%

python ${UTIL*SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET*HOME}/relion/log/BOX/*.box &> ${DATASET_HOME}/relion/log/particle_set_comp.txt

##

# identify particles with CASSPER

##

# create/active Conda environment

# conda create --name cassper python=3.6

# conda activate cassper

# conda install -c anaconda pip joblib # add. joblib dependency, not in requirements.txt

# conda update --all

# conda clean --all

# pip install -r ${DATASET_HOME}/relion/cassper/git/requirements.txt

# conda env remove --name cassper

mkdir -p ${DATASET_HOME}/relion/cassper

# Install cassper

git clone https://github.com/airis4d/CASSPER.git ${DATASET_HOME}/relion/cassper/git
cd ${DATASET_HOME}/relion/cassper/git/Train_and_Predict

# cd /gpfs/slayman/pi/gerstein/cjc255/tools

# git clone https://github.com/airis4d/~/CASSPER.git

# cp -r /gpfs/slayman/pi/gerstein/cjc255/tools/CASSPER/\* $DATASET_HOME/relion/cassper

# Activate environment

$SRUN_GPU
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/cassper/
module load cuDNN/6.0-CUDA-8.0.61
module load CUDA/8.0.61

# Apply patches

# - Train_and_Predict/extract_coordinates_from_labels.py

# - Train_and_Predict/PSTrain.py

# - Train_and_Predict/train2.py

cp ${CASSPER_UTIL}/patches/Train_and_Predict/\* ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/.

### Pick particles with CASSPER general model

# make required dirs

mkdir -p ${DATASET_HOME}/relion/cassper/general/STAR
mkdir -p ${DATASET_HOME}/relion/cassper/general/Predict_labels

# copy class_dict.csv

cp ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/class_dict.csv ${DATASET_HOME}/relion/cassper/general/

# preprocess micrographs to PNGs as input to model

# -> PNGs stored in ${DATASET_HOME}/relion/cassper/general/Predict/

# you must be in ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/

cd ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/
mkdir -p ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/ProtiSEM_metadata
python ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/PSPred.py ${DATASET_HOME}/relion/cassper/general/ ${DATASET_HOME}/relion/test_img/

# run general model

# src - https://drive.google.com/drive/folders/1BM-YxFBcFSCCQX1rakDIvk9chP-w1gMH

# create image segmentations

# -> predicted segmentations in ${DATASET_HOME}/relion/cassper/general/Predict_labels/

python ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/predict2.py --dataset ${DATASET_HOME}/relion/cassper/general/ --model FRRN-B --image "" --checkpoint_path ${CASSPER_UTIL}/TSaved/BestFr_InceptionV4_model_FRRN-B_F1.ckpt

# extract particle coordinates from segmentation

python ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/extract_coordinates_from_labels.py -i ${DATASET_HOME}/relion/cassper/general/Predict_labels -o ${DATASET_HOME}/relion/cassper/general/STAR

# popup window -> choose particle size via scroll bar, then press `esc` (only needed once)

# size of 48 used for 10017

# convert particles from STAR to BOX format

# change Conda environment (conflict of Numpy versions b/w cassper and @Seb's comparison script)

source ${CONDA_ACTIVATE} ${CONDA_ENVS}/topaz
mkdir -p ${DATASET_HOME}/relion/cassper/general/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/cassper/general/STAR/\*.star ${DATASET_HOME}/relion/cassper/general/BOX/ -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force

# score particles

# general -> 74.968%

python ${UTIL*SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET*HOME}/relion/cassper/general/BOX/*.box &> ${DATASET_HOME}/relion/cassper/general/particle_set_comp.txt

### Train CASSPER model from scratch and pick particles

for sub_dir in "pre_train_labels" "pre_val_labels" "labels" "test" "test_labels" "checkpoints" "TSaved" "STAR" "Predict_labels" "BOX"
do mkdir -p ${DATASET_HOME}/relion/cassper/train_from_scratch/${sub_dir}
done

# -> train/val segmentation labels in pre_train_labels/ and pre_val_labels/ ()

# -> model output PNGs will be stored in train/ and val/

# -> RGB PNGs will be stored in train_labels/ and val_labels/

# copy author labels

ls /gpfs/ysm/scratch60/gerstein/cjc255/imppel/10017/relion/cassper/git/Labelling*Tool/Created_labels/10017/*.png | head -8 | xargs -n 1 -I {} cp {} ${DATASET*HOME}/relion/cassper/train_from_scratch/pre_train_labels/
ls /gpfs/ysm/scratch60/gerstein/cjc255/imppel/10017/relion/cassper/git/Labelling_Tool/Created_labels/10017/*.png | head -10 | tail -2 | xargs -n 1 -I {} cp {} ${DATASET*HOME}/relion/cassper/train_from_scratch/pre_val_labels/
cp -s ${DATASET_HOME}/relion/*.mrc ${DATASET*HOME}/relion/train_img
cp -s ${DATASET_HOME}/relion/*.mrc ${DATASET_HOME}/relion/val_img

# copy David's labels

cp ${CASSPER_UTIL}/labels/${DATASET*ID}/pre_train_labels/*.png ${DATASET_HOME}/relion/cassper/train_from_scratch/pre_train_labels/
cp ${CASSPER_UTIL}/labels/${DATASET*ID}/pre_val_labels/*.png ${DATASET_HOME}/relion/cassper/train_from_scratch/pre_val_labels/

# copy class_dict.csv

cp ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/class_dict.csv ${DATASET_HOME}/relion/cassper/train_from_scratch/

# generate new labels (presumably already done by David)

# python ${DATASET_HOME}/relion/cassper/git/Labelling_Tool/label_generator.py -i ${DATASET_HOME}/relion/train_img -o ${DATASET_HOME}/relion/cassper/train_from_scratch/pre_train_labels

# python ${DATASET_HOME}/relion/cassper/git/Labelling_Tool/label_generator.py -i ${DATASET_HOME}/relion/val_img -o ${DATASET_HOME}/relion/cassper/train_from_scratch/pre_val_labels

# preprocess the labels, turn mrcs into pngs with PSTrain.py

mkdir -p ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/ProtiSEM_metadata

cat << END >> ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/run_submit_preprocess.script
#!/bin/bash
#SBATCH --job-name=cassper_preprocess
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/cassper/git/Train_and_Predict/slurm-%j.out

source ${CONDA_ACTIVATE} ${CONDA_ENVS}/cassper/
sleep 5
module load cuDNN/6.0-CUDA-8.0.61
sleep 5
module load CUDA/8.0.61
sleep 5

cd ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/
python ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/PSTrain.py ${DATASET_HOME}/relion/cassper/train_from_scratch/ ${DATASET_HOME}/relion/ ${DATASET_HOME}/relion/cassper/train_from_scratch/
END

sbatch ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/run_submit_preprocess.script

# train model from scratch (took 18 min)

# -> model gets saved TSaved/

# 10017 -> ran with default settings: model-FC-DenseNet56 and frontend-ResNet101

# you should both try the default and the following: model-FRRN-B and frontend-InceptionV4

# default: model: FC-DenseNet56 and frontend: ResNet101

# # default

CASSPER_MODEL="FC-DenseNet56"
CASSPER_FRONTEND="ResNet101"

# # other

# CASSPER_MODEL="FRRN-B"

# CASSPER_FRONTEND="InceptionV4"

cat << END >> ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/run_submit_train.script
#!/bin/bash
#SBATCH --job-name=cassper_train
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/cassper/git/Train_and_Predict/slurm-%j.out

source ${CONDA_ACTIVATE} ${CONDA_ENVS}/cassper/
sleep 5
module load cuDNN/6.0-CUDA-8.0.61
sleep 5
module load CUDA/8.0.61
sleep 5

cd ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/
python ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/train2.py --dataset ${DATASET_HOME}/relion/cassper/train_from_scratch/ --num_epochs 100 --continue_training 0 --h_flip 1 --v_flip 1 --rotation 8 --brightness 0.5 --model ${CASSPER_MODEL} --frontend ${CASSPER_FRONTEND}
END

sbatch ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/run_submit_train.script

# preprocess micrographs to PNGs as input to model

# -> PNGs are stored in Predict/

# must be in git/Train_and_Predict

cd ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/
python ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/PSPred.py ${DATASET_HOME}/relion/cassper/train_from_scratch/ ${DATASET_HOME}/relion/test_img/

# run trained model

# BestFr_ResNet101_model_FC-DenseNet56_F1.ckpt or BestFr_InceptionV4_model_FRRN-B_F1.ckpt

python ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/predict2.py --dataset ${DATASET_HOME}/relion/cassper/train_from_scratch/ --model ${CASSPER_MODEL} --image "" --checkpoint_path ${DATASET_HOME}/relion/cassper/train_from_scratch/TSaved/BestFr_${CASSPER*FRONTEND}\_model*${CASSPER_MODEL}\_F1.ckpt
python ${DATASET_HOME}/relion/cassper/git/Train_and_Predict/extract_coordinates_from_labels.py -i ${DATASET_HOME}/relion/cassper/train_from_scratch/Predict_labels -o ${DATASET_HOME}/relion/cassper/train_from_scratch/STAR

# convert particles from STAR to BOX format

# change Conda environment (conflict of Numpy versions b/w cassper and @Seb's comparison script)

source ${CONDA_ACTIVATE} ${CONDA_ENVS}/topaz
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/cassper/train_from_scratch/STAR/\*.star ${DATASET_HOME}/relion/cassper/train_from_scratch/BOX/ -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force

# score particles

# 10057 score = 0.728-3.053%

# author's labels -> score = 1.266%

python ${UTIL*SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET*HOME}/relion/cassper/train_from_scratch/BOX/*.box &> ${DATASET_HOME}/relion/cassper/train_from_scratch/particle_set_comp.txt

# -------- TODO ----------

### 2D class analysis

# adjust box file naming convention for 10057

for f in ${DATASET_HOME}/ground_truth/*.*.*.box
    do i="${f%.box}"
mv -i -- "$f" "${i//./\_}.box"
done

cp ${DATASET_HOME}/ground_truth/*.${GT_SUFFIX} ${DATASET_HOME}/relion/

# 5) Import

# -> Others/Input file: ../ground_truth/\*.box

# -> Others/Node type: Particle coordinate

rm ${DATASET_HOME}/relion/*.${GT_SUFFIX}

# 6) Particle extract

# -> I/O/micrograph STAR file: CTFFind STAR file

# -> I/0/Input coordinates: last job's STAR file

# -> extract/particle box size: 240 (10057)

# -> running: 5 MPI procs, Yes, pi_gerstein, sbatch

# 7) 2D classification

### src: :https://discuss.cryosparc.com/t/good-cryosparc-2d-classes-blurry-relion-classes/3007

# -> I/O/input images STAR file: extract STAR file

# -> CTF/ignore CTFs until first peak? Yes

# -> Optimization/number of classes: 32

# -> Optimization/use fast subsets: Yes

# -> Optimization/mask diameter: 176\*1.35 ~ 238

# -> Optimization/limit resolution E-step to: 10

# -> Compute/number of pooled particles: 32

# -> Compute/use GPU acceleration: Yes

# -> 5 MPI procs, Yes, gpu, sbatch, no. of GPUs=4

```

```
