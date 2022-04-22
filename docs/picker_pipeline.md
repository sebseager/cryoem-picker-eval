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

Set these after sourcing the above variables (or append them to the end of your parameter script to set everything at once).

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
GT_FILES="${DATASET_HOME}/ground_truth/*.${GT_SUFFIX}"
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

Predict test set with general model. **Note: this must be run on a GPU!**

```bash
# download general model
mkdir -p ${DATASET_HOME}/relion/cryolo/general
cd ${DATASET_HOME}/relion/cryolo/general
wget ftp://ftp.gwdg.de/pub/misc/sphire/crYOLO-GENERAL-MODELS/gmodel_phosnet_202005_N63_c17.h5

# write slurm batch job
cat << END > ${DATASET_HOME}/relion/cryolo/general/run_submit.script
#!/bin/bash
#SBATCH --job-name=cryolo_general
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/cryolo/general/slurm-%j.out

# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/cryolo

# configure run
cryolo_gui.py config ${DATASET_HOME}/relion/cryolo/general/config_cryolo.json ${EMAN_BOXSIZE_PIX} --filter LOWPASS --low_pass_cutoff 0.1

# predict
cryolo_predict.py -c ${DATASET_HOME}/relion/cryolo/general/config_cryolo.json -w ${DATASET_HOME}/relion/cryolo/general/gmodel_phosnet_202005_N63_c17.h5 -i ${DATASET_HOME}/relion/test_img/ -g 0 -o ${DATASET_HOME}/relion/cryolo/general/ -t 0.3
END

# submit script
sbatch ${DATASET_HOME}/relion/cryolo/general/run_submit.script
```

Refine general model weights. **Note: this must be run on a GPU!**

```bash
# make output directories
mkdir -p ${DATASET_HOME}/relion/cryolo/refined

# write slurm batch job
cat << END > ${DATASET_HOME}/relion/cryolo/refined/train_submit.script
#!/bin/bash
#SBATCH --job-name=cryolo_train
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/cryolo/refined/slurm-%j.out

# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/cryolo

# configure train
cryolo_gui.py config ${DATASET_HOME}/relion/cryolo/refined/config_cryolo_train.json ${EMAN_BOXSIZE_PIX} --train_image_folder ${DATASET_HOME}/relion/train_img/ --train_annot_folder ${DATASET_HOME}/relion/train_annot/ --valid_image_folder ${DATASET_HOME}/relion/val_img/ --valid_annot_folder ${DATASET_HOME}/relion/val_annot/ --pretrained_weights ${DATASET_HOME}/relion/cryolo/general/gmodel_phosnet_202005_N63_c17.h5 --saved_weights_name ${DATASET_HOME}/relion/cryolo/refined/refined_weights.h5

# train
cryolo_train.py -c ${DATASET_HOME}/relion/cryolo/refined/config_cryolo_train.json -w 0 -g 0 --fine_tune -lft 2
END

# submit script
sbatch ${DATASET_HOME}/relion/cryolo/refined/train_submit.script
```

Predict with refined model. **Note: this must be run on a GPU!**

```bash
# write slurm batch job
cat << END > ${DATASET_HOME}/relion/cryolo/refined/run_submit.script
#!/bin/bash
#SBATCH --job-name=cryolo_refined
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/cryolo/refined/slurm-%j.out

# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/cryolo

# configure run
cryolo_gui.py config ${DATASET_HOME}/relion/cryolo/refined/config_cryolo_pred.json ${EMAN_BOXSIZE_PIX} --filter LOWPASS --low_pass_cutoff 0.1

# predict
cryolo_predict.py -c ${DATASET_HOME}/relion/cryolo/refined/config_cryolo_pred.json -w ${DATASET_HOME}/relion/cryolo/refined/refined_weights.h5 -i ${DATASET_HOME}/relion/test_img/ -g 0 -o ${DATASET_HOME}/relion/cryolo/refined/ -t 0.3
END

# submit script
sbatch ${DATASET_HOME}/relion/cryolo/refined/run_submit.script
```

Score general and refined models

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
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/cryolo/general/STAR/*.star ${DATASET_HOME}/relion/cryolo/general/BOX -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force

# score general model
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/cryolo/general/BOX/*.box &> ${DATASET_HOME}/relion/cryolo/general/particle_set_comp.txt

# convert refined particles to box
mkdir -p ${DATASET_HOME}/relion/cryolo/refined/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/cryolo/refined/STAR/*.star ${DATASET_HOME}/relion/cryolo/refined/BOX -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force

# score refined model
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/cryolo/refined/BOX/*.box &> ${DATASET_HOME}/relion/cryolo/refined/particle_set_comp.txt

# print scores
echo "general" $(tail -1 ${DATASET_HOME}/relion/cryolo/general/particle_set_comp.txt)
echo "refined" $(tail -1 ${DATASET_HOME}/relion/cryolo/refined/particle_set_comp.txt)
```

## Topaz

Predict test set with general model. **Note: this must be run on a GPU!**

```bash
# make output directories
mkdir -p ${DATASET_HOME}/relion/topaz/general/

# write slurm batch job
cat << END > ${DATASET_HOME}/relion/topaz/general/run_submit.script
#!/bin/bash
#SBATCH --job-name=topaz_general
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/topaz/general/slurm-%j.out

# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/topaz

# apply micrograph downsampling
cd ${DATASET_HOME}/relion/topaz/general
topaz preprocess -s ${TOPAZ_SCALE} -o ${DATASET_HOME}/relion/topaz/general/test_img_downsampled/ ${DATASET_HOME}/relion/test_img/*.mrc

# pick particles
topaz extract -r ${TOPAZ_PARTICLE_RAD} -x ${TOPAZ_SCALE} -o ${DATASET_HOME}/relion/topaz/general/predicted_particles_all_upsampled.txt ${DATASET_HOME}/relion/topaz/general/test_img_downsampled/*.mrc
END

# submit script
sbatch ${DATASET_HOME}/relion/topaz/general/run_submit.script
```

Train model from scratch. **Note: this must be run on a GPU!**

```bash
# make output directories
mkdir -p ${DATASET_HOME}/relion/topaz/refined/

# write slurm batch job
cat << END > ${DATASET_HOME}/relion/topaz/refined/train_submit.script
#!/bin/bash
#SBATCH --job-name=topaz_train
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/topaz/refined/slurm-%j.out

# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/topaz

# preprocess train micrographs
topaz preprocess -s ${TOPAZ_SCALE} -o ${DATASET_HOME}/relion/topaz/refined/train_img_downsampled/ ${DATASET_HOME}/relion/train_img/*.mrc

# convert train particles to topaz input format
mkdir -p ${DATASET_HOME}/relion/topaz/refined/train_annot/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/train_annot/*.box ${DATASET_HOME}/relion/topaz/refined/train_annot/ -f box -t box -b ${EMAN_BOXSIZE_PIX} --round 0 -s ""
topaz convert -s ${TOPAZ_SCALE} ${DATASET_HOME}/relion/topaz/refined/train_annot/*.box -o ${DATASET_HOME}/relion/topaz/refined/train_particles.txt

# preprocess validation micrographs
topaz preprocess -s ${TOPAZ_SCALE} -o ${DATASET_HOME}/relion/topaz/refined/val_img_downsampled/ ${DATASET_HOME}/relion/val_img/*.mrc

# convert validation particles to topaz input format
mkdir -p ${DATASET_HOME}/relion/topaz/refined/val_annot/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/val_annot/*.box ${DATASET_HOME}/relion/topaz/refined/val_annot/ -f box -t box -b ${EMAN_BOXSIZE_PIX} --round 0 -s ""
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
END

# submit script
sbatch ${DATASET_HOME}/relion/topaz/refined/train_submit.script
```

Predict test set with refined model. **Note: this must be run on a GPU!**

```bash
# write slurm batch job
cat << END > ${DATASET_HOME}/relion/topaz/refined/run_submit.script
#!/bin/bash
#SBATCH --job-name=topaz_refined
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/topaz/refined/slurm-%j.out

# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/topaz

# pick particles using model
topaz extract -r ${TOPAZ_PARTICLE_RAD} -x ${TOPAZ_SCALE} -m ${DATASET_HOME}/relion/topaz/refined/model_epoch10.sav -o ${DATASET_HOME}/relion/topaz/refined/predicted_particles_all_upsampled.txt ${DATASET_HOME}/relion/topaz/general/test_img_downsampled/*.mrc
END

# submit script
sbatch ${DATASET_HOME}/relion/topaz/refined/run_submit.script
```

Score general and refined models

```bash
# activate evaluation environment
conda deactivate
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/imppel/

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

Installation and patching

```bash
# get latest version from github
git clone https://github.com/jianlin-cheng/AutoCryoPicker.git ${PICKER_INSTALL_DIR}/AutoCryoPicker/

# apply our patch
cp YOUR/PATH/TO/patches/autocryopicker/AutoPicker_Final_Demo.m ${PICKER_INSTALL_DIR}/AutoCryoPicker//Signle\ Particle\ Detection_Demo/AutoPicker_Final_Demo.m
```

Convert micrographs to PNG images

```bash
python ${UTIL_SCRIPT_DIR}/mrc_to_img.py ${DATASET_HOME}/relion/test_img/*.mrc -f png -o ${DATASET_HOME}/pngs
```

Pick particles (general model only). **Note: this must be run on a GPU!**

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
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/autocryopicker/slurm-%j.out

# load matlab
module load MATLAB/2020b
sleep 5s

cd ${PICKER_INSTALL_DIR}/AutoCryoPicker/Signle\ Particle\ Detection_Demo/
out_dir=${DATASET_HOME}/relion/autocryopicker/BOX/

# process each png one by one
for f in ${DATASET_HOME}/pngs/*.png; do
    out_name=\$(basename \$f)
    label_file="\${out_dir}/\${out_name%.png}.box"

    # run picker
    matlab -nosplash -nodisplay -r "mrc='\$f';out_dir='\$out_dir';AutoPicker_Final_Demo" -logfile "\$label_file"

    # convert stdout to box file
    awk '/AUTOCRYOPICKER_DETECTIONS_START/ ? c++ : c' \${label_file} > \${label_file/.box/.tmp} && mv \${label_file/.box/.tmp} \${label_file}
done
END

# submit the script
sbatch ${DATASET_HOME}/relion/autocryopicker/run_submit.script
```

Score model

```bash
# remove temporary files generated by AutoCryoPicker
rm {DATASET_HOME}/relion/autocryopicker/BOX/*.png

# activate evaluation environment
conda deactivate
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/imppel/

# calculate score
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/autocryopicker/BOX/*.box &> ${DATASET_HOME}/relion/autocryopicker/particle_set_comp.txt

# print score
echo "general" $(tail -1 ${DATASET_HOME}/relion/autocryopicker/general/particle_set_comp.txt)
```

## ASPIRE APPLE-picker

Automated picking (general model only). **Note: this must be run on a GPU!**

```bash
# make output directory
mkdir -p ${DATASET_HOME}/relion/aspire/STAR/

# rescale micrographs
mkdir -p ${DATASET_HOME}/relion/aspire/test_img_downsampled/
for input in ${DATASET_HOME}/relion/test_img/*.mrc; do
    basename=${input##*/}
    singularity exec /gpfs/ysm/datasets/cryoem/eman2.3_ubuntu18.04.sif e2proc2d.py --meanshrink=${ASPIRE_SCALE} ${input} ${DATASET_HOME}/relion/aspire/test_img_downsampled/${basename}
done

# write batch script to process all micrographs
cat << END > ${DATASET_HOME}/relion/aspire/run_submit.script
#!/bin/bash
#SBATCH --job-name=aspire
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/aspire/slurm-%j.out

# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/aspire

# this script is a wrapper for APPLE picker and bypasses the ASPIRE config system
python ${UTIL_SCRIPT_DIR}/../pickers/aspire/apple_cli.py ${DATASET_HOME}/relion/aspire/test_img_downsampled/*.mrc -o ${DATASET_HOME}/relion/aspire/STAR/ --particle_size ${ASPIRE_BOXSIZE_PIX} --max_particle_size $((ASPIRE_BOXSIZE_PIX * 2)) --min_particle_size $((ASPIRE_BOXSIZE_PIX / 4)) --minimum_overlap_amount $((ASPIRE_BOXSIZE_PIX / 10)) --query_image_size ${QUERY_IMAGE_SIZE} --tau1 ${TAU1} --tau2 ${TAU2} --container_size ${CONTAINER}
END

# run batch script
sbatch ${DATASET_HOME}/relion/aspire/run_submit.script
```

Postprocess particles

```bash
# convert from STAR to BOX format
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/aspire/STAR/*.star ${DATASET_HOME}/relion/aspire/tmp/ -f star -t box -b ${ASPIRE_BOXSIZE_PIX} --force --round 0

# upsample particle coordinates using topaz's scaling utility
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/topaz
mkdir -p ${DATASET_HOME}/relion/aspire/BOX/
for input in ${DATASET_HOME}/relion/aspire/tmp/*.box; do
    topaz convert ${input} -o ${DATASET_HOME}/relion/aspire/BOX/ --up-scale ${ASPIRE_SCALE} --to box --boxsize ${EMAN_BOXSIZE_PIX}
done

# remove unscaled boxfiles
rm -rf ${DATASET_HOME}/relion/aspire/tmp/
```

Score model

```bash
# activate evaluation environment
conda deactivate
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/imppel/

# compare particle picks against ground ground truth
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/aspire/BOX/*.box &> ${DATASET_HOME}/relion/aspire/particle_set_comp.txt
```

## DeepPicker

Installation and patching

```bash
# get latest version from github
git clone https://github.com/nejyeah/DeepPicker-python.git ${PICKER_INSTALL_DIR}/deeppicker

# apply our patch
cp YOUR/PATH/TO/patches/deeppicker/*.py ${PICKER_INSTALL_DIR}/deeppicker
```

Pick particles with general model. **Note: this must be run on a GPU!**

```bash
# make output directories
mkdir -p ${DATASET_HOME}/relion/deeppicker/general/STAR/

# write slurm batch job
cat << END > ${DATASET_HOME}/relion/deeppicker/general/run_submit.script
#!/bin/bash
#SBATCH --job-name=deeppicker_general
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/deeppicker/general/slurm-%j.out

# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/deeppicker

# pick with general model
python ${PICKER_INSTALL_DIR}/deeppicker/autoPick.py --inputDir ${DATASET_HOME}/relion/test_img/ --pre_trained_model ${PICKER_INSTALL_DIR}/deeppicker/trained_model/model_demo_type3 --particle_size ${EMAN_BOXSIZE_PIX} --mrc_number -1 --outputDir ${DATASET_HOME}/relion/deeppicker/general/STAR/ --coordinate_symbol _deeppicker --threshold 0.5
END

# run batch script
sbatch ${DATASET_HOME}/relion/deeppicker/general/run_submit.script
```

Train model from scratch. **Note: this must be run on a GPU!**

```bash
# make output directories
mkdir -p ${DATASET_HOME}/relion/deeppicker/train/
mkdir -p ${DATASET_HOME}/relion/deeppicker/val/
mkdir -p ${DATASET_HOME}/relion/deeppicker/refined/STAR/

# write slurm batch job
cat << END > ${DATASET_HOME}/relion/deeppicker/refined/train_submit.script
#!/bin/bash
#SBATCH --job-name=deeppicker_train
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/deeppicker/refined/slurm-%j.out

# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/deeppicker

# collect training data
cp -s ${DATASET_HOME}/relion/train_img/*.mrc ${DATASET_HOME}/relion/deeppicker/train/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/train_annot/*.box ${DATASET_HOME}/relion/deeppicker/train/ -f box -t star -b ${EMAN_BOXSIZE_PIX} --header --force

# collect validation data (eval in DeepPicker terminology)
cp -s ${DATASET_HOME}/relion/val_img/*.mrc ${DATASET_HOME}/relion/deeppicker/val/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/val_annot/*.box ${DATASET_HOME}/relion/deeppicker/val/ -f box -t star -b ${EMAN_BOXSIZE_PIX} --header --force

# train model from scratch
python ${PICKER_INSTALL_DIR}/deeppicker/train.py --train_type 1 --train_inputDir ${DATASET_HOME}/relion/deeppicker/train/ --particle_size ${EMAN_BOXSIZE_PIX} --coordinate_symbol "" --model_retrain --model_load_file ${PICKER_INSTALL_DIR}/deeppicker/trained_model/model_demo_type3 --model_save_dir ${DATASET_HOME}/relion/deeppicker/refined/ --model_save_file model_demo_type3_refined
END

# run batch script
sbatch ${DATASET_HOME}/relion/deeppicker/refined/train_submit.script
```

Pick particles using trained model. **Note: this must be run on a GPU!**

_Note that the output of this picking job may contain an error to the effect that the given model file cannot be found. This is expected, as the error comes from a direct file path check, while the tensorflow saver module both writes and reads back files with various extensions appended to the model name._

```bash
# write slurm batch job
cat << END > ${DATASET_HOME}/relion/deeppicker/refined/run_submit.script
#!/bin/bash
#SBATCH --job-name=deeppicker_refined
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/deeppicker/refined/slurm-%j.out

source ${CONDA_ACTIVATE} ${CONDA_ENVS}/deeppicker

python ${PICKER_INSTALL_DIR}/deeppicker/autoPick.py --inputDir ${DATASET_HOME}/relion/test_img/ --pre_trained_model ${DATASET_HOME}/relion/deeppicker/refined/model_demo_type3_refined --particle_size ${EMAN_BOXSIZE_PIX} --mrc_number -1 --outputDir ${DATASET_HOME}/relion/deeppicker/refined/STAR/ --coordinate_symbol _deeppicker --threshold 0.5
END

# submit script
sbatch ${DATASET_HOME}/relion/deeppicker/refined/run_submit.script
```

Score model

```bash
# convert general particles to box
mkdir -p ${DATASET_HOME}/relion/deeppicker/general/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/deeppicker/general/STAR/*.star ${DATASET_HOME}/relion/deeppicker/general/BOX/ -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force

# score general model
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/deeppicker/general/BOX/*.box &> ${DATASET_HOME}/relion/deeppicker/general/particle_set_comp.txt

# convert refined particles to box
mkdir -p ${DATASET_HOME}/relion/deeppicker/refined/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/deeppicker/refined/STAR/*.star ${DATASET_HOME}/relion/deeppicker/refined/BOX/ -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force

# score refined model
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/deeppicker/refined/BOX/*.box &> ${DATASET_HOME}/relion/deeppicker/refined/particle_set_comp.txt

# print scores
echo "general" $(tail -1 ${DATASET_HOME}/relion/deeppicker/general/particle_set_comp.txt)
echo "refined" $(tail -1 ${DATASET_HOME}/relion/deeppicker/refined/particle_set_comp.txt)

# select one micrograph to visualize and grab its file stem
mkdir -p ${DATASET_HOME}/relion/deeppicker/vis_cmp/
vis_mrc=$(basename -- $(ls ${DATASET_HOME}/relion/test_img/ | head -1))
vis_mrc=${vis_mrc%.*}

# visual comparison
python ${UTIL_SCRIPT_DIR}/plot_boxfile.py -m ${DATASET_HOME}/relion/test_img/${vis_mrc}*.mrc -g ${DATASET_HOME}/relion/test_annot/${vis_mrc}*.box -p ${DATASET_HOME}/relion/deeppicker/general/BOX/${vis_mrc}*.box ${DATASET_HOME}/relion/deeppicker/refined/BOX/${vis_mrc}*.box --num_gt 32 -o ${DATASET_HOME}/relion/deeppicker/vis_cmp/ --force
```

## PARSED

Installation and patching

```bash
# note: the original software is a static .zip in the paper's supplementary
#       materials, and is available in this repo at pickers/parsed/original/
# note: PARSED manual is available at pickers/parsed/PARSED_Manual_V1.pdf

# copy our patched version of PARSED to your install directory
mkdir -p ${PICKER_INSTALL_DIR}/PARSED
cp YOUR/PATH/TO/pickers/parsed/*.{py,h5} ${PICKER_INSTALL_DIR}/PARSED

# create conda environment
# note: opencv2 needs to be v3.4.3
#       (https://github.com/facebookresearch/maskrcnn-benchmark/issues/339)
conda create -n parsed -c conda-forge python=3.6 h5py=2.7.1 keras=2.0.8 numba=0.37.0 pandas=0.20.3 matplotlib=2.1.0 mrcfile=1.1.2 trackpy=0.4.1 tensorflow-gpu opencv=3.4.3 tqdm
conda activate parsed
```

Pick particles with PARSED (general model only). **Note: this must be run on a GPU!**

```bash
# make output directory
mkdir -p ${DATASET_HOME}/relion/parsed/STAR/

# write batch script to pick particles with general model
# don't use blob thresholding (i.e., no data specific optimizations)
cat << END > ${DATASET_HOME}/relion/parsed/run_submit.script
#!/bin/bash
#SBATCH --job-name=parsed
#SBATCH -N1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=5:00:00
#SBATCH -p pi_gerstein_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=${DATASET_HOME}/relion/parsed/slurm-%j.out

module load CUDA/8.0.61
sleep 5s
module load cuDNN/8.0.5.39-CUDA-11.1.1
sleep 5s
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/parsed
sleep 5s

cd ${PICKER_INSTALL_DIR}/PARSED
python -W ignore parsed_main.py --model=pre_train_model.h5 --data_path=${DATASET_HOME}/relion/test_img/ --output_path=${DATASET_HOME}/relion/parsed/STAR/ --file_pattern=*.mrc --job_suffix=parsed --angpixel=${ANG_PIX_RES} --img_size=${MRC_LONGSIDE_PIX} --edge_cut=0  --core_num=4 --aperture=${EMAN_BOXSIZE_A}
END

# submit script
sbatch ${DATASET_HOME}/relion/parsed/run_submit.script
```

Plot distribution of detected particles

```bash
# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/parsed/

# write distribution image
cd ${PICKER_INSTALL_DIR}/PARSED
python particle_mass.py drawmass --pick_output=${DATASET_HOME}/relion/parsed/STAR/ --job_suffix=parsed --tmp_hist=${DATASET_HOME}/relion/parsed/hist.png

# determine threshold
#   1) open the image at ${DATASET_HOME}/relion/parsed/hist.png
#   2) if disitrbution is unimodal, use no thresholding
#   3) if distribution is multimodal, assign a threshold to PARSED_THRES
#      that will separate the two peaks
PARSED_THRES=
python particle_mass.py cutoff --pick_output=${DATASET_HOME}/relion/parsed/STAR/ --job_suffix=parsed --output_suffix=parsed_thres --thres=$PARSED_THRES
```

Score model

```bash
# activate environment
source ${CONDA_ACTIVATE} ${CONDA_ENVS}/imppel/

# convert general predictions to box
mkdir -p ${DATASET_HOME}/relion/parsed/BOX/
python ${UTIL_SCRIPT_DIR}/coord_converter.py ${DATASET_HOME}/relion/parsed/STAR/*_parsed.star ${DATASET_HOME}/relion/parsed/BOX/ -f star -t box -b ${EMAN_BOXSIZE_PIX} --header --force
rm -rf ${DATASET_HOME}/relion/parsed/STAR/

# score general model
python ${UTIL_SCRIPT_DIR}/score_detections.py -g ${DATASET_HOME}/relion/test_annot/*.box -p ${DATASET_HOME}/relion/parsed/BOX/*.box &> ${DATASET_HOME}/relion/parsed/particle_set_comp.txt

# print scores
echo "general" $(tail -1 ${DATASET_HOME}/relion/parsed/general/particle_set_comp.txt)
```

## RELION LoG Picker

TODO

## CASSPER

TODO
