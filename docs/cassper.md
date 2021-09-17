#!/usr/bin/env bash
#
#	CASSPER install README
#	Author: Christopher JF Cameron, David Peng
#

##
#	Conda environment set up
##

#	clone Git
#cd /gpfs/slayman/pi/gerstein/cjc255/tools
#git clone https://github.com/airis4d/CASSPER.git

#	load GPU node
srun --x11 -p pi_gerstein_gpu --gres gpu:1 --time=2:00:00 --pty bash

#	create/active Conda environment
# conda create --name cassper python=3.6
conda activate cassper
# conda install -c anaconda pip joblib		# add. joblib dependency, not in requirements.txt
# conda update --all
# conda clean --all
# pip install -r /gpfs/slayman/pi/gerstein/cjc255/tools/CASSPER/requirements.txt
# conda env remove --name cassper
module load cuDNN/6.0-CUDA-8.0.61
module load CUDA/8.0.61

##
#	Pre-trained model --- INCOMPLETE (test 1)
##

#	1) download model weights and predicted labels from: https://drive.google.com/drive/folders/1bMxple_U1Q0nwL2RXJFycVrZ4rdji5oj
#	2) place folders in /gpfs/slayman/pi/gerstein/cjc255/tools/CASSPER/Train_and_Predict/Protein1/

#	predict coordinates
#	note - GUI closes on it's own after choosing a radius and moving cursor off window (slow to process images)
mv /gpfs/slayman/pi/gerstein/cjc255/tools/CASSPER/Train_and_Predict/Protein1/labels /gpfs/slayman/pi/gerstein/cjc255/tools/CASSPER/Train_and_Predict/Protein1/Predict_labels
python /gpfs/slayman/pi/gerstein/cjc255/tools/CASSPER/Train_and_Predict/extract_coordinates_from_labels.py --input /gpfs/slayman/pi/gerstein/cjc255/tools/CASSPER/Train_and_Predict/Protein1/Predict_labels --output /gpfs/slayman/pi/gerstein/cjc255/tools/CASSPER/Train_and_Predict/Protein1/star_coordinates

##
#	Train model from scratch --- INCOMPLETE (test 2)
##

#	comment out line 5: import CASS_scale as sw
#	non-harmful tensforflow warning thrown - ignore
python /gpfs/slayman/pi/gerstein/cjc255/tools/CASSPER/Train_and_Predict/predict_coordinates.py -h

	3) Predict labels on new micrographs (using pretrained or already trained model)

go to Train_and_Predict

pwd: .../Train_and_Predict

put mrc files into mrc_files. Make sure mrc files are not movies.

edit predict_labels.sh and comment out the 2nd line (source ...) since we already loaded requirements

run `./predict_labels.sh Protein1/`. Make sure to include / at the end

Labels are now saved in Protein1/Predict_labels, and you can run step 1) to extract star coords.


