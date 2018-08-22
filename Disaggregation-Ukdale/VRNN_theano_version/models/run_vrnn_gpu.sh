#!/bin/bash
echo "starting gpu"
#cd /data/home/gbejara1/Research/Disaggregation/other_models/LatentVariable
export PYTHONPATH=$HOME/Research/Disaggregation/UKDALE/nilmtk:$HOME/Research/Disaggregation/UKDALE/nilm_metadata:$HOME/Research/Disaggregation/UKDALE
source activate latentVar
export CUDA_VISIBLE_DEVICES=0,1
module load cuda80/toolkit/8.0.61
echo $PATH
echo $PYTHONPATH
THEANO_FLAGS="gcc.cxxflags='-march=core2'" python vrnn_gmm_aggVS1_pureGeneration.py
echo "ending"

#echo $PATH
#PYTHONPATH=$HOME/Research/Disaggregation/other_models/LatentVariable python vrnn_gmm_aggVS1.py
