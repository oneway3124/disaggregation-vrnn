#!/bin/bash
echo "starting own vrnn"
export PYTHONPATH=$HOME/Research/Disaggregation/UKDALE:$HOME/Research/Disaggregation/UKDALE/nilmtk:$HOME/Research/Disaggregation/UKDALE/nilm_metadata
source activate latentVar
THEANO_FLAGS="gcc.cxxflags='-march=core2'" python vrnn_gmm_generation.py
echo "ending"
