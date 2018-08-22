###############  VERSIONS ############

- dataSet_v0.py is a version a little messy of the loading dataSet function
- dataSet.py is an improved and more organized (done Jan19 - IJCAI). Without the need of new function in NILMTK (getAll but a direct read of powerSeries). Both groups of models: autoEncoders, lstm and VRNNs call this file functions to load the data set.
- autoEncoder_tf.py make use of the dataSet.py to load the data (train, val, test). And computes the general cost without taking into account the different appliances.
- autoEncoder_tf_costSeparated.py Transposes the yPred and yReal in order to get mse by each appliance in mse function
- autoEncoder_tf_App.py receives one more parameter to do the autoencoding just against one appliance
- VRNN_theano/models/* contains all the vrnn models based on the paper: Recurrent Latent Variable Model for Sequential Data (Junyoung Chung, Kyle Kastner ...)


###############   SOME PRE KNOWLEDGE ABOUT THE MODEL   ###############

- It runs with python2.7
- It is strongly recommended to work on a conda environment and install all the neccesary libraries there to be able to run with the same settings in spiedie or any other server.
- It's assumed that you can make a reference to the Tensorflow, NEURALNILM (nilmtk/nilmtk/electric.py: getAllValues()) and NILMTK libraries. Now: The generator powerSeriesAll() is used in order to get the pd.Series with the signal. To get all the data directly instead of the objects, dictionaries and rest of structures that nilmtk works with. Example:

	export PYTHONPATH=/home/gissella/Documents/Research/Disaggregation/UK-DALE/neuralnilm:/home/gissella/Documents/Research/Disaggregation/UK-DALE/nilmtk:/home/gissella/Documents/Research/Disaggregation/UK-DALE/nilm_metadata

- Make sure python paths are set to the proper nilmtk,neuralnilm, and nilm_metadata directories (see this graphic for a better understanding of nilm_metadata structure: https://github.com/nilmtk/nilm_metadata/blob/master/docs/source/tutorial.rst)
Unless you are running pandas version 0.19.2, you will need to update timeframegroup.py on line 15 to pd.PeriodIndex. Better switch to 0.19.2.


##############        RUNNING AND PARAMTERS   ####################

Replicate models of Kelly and our own autoEncoder work with config.txt file and config_kelly.txt. The meaning of the parameters are:
- nilmkt_fileName: the path of the input data
- n_steps (sequenceLength) ukdale input file is a unique large sequence of values. To build a batch, we generate several sequences of this length to become the training or test set.
- hiddenUnitsLayer1: the number of units in the first hidden layer of the lstm
- iterations (numEpochs): number of epochs to consider in the training
- strideTrain: the larger the strideTraining the less number of sequences in the training Set. If this is smaller than the sequenceLength, the sequences for the 
training are overlapped.
- strideTest: equal to the sequenceLength and to get just one prediction per timeStep (from one sequence). If we test with overlapped sequences (not implemented yet), we will have to average the output for certain points as they do it in Kelly's paper.
- applianceToTest: this autoEncoder's output is the consumption of one appliance (disaggregation)
numSequences 31849
- loadAsKelly (y/n): sample data like kelly does? (not anymore: typeLoad)
- typeLoad
	1: it respects the same sampling from the ranges of train, test and validation that Kelly did
	2:  it samples from all over the training range (it does not use test and validation range) without overalapping
- target_inclusion_prob: if sampling like kelly, at what probability do you want to have active sequences?
VRNN models work with their own config.txt file explained in the folder VRNN_theano/models and also have their own README and utils files.
Note that the basic auto encoder and Kelly auto encoder use different config files. The basic one uses config.txt, while Kelly's uses config_kelly.txt
############       DETAILED EXPLANATION OF THE CODE   ###############

- We work with 3 options for loading
	- Unlike Kelly's work, we decided to not split the training, validation sets by windows of time but to have a unique window where we split by percentage the instances obtained for training and validation sets.
	- Like Kelly's load
	- Like Kelly's load but sampling for the whole range without limit between training and testing. However, overlapping is not allowed, This is guarantee with the stride


############    UTILS - ERRORS SOLUTIONS

FOR INSTALLATION

When installing theano with conda, missing to install python2x. So just copy it from somewhere

Install pandas=0.19.2 Because that is the one with Period in module pd.TimeSeries (or later)
conda install pandas=0.19.2 (otherwise also problem with pd.tseries.period.PeriodIndex)
No module yaml, networkx, builtins:
pip install pyyaml
pip install networkx
pip install future #for builtins
pip install --upgrade IPython
pip install matplotlib
pip install tables # for  HDFStore requires PyTables

conda install scikit-learn
conda install -c anaconda mkl-service
conda install -c anaconda lxml


FOR RUNNING IN SPIEDIE:
export PYTHONPATH=/home1/gbejara1/Documents/Research/Disaggregation/UKDALE/nilmtk:/home1/gbejara1/Documents/Research/Disaggregation/UKDALE/neuralnilm:/home1/gbejara1/Documents/Research/Disaggregation/UKDALE/nilm_metadata
