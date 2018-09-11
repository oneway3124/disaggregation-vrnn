VRNN-DIS1 and VRNN-DIS-ALL 

##############  SOME PRE KNOWLEDGE ABOUT THE MODEL   ###############
- These models are constructed based on the VRNN model implemented in theano by its authors here: https://github.com/jych/nips2015_vrnn. 
- It runs with python2.7 and the rest of the libraries' verions can be checked in the .yml file
- Create and activate the environment from latentVar.yml

############# STRUCTURE OF THIS REPOSITORY   ##################

- auxiliary-libraries: where the previous mentioned repositories are saved
- Disaggregation-[dataset]: they are similarly arranged, a folder for the metadata, a folder where the original and preprocessed dataset files are saved, the code and the output.
- latentVar.yml: conda environment with all the modules and libraries installed

########## AUXILIARY LIBRARIES ##############
This model uses some other modules tha have been modified and included in this repository:

1) NEURALNILM: Load the .h5 file with the NIMLTK module and then process the UKDALE dataset in order to get instances with certain target (appliance) activation or without. Original repository in https://github.com/JackKelly/neuralnilm
2) NILMTK and NILM_METADATA: Loads the .h5 files, few changes are done based on the originals mostly for versioning problems. Originals https://github.com/nilmtk/nilmtk and https://github.com/nilmtk/nilm_metadata
2) CLE: Used for training and monitoring modified mostly to add variables to monitor and the scheduling sampling mechanisms. Original repository: https://github.com/jych/cle)

#############   SUB REPOSITORIES  ##################
Each of the dataset-repository is divided in 3 subfolders:

1) Metadata: statistics and figures to support some of the parameters that we selected such as the length of instances (sequences) the mininum value to consider the appliance is ON, the maximum value to consider is OFF, etc.
2) Datasets: folder where the original data should placed (.h5). It also contains a PICKLES subfolder to save some preprocessed data, specially for repeated experiments.
3) Preprocessing: where the logic of the preprocessing is stored. It's main function is to sample the instances based on the parameters in the config file such as sampling rate (period), sequence length, overlapping or not for training, etc. 
4) VRNN_theano_version: where the models and output for the VRNN models are stored. It also contains

########### PARAMETERS ##########

The parameters are defined inside the code files such as buildings, APPLIANCES (in the preprocessing file), windows (range of dates to load and/or ranges to separate for training and testing). However, most of the parameters are similar through all vrnn models and are read from a config file:

[PATHS]
- pickleModel is the file with the best saved model from which to read the parameters learned. Mostly used in the test files
- data_path is the path of data files: .../PecanStreet-dataport/datasets
- save_path is the folder where the output is going to be saved, usually .../VRNN_theano_version/output

[PREPROCESSING]
- period: the frequency rate at which the time steps are collected in the data
- n_steps: sequence length
- stride_train: stride or space between instances in the training set
	- 1: when typeLoad is 0 (because it will use 1 to have overlapped instances in the training, but will use n_steps for the stride in the testing instances)
	- n_steps: when the typeLoad is 1 (because as we randomly choose instances, we need to guarantee that the testing instances or any part of them are also part of the training set)
- stride_test: stride or space between instances in the testing set (it has to be equal or greater  than length of the sequence to avoid overlapping)
- typeLoad: 
	- 0 for original load where the total set is divided in training, validation and testing in an specific order. 
	- 2 Building instances randomly, without any overlap and assigning specific number to train, test and val. Unlike Kelly's work, we decided to not split the training, validation sets by windows of time but to have a unique window where we split by percentage the instances obtained for training and validation sets.
	- 1 (not used for this dataset. For ukdale: kelly's load)


[ALGORITHM]
- monitoring_freq: number of training batches after which perform a validation process. Say we have 100 mini-batches and the monitoring_freq is 20. The validation process would be done after only the process of the 10 mini-batches.
- epoch: number of iterations in which all the mini-batches were processed. It means, in one epoch, 100 mini-batches were processed.
- batch_size: size of the mini-batch
- kSchedSamp: speed of convergence for the inverse sigmoid decay. Taken from https://arxiv.org/abs/1506.03099
- lr: initial learning rate

[MODEL]
- x_dim: aggregated signal (1)
- y_dim: disaggregated signal (number of appliances)
- z_dim: units in z layer
- k_target_dim: (num_k) number of mixture gaussian distributions
- rnn_dim: number of units for the recurrent laye
- q_z_dim: size of unints for the feature extractor of the encoder distribution
- p_z_dim: size of unints for the feature extractor of the prior distribution
- p_x_dim: size of units for the decoder distribution
- x2s_dim: size of the feature extractor for the x signal
- y2s_dim: size of the feature extractor for the y (disaggregated) signals
- z2s_dim: size of the feature extractor for the z (latent variable)
- typeActivFunc: activation function for the mu (theta parameter) of the decoder distribution


############    UTILS - ERRORS SOLUTIONS ######

FOR INSTALLATION

- When installing theano with conda, missing to install python2x. So just copy it from somewhere


