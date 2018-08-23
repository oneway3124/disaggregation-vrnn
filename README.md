VRNN-DIS1 and VRNN-DIS-ALL models are constructed based on the VRNN model implemented in theano by its authors here: https://github.com/jych/nips2015_vrnn. This model uses some other modules tha have been modified and included in this repository:
1) Load and processing for VRNN-DIS-1 model make uses of neuralnilm and nilmtk modules that we already modified and embedded in this repository (originals in https://github.com/JackKelly/neuralnilm, https://github.com/nilmtk/nilmtk and https://github.com/nilmtk/nilm_metadata)
2) Training and monitoring modified for our VRNN-DIS1 and VRNN-DIS-ALL models in https://github.com/gissemari/cle (original modules in https://github.com/jych/cle)
3) One independent repository for each data set that se use (UK-DALE, Dataport, REDD).

#### STRUCTURE OF THIS REPOSITORY

- auxiliary-libraries: where the previous mentioned repositories are saved
- Disaggregation-[dataset]: is a embedded copy of the previous individual repositories UK-DALE, Dataport and REDD. They are similarly arranged in the following folders:
	- metadata: where there are some statistics and figures to support some of the parameters that we selected
	- datasets: where the original data is stored. It also contains a PICKLES folder to save some preprocessed data, specially for repeated experiments
	- preprocessing: where the logic of the preprocessing
	- VRNN_theano_version: where the models and output for the VRNN models are stored

#### VRNN
Each of the dataset-repository have its own README file with more details about their config files and preprocessing logic.
