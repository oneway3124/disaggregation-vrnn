############# VERSIONS details #####################
- vrnn_gauus is similar to original with the difference that instead of sampling two dim + bernoulli, we sample all from gaussian (mixture)

- vrnn_gmm_aggVSdisagg adds, to the cost function, the similarity or mse between the real disaggregated signal versus the predicted one

- vrnn_gmm_aggVSdisagg_distrib adds the autoencoding to the disaggregation instead of the original. So we create different theta_mu for each appliance, and we wanted to do that in an array or sth like that but we could not. We had errors related to scan and objects that could not by passed by the outputs. So we are creating different version for each number of appliance

- vrnn_gmm_aggVSdisagg_distrib_eachApp is the same but instead of calling GMM with x_in, it calls it with y_in and it is working. All from the same theta_mu

- vrnn_gmm_aggVSdisagg_xy is similar to vrnn_gmm_aggVSdisagg but instead of adding mse of just the separated output of the rnn, it compares the reconstruction with the sum of the rnn output and applies the mse there

- vrnn_gmm_pureGeneration there exist three cases here. The first one where we reconstruct disaggregated appliance consumption from the aggregated one (disaggregation problem: vrnn_gmm_aggVSdisagg_distrib5). The second where we reconstruct in the training and generate in the validation (generation problem). The third is where we train by generating next step and validate as well generating and carrying the error because any step is seen. The las two works with the parameter: genCase (when case 1 it used to be vrnn_...pureGeneration.py). So better used -> vrnn_gmm_generation is

- vrnn_AE_prior is a model that does not exactly work with latent variable, so we should not actually call it vrnn because at the training and testing time it does not use the prior learned.

- vrnn_AE_priorXY is a model where we do use the prior learned in the testing (generation-sampling) time

- vrnn_AE, vrnn_gmm_disVSdis are the respective models to vrnn_AE_priorXY and vrnn_gmm_generation because they do the same in the training but instead of switching the testing as a generation, they apply the same process to the validation in order to tweak the parameters. Later we would have to construct other parts of code to just generate based on the best parameteres found.
- vrnn_AE-fixed is taking into account x,yfor calculating h_t. In training we see real t, in testing we get the generated y. Also, load_vrnn_dis_dis is a version that does the final true generation without seeing any input at the end of the program.

- vrnn_gmm_aggVS1 is like vrnn_gmm_disVSdis but matching aggregated vesus disaggregted instead of training for absolute generation of individual appliances. 

- vrnn_dis-all version of vrnn for all at once. It uses different p(y) distributions/layers for each appliance (from a same theta).
- vrnn_dis-all-difY allow to select different set of appliances
 


############# SET PARAMTERS AND VARIABLES  #########################################
Parameters are set in several places:
- config.txt: some of the paramters that not that obvious are
	- monitoring_freq: number of training batch after testing in validation set. It depends on the size of the training set
	- flgAgg: -1 sample original x, 0: first appliance - 1 second appliance ... 4: 5th appliance
	- genCase: generates from reconstruction for same time step (0) or generation for next step (1)
	- typeLoad: 0 for original load where the total set is divided in training, validation and testing. 1 Kelly's load (for the case of UKDALE). 2: for a similar load to Kelly's but with our own OFF values for filtering
- Hardcode: buildings, APPLIANCES (we call it by specifying the number of appliances we want at loadCSVdataVRNN
), windows of time and instances to plot


Set PYTHONPATH with the path to the libraries:
Cle found here https://github.com/jych/cle
- nilmtk: 
	/home/gissella/Documents/Research/Disaggregation/UK-DALE/nilmtk:
	/home/gissella/Documents/Research/Disaggregation/UK-DALE/nilm_metadata
- cle.cle: (not neccesary if cle is at the same level of the called python file)
	/home/gissella/Documents/Research/Disaggregation/UK-DALE/VRNN_theano/models
- dataSet
	/home/gissella/Documents/Research/Disaggregation/UK-DALE/VRNN_theano/datasets

export PYTHONPATH=$PYTHONPATH:~/Documents/Research/Disaggregation/UK-DALE:~/Documents/Research/Disaggregation/UK-DALE/nilmtk:~/Documents/Research/Disaggregation/UK-DALE/neuralnilm:~/Documents/Research/Disaggregation/UK-DALE/nilm_metadata:~/Documents/Research/Disaggregation/UK-DALE/preprocessing