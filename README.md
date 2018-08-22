VRNN-DIS1 and VRNN-DIS-ALL models are constructed based on the VRNN model implemented in theano by its authors here: https://github.com/jych/nips2015_vrnn. This model uses some other modules like:
1) Load and processing for VRNN-DIS-1 model make uses of neuralnilm and nilmtk modules that we already modified and embedded in this repository (originals in https://github.com/JackKelly/neuralnilm, https://github.com/nilmtk/nilmtk and https://github.com/nilmtk/nilm_metadata)
2) Training and monitoring modified for our VRNN-DIS1 and VRNN-DIS-ALL models in https://github.com/gissemari/cle (original modules in https://github.com/jych/cle)
3) One independent repository for each data set that se use (UK-DALE, Dataport, REDD).

#### FIRST TIME, CREATING SUPERPROJECT ###
git submodule add  http ...
git submodule add  http ...
git commit -am "creating submodules"
git push


#### CLONE SUPERPROECT
You can clone this projects in two ways (https://git-scm.com/book/en/v2/Git-Tools-Submodules):
1) Clone and then init each submodule:
	git clone https://gissemari@bitbucket.org/mlrg/bejarano-disaggregation.git
	cd subModuleName/
	git submodule init
2) Clone with the command: git clone --recurse-submodules https://gissemari@bitbucket.org/mlrg/bejarano-disaggregation.git

#### UPDATE (PULL)
Use this command to log in and update your cloned project and each submodule
	# pushing from other place and 'pulling' from the super project
	git submodule update --remote

#### PUSH CHANGES
1)  You need to go into each submodule and check out a branch to work on
	git checkout master (no origin)
	git add ..
	git commit -am '...'
	git push
2) Update from the super project
	cd ..
	git submodule update --remote --rebase
3) you can ask Git to check that all your submodules have been pushed properly before pushing the main project
4) git submodule update
5) In super project: git add subModuleName, git commit -am '...', git push
