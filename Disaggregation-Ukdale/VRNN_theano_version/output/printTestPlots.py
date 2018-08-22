import pickle

try:
	outputGeneration = pickle.load(open("outputRealGeneration.pkl","rb"))
 except (OSError, IOError) as e:
 	print("problem with pickle")
 else:
 	size = len(np.transpose(outputGeneration[1],[1,0,2]))
 	for i in range(size):
	 	plt.figure(1)
	    plt.plot(np.transpose(outputGeneration[1],[1,0,2])[i])
