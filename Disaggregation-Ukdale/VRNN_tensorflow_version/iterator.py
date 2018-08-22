import numpy as np

class Iterator:
	
	def __init__(self,Xdata, Ydata, start=0,end=None,batch_size=0,nbatches=0,n_steps=1,shape_diff=False):
		self.Xdata = Xdata
		self.Ydata = Ydata
		self.start = start
		self.batch_size = batch_size
		self.n_steps = n_steps
		if not shape_diff:
			assert np.shape(self.Xdata) == np.shape(self.Ydata)
		if end is None:
			total_inst = np.shape(self.Xdata[start:])[0]
		else:
			total_inst = np.shape(self.Xdata[start:end+1])[0]
		batches = total_inst // batch_size
		if nbatches != 0:
			if nbatches >= batches:
				self.nbatches = batches
			elif nbatches < batches:
				self.nbatches = nbatches
		
		else:
			self.nbatches = batches

	def get_split(self):
		idx = self.start
		x = []
		y = []
		for i in range(self.nbatches):
			x.append(np.array(self.Xdata[idx : idx + self.batch_size]))
			y.append(np.array(self.Ydata[idx : idx + self.batch_size]))
			idx = idx + self.batch_size
		return (x,y)

	def get_target(self):
		idx = self.start
		for i in range(self.nbatches):
			y = np.array(self.Ydata[idx : idx + self.batch_size])
			idx = idx + self.batch_size
			yield y
					
			
