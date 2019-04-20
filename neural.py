import numpy as np 

#Layer class
#Implemented to use Bias
#Activation Function is sigmoid
class layer():
	def __init__(self,in_n,out_n,bias=True):
		self.w = np.random.randn(in_n,out_n)
		self.bias = bias
		if bias:
			self.b = np.random.randn(1,out_n)
		else:
			self.b = np.zeros([1,out_n])
	def act_fun(self,x,deriv=False):
		if deriv:
			return x*(1 - x)
		return 1.0 / (1 + np.exp(-x))
	def forward(self,in_data):
		self.i = in_data
		if self.bias:
			self.z = in_data.dot(self.w) + np.ones([len(in_data),1]).dot(self.b)
		else:
			self.z = in_data.dot(self.w)
		self.o = self.act_fun(self.z)
	def backward(self,out_err):
		self.d = out_err*self.act_fun(self.o,deriv=True)
		self.delta = self.d.dot(self.w.T)
		self.w = self.w - self.i.T.dot(self.d)	
		if self.bias:
			self.b = self.b - np.ones([1,len(self.i)]).dot(out_err)
	def save(self,name):
		np.savetxt(name+"-w.dat",self.w)
		if self.bias:
			np.savetxt(name+"-b.dat",self.b)
	def load(self,name):
		self.w = np.loadtxt(name+"-w.dat")
		if self.bias:
			self.b = np.loadtxt(name+"-b.dat") 
class neural_net():
	def __init__(self,data):
		self.l = []
		for i in data:
			self.l.append(layer(*i))
	def forward(self,in_data):
		out = in_data
		for i in self.l:
			i.forward(out)
			out = i.o
		return self.l[-1].o
	def train(self,in_set,out_set,N):
		for i in range(N):
			o = self.forward(in_set)
			err = (o-out_set)
			for j in range(len(self.l)):
				self.l[len(self.l)-j-1].backward(err)
				err = self.l[len(self.l)-j-1].delta
	def save_weights(self):
		for i in range(len(self.l)):
			self.l[i].save(str(i))
	def load_weights(self):
		for i in range(len(self.l)):
			self.l[i].load(str(i))
