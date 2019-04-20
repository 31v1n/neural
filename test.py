import numpy as np
import neural as nn
#Sample with AND logic gate:
N = 500
s = [[2,3,False],[3,3,False],[3,1,False]]
in_set = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
out_set = np.array([[1.],[0.],[0.],[1.]])

n = nn.neural_net(s)
n.train(in_set,out_set,N)
print(n.forward(in_set))
