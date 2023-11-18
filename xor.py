import neurogen
import numpy as np

i = np.array([[0,0],[0,1],[1,0],[1,1]])
t = np.array([[0],[1],[1],[0]])

nn=neurogen.MLP([2,3,1],0.1)
print(nn.predict(i))
neurogen.train(nn,5000,i,t)
print(nn.predict(i))