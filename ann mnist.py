import numpy as np
import csv,neurogen
import matplotlib.pyplot as plt

def convlabel(val):
	out = np.zeros((1,10))
	out[0][int(val)]=1
	return out[0]

num,lab=[],[]
with open("source/mnist_train.csv","r") as f:
	reader = csv.reader(f)
	c=1
	for i in reader:
		lab.append(convlabel(i[0]))
		num.append(neurogen.normalize(np.asfarray(i[1:]),255))
		c+=1
		if c>150:
			break

lab=np.array(lab)
num=np.array(num)
nn = neurogen.MLP([784,200,10],0.1)
neurogen.train(nn,5000,num[:100],lab[:100],False)

def random_check(i):
	t=num[i]
	print(np.argmax(nn.predict(t)))
	plt.imshow(t.reshape(28,28),cmap="Greys")
	plt.show()

random_check(106)
random_check(111)
random_check(143)
random_check(119)