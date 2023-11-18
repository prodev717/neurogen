import numpy as np
import pickle,copy

def save_network(nn,file_name):
    with open(file_name+".pkl","wb") as f:
        pickle.dump(nn,f,pickle.HIGHEST_PROTOCOL)
        
def load_network(file_name):
    with open(file_name+".pkl","rb") as f:
        return pickle.load(f)
        
def dcopy(nn):
    return copy.deepcopy(nn)

def sigmoid(value):
    return 1/(1+np.exp(-value))

def sigmoid_derivative(value):
    return value*(1-value)

def normalize(values,mx=0):
    return values/np.linalg.norm(values) if mx==0 else values/mx

def train(nn,epochs,inputs,target,print_loss=True,loss_step=1000,loss_type="float"):
    for i in range(epochs):
        nn.backpropagation(inputs,target)
        if print_loss and i%loss_step==0:
            loss = np.mean(np.square(nn.predict(inputs)-target))
            if loss_type=="float":
                print(f"loss : {loss:.6f}")
            if loss_type=="raw":
                print(f"loss : {loss}")

class SLP:
    def __init__(self,no_of_inputs,no_of_outputs,learning_rate):
        self.learning_rate=learning_rate
        self.no_of_inputs=no_of_inputs
        self.no_of_outputs=no_of_outputs
        self.weights=np.random.randn(self.no_of_inputs,no_of_outputs)
        self.bias=np.zeros((1,self.no_of_outputs))
    def feedforward(self,inputs):
        return sigmoid(np.dot(inputs,self.weights)+self.bias)
    def backpropagation(self,inputs,target):
        output=self.feedforward(inputs)
        error = (output-target)*sigmoid_derivative(output)
        self.weights-=self.learning_rate*np.dot(inputs.T,error)
        self.bias-=self.learning_rate*np.sum(error,axis=0,keepdims=True)
    def backpropagation_MLP(self,error,inputs,outputs):
        e=error*sigmoid_derivative(outputs)
        self.weights-=self.learning_rate*np.dot(inputs.T,e)
        self.bias-=self.learning_rate*np.sum(e,axis=0,keepdims=True)
    def predict(self,inputs):
        return self.feedforward(inputs)

class MLP:
    def __init__(self,config,learning_rate):
        self.config=config
        self.layers=[]
        self.learning_rate=learning_rate
        self.modlayers(self.config)
    def modlayers(self,config):
    	self.layers=[]
    	for i in range(len(config)-1):
    		self.layers.append(SLP(config[i],config[i+1],self.learning_rate))
    def feedforward(self,inputs):
        io=inputs
        out=[io]
        for i in self.layers:
            io = i.feedforward(io)
            out.append(io)
        return out
    def backpropagation(self,inputs,target,custom_error=None):
        out = self.feedforward(inputs)
        e = 0
        for i in reversed(range(len(self.layers))):
            if i==len(self.layers)-1:
                error = out[i+1]-target
                e = error
            else:
                error = np.dot(e,self.layers[i+1].weights.T)
                e = error
            self.layers[i].backpropagation_MLP(error,out[i],out[i+1])
    def predict(self,inputs):
        return self.feedforward(inputs)[-1]
    def mutate(self,mutation_rate):
        if np.random.rand()<mutation_rate:
            a = np.random.choice(self.layers).weights
            a[np.random.randint(0,a.shape[0])][np.random.randint(0,a.shape[1])]+=np.random.choice([-0.1,0.1])
        if np.random.rand()<mutation_rate:
            a = np.random.choice(self.layers).weights
            a[np.random.randint(0,a.shape[0])][np.random.randint(0,a.shape[1])]=np.random.rand()
        if np.random.rand()<mutation_rate:
            a = np.random.choice(self.layers).weights
            a[np.random.randint(0,a.shape[0])][np.random.randint(0,a.shape[1])]=0
        if np.random.rand()<mutation_rate:
            a = np.random.choice(self.layers).bias
            a = np.random.rand()
        if np.random.rand()<mutation_rate:
            a = np.random.choice(self.layers).bias
            a += np.random.choice([-0.1,0.1])