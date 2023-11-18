from neurogen import *

gen = 1
population = 10
mutation_rate = 0.3
print_step = 100

pop = []
for i in range(population):
	pop.append(MLP([2,3,1],0.1))

inps = np.array([[0,0],[0,1],[1,0],[1,1]])
tar = np.array([[0],[1],[1],[0]])
loss = 0.40

def cost_function(nn):
	return np.mean(np.square(nn.predict(inps)-tar))

print(pop[0].predict(inps))

while loss>0.001:
	pop.sort(key=cost_function)
	loss = cost_function(pop[0])
	if gen%print_step==0:
		print(gen,loss)
	fittest = pop[0]
	
	pop=[]
	for i in range(population):
		pop.append(dcopy(fittest))
	for i in pop:
		i.mutate(mutation_rate)
	gen+=1

print(pop[0].predict(inps))
