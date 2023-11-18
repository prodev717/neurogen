from ursina import *
from neurogen import *
import math
#hybrid algrothim - genetic and random hill climbing
population = 50

class Car(Entity):
	pop=[]
	def __init__(self,brain):
		if np.random.rand()<0.5:
			super().__init__(model="Car1",collider="box",y=0.4,z=np.random.randint(-24,25),rotation_y=180)
		else:
			super().__init__(model="Car1",collider="box",y=0.4,z=np.random.randint(-24,25))
		self.brain=brain
		Car.pop.append(self)
	def update(self):
		r1 = raycast(self.position,self.forward.normalized(),ignore=tuple(Car.pop),distance=6).distance
		r2 = raycast(self.position,self.left.normalized(),ignore=tuple(Car.pop),distance=6).distance
		r3 = raycast(self.position,self.right.normalized(),ignore=tuple(Car.pop),distance=6).distance
		r4 = raycast(self.position,(self.left+self.forward).normalized(),ignore=tuple(Car.pop),distance=6).distance
		r5 = raycast(self.position,(self.right+self.forward).normalized(),ignore=tuple(Car.pop),distance=6).distance
		self.position+=self.forward*time.dt*2
		res = self.brain.predict(normalize(np.array([r1,r2,r3,r4,r5]),mx=6))[0]
		if np.argmax(res)==1:self.rotation_y-=math.radians(30)
		if np.argmax(res)==2:self.rotation_y+=math.radians(30)
		if self.intersects(ground).hit:
			Car.pop.remove(self)
			destroy(self)

def update():
	if len(Car.pop)==0:
		for i in range(population):
			Car(MLP([5,8,3],0.1))

game = Ursina()
for i in range(population):
	Car(MLP([5,8,3],0.1))
ground = Entity(model="road",scale=26,collider="mesh",texture="grass")
EditorCamera()
game.run()