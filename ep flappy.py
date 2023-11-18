import pygame
from neurogen import *
import math

pygame.init()

population = 700
mutation_rate = 0.25
gen = 1
print_step=10

class Bird:
	def __init__(self):
		self.x,self.y=100,100
		self.v,self.g=0,1
		self.c=(200,200,100)
		self.score=0
		self.brain=MLP([3,5,2],0.1)
	def draw(self):
		self.rec = pygame.rect.Rect(self.x,self.y,50,50)
		pygame.draw.rect(screen,self.c,self.rec)
		pygame.draw.line(screen,(255,255,255),(self.x,self.y+25),(pipes[cp%2].x+50,self.y+25))
		pygame.draw.line(screen,(255,255,255),(self.x,self.y),(pipes[cp%2].x+50,pipes[cp%2].r))
		pygame.draw.line(screen,(255,255,255),(self.x,self.y+50),(pipes[cp%2].x+50,pipes[cp%2].r+150))
		self.d1 = math.dist((self.x,self.y+25),(pipes[cp%2].x+50,self.y+25))
		self.d2 = math.dist((self.x,self.y),(pipes[cp%2].x+50,pipes[cp%2].r))
		self.d3 = math.dist((self.x,self.y+50),(pipes[cp%2].x+50,pipes[cp%2].r+150))
	def update(self):
		self.v+=self.g
		self.y+=self.v 
	def jump(self):
		self.v=-7
	def think(self):
		d1,d2,d3=normalize(self.d2,mx=600),normalize(self.d2,mx=600),normalize(self.d3,mx=600)
		res = np.argmax(self.brain.predict((d1,d2,d3)))
		if res==0:
			self.jump()
	def out(self):
		if self.rec.colliderect(ground) or self.rec.colliderect(sky):
			return True
		else:
			for i in pipes:
				if i.rec1.colliderect(self.rec) or i.rec2.colliderect(self.rec):
					return True
			return False
	def check(self):
		global cp
		if pipes[cp%2].x+50==self.x:
			self.score+=1
			cp+=1

class Pipe:
	def __init__(self,x):
		self.x=x
		self.r=np.random.randint(0,250)
	def draw(self):
		self.rec1 = pygame.rect.Rect(self.x,0,50,self.r)
		self.rec2 = pygame.rect.Rect(self.x,self.r+150,50,400-(self.r+150))
		pygame.draw.rect(screen,(100,200,200),self.rec1)
		pygame.draw.rect(screen,(100,200,200),self.rec2)
	def update(self):
		if self.x<=-50:
			self.x=600
			self.r=np.random.randint(0,250)
		self.x-=2

screen = pygame.display.set_mode((600,400))
pygame.display.set_caption("ne flappy")
clock = pygame.time.Clock()
run = True

cp=0
players = []
for i in range(population):
	players.append(Bird())
pipes = []
pipes.append(Pipe(600))
pipes.append(Pipe(900))
ground = pygame.rect.Rect(0,401,600,20)
sky = pygame.rect.Rect(-21,0,600,20)
dead = []

def fitness_function(x):
	return x.score

def nextGen():
	global dead,gen,print_step
	gen+=1
	dead.sort(key=fitness_function,reverse=True)
	fittest = dead[0].brain
	dead=[]
	for i in range(population):
		np = Bird()
		np.brain = dcopy(fittest)
		np.brain.mutate(mutation_rate)
		players.append(np)
	if gen%print_step==0:
		print(gen)

while run:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run=False
	screen.fill((0,0,0))
	for i in pipes:
		i.draw()
		i.update()
	for i in players:
		i.draw()
		i.update()
		i.check()
		i.think()
		if i.out():
			dead.append(i)
			players.remove(i)
	if len(players)==0:
		nextGen()
	pygame.display.update()
	clock.tick(30)
pygame.quit()