import random
import math
def MUTATE(X,RATE=.05):
	def MU(X,RATE):
		if random.random()<=RATE:
			return random.random()
		else:
			return X
	return [MU(x,RATE) for x in X]
def CROSSOVER(A,B,RATE=.5):
	if random.random()<=RATE:
		return [*A[0:len(A)//2],*B[len(B)//2:]],[*B[0:len(B)//2],*A[len(A)//2:]]
	else:
		return [A,B]
def FIT(X,Y,N):
	fout = PWS(N(X),Y)
	return sum(DOT(fout,fout))
def LCheck(LC):
	try:
		return len(LC)
	except:
		return 0
def ACTIVATE(ATI):
	return 1/(1+math.e**-ATI)
def DOT(X,Y):
	return [X[F]*Y[F] for F in range(min(LCheck(X),LCheck(Y)))]
def PWS(X,Y):
	return [Y[F]-X[F] for F in range(min(LCheck(X),LCheck(Y)))]
class Neuron:
	def __init__(self,N=1):
		self.set_bias(random.random())
		self.set_weights([random.random() for n in range(N)])
	def set_bias(self,BIAS):
		self.bias = BIAS
	def set_weights(self,WEIGHTS):
		self.weights = WEIGHTS
	def get_bias(self):
		return self.bias
	def get_weights(self):
		return self.weights
	def __repr__(self):
		return '{}'.format(self)
	def __str__(self):
		return '{}+{}'.format(self.get_weights(),self.get_bias())
	def get_mutate(self,MuRate=.05):
		new_neur = Neuron()
		new_neur.set_weights(MUTATE(self.get_weights(),MuRate))
		new_neur.set_bias(MUTATE([self.get_bias()],MuRate)[0])
		return new_neur
	def get_crossover(self,other):
		return CROSSOVER(self.get_weights(),other.get_weights())
	def parent(self,other):
		cross = self.get_crossover(other)
		left,right = Neuron(),Neuron()
		left.set_weights(cross[0]),right.set_weights(cross[1])
		left.set_bias(self.get_bias()),right.set_bias(other.get_bias())
		left.get_mutate()
		right.get_mutate()
		return [left,right]
	def generate(self,other,N=1):
		return [self.parent(other) for n in range(N)]
	def __call__(self,other):
		return ACTIVATE(sum(DOT(self.get_weights(),other))+self.get_bias())
	def get_fit(self,X,Y):
		return FIT(X,Y,self)
	def get_copy(self):
		new_sel = self.get_mutate(0)
		return new_sel
class Layer(Neuron):
	def __init__(self,M,N=3):
		self.set_layer([Neuron(M) for n in range(N)])
		self.set_m(M)
		self.set_n(N)
	def set_m(self,M):
		self.m = M
	def set_n(self,N):
		self.n = N
	def get_m(self):
		return self.m
	def get_n(self):
		return self.n
	def set_layer(self,LAYER):
		self.layer = LAYER
	def get_layer(self):
		return self.layer
	def __str__(self):
		return '{}'.format(self.get_layer())
	def __repr__(self):
		return '{}'.format(self)
	def __call__(self,other):
		return [F(other) for F in self.get_layer()]
	def get_mutate(self,MuRate=.05):
		new_lay = Layer(2)
		new_lay.set_layer([F.get_mutate(MuRate) for F in self.get_layer()])
		return new_lay
	def get_crossover(self,other):
		return CROSSOVER(self.get_layer(),other.get_layer())
	def parent(self,other,MuRate=.05):
		cross = self.get_crossover(other)
		left,right = Layer(3,3),Layer(3,3)
		left.set_layer(cross[0]),right.set_layer(cross[1])
		left.get_mutate(MuRate)
		right.get_mutate(MuRate)
		return [left,right]
	def get_ranin(self):
		return self([random.random() for m in range(self.get_m())])
	def get_tfit(self,other):
		return sum([1-F for F in self(other)])
	def get_stfit(self,*other):
		return sum([self.get_tfit(F) for F in other])
class Network(Layer):
	def __init__(self,M,N,D=3):
		self.set_network([Layer(M,N),*[Layer(N,N) for n in range(D-1)]])
	def set_network(self,NETWORK):
		self.layer = NETWORK
	def get_network(self):
		return self.layer
	def __repr__(self):
		return '{}'.format(self)
	def __str__(self):
		return '{}'.format(self.get_network())
	def __call__(self,other):
		calout = self.get_network()[0](other)
		for L in self.get_network()[1:]:
			calout = L(calout)
		return calout
	def get_mutate(self,MuRate=.05):
		new_net = Network(2,3)
		new_net.set_network([F.get_mutate(MuRate) for F in self.get_network()])
		return new_net
	def get_crossover(self,other):
		return CROSSOVER(self.get_network(),other.get_network())
	def parent(self,other,MuRate=.05):
		cross = self.get_crossover(other)
		left,right = Network(3,3),Network(3,3)
		left.set_network(cross[0]),right.set_network(cross[1])
		left.get_mutate(MuRate)
		right.get_mutate(MuRate)
		return [left,right]
class Decriminator(Network):
	def __init__(self,M,N,D=3):
		super().__init__(M,N,D)
		self.layer.append(Neuron(N))
	def get_mutate(self,MuRate=.05):
		new_net = Decriminator(2,3)
		new_net.set_network([F.get_mutate(MuRate) for F in self.get_network()])
		return new_net
	def get_crossover(self,other):
		return CROSSOVER(self.get_network(),other.get_network())
	def parent(self,other,MuRate=.05):
		cross = self.get_crossover(other)
		left,right = Decriminator(3,3),Decriminator(3,3)
		left.set_network(cross[0]),right.set_network(cross[1])
		left.get_mutate(MuRate)
		right.get_mutate(MuRate)
		return [left,right]
	def get_tfit(self,other):
		return 1-self(other)
	def get_stfit(self,*other):
		return sum([self.get_tfit(F) for F in other])
class GANN(Decriminator):
	def __init__(self,M,N,D=3):
		self.set_dec(Decriminator(M,N,D))
		self.set_gen(Network(M,N,D))
		self.set_network([self.get_gen(),self.get_dec()])
	def set_dec(self,DEC):
		self.dec = DEC
	def set_gen(self,GEN):
		self.gen = GEN
	def get_dec(self):
		return self.dec
	def get_gen(self):
		return self.gen
class Flask(Layer):
	def __init__(self,Population,M,N):
		self.set_layer([GANN(M,N) for population in range(Population)])
	def get_tfit(self,other):
		return sum([1-F for F in self(other)])
	def get_stfit(self,*other):
		return [self.get_tfit(F) for F in other]
	def get_sample(self,K=1):
		return random.sample(self.get_layer(),K)
	def get_comp(self,other):
		l,r = self.get_sample(2)
		if l.get_stfit(*other)<r.get_stfit(*other):
			return l
		else:
			return r
	def get_tournament(self,other,N=8):
		new_fla = Flask(N,3,2)
		new_fla.set_layer([self.get_comp(other) for n in range(N)])
		return new_fla
	def parent(self,MuRate=.05):
		l,r = self.get_sample(2)
		a,b = self.get_sample(2)
		new_fla = Flask(3,2,3)
		new_fla.set_layer([r.get_mutate(),*l.parent(r),*a.parent(b),a.get_mutate()])
		return new_fla
	def get_mutate(self,MuRate=.04166):
		new_net = Network(2,3)
		new_net.set_network([F.get_mutate(MuRate) for F in self.get_network()])
		return new_net
	def get_crossover(self,other):
		return CROSSOVER(self.get_network(),other.get_network())
a = Flask(100000,2,1)
f = [0,3],[1,1],[2,4],[3,1],[4,5]
b = a.get_tournament(f,100)
for epoch in range(1000):
	b = b.get_tournament(f,100)
	print(b.get_sample(3)[-1].get_gen().get_stfit(*f))