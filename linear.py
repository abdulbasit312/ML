import theano
import numpy
x=theano.tensor.fvector('x')
y=theano.tensor.fvector('y')
m=theano.shared(0.6,'m')
c=theano.shared(0.8,'c')
yhat=m*x+c
#cost function
cost=theano.tensor.mean(theano.tensor.sqr(y-yhat))/2
# gradient descent algorithm
LR=0.1
gradm=theano.tensor.grad(cost,m)
gradc=theano.tensor.grad(cost,c)
mn=m-LR*gradm
cn=c-LR*gradc
train=theano.function([x,y],cost,updates=[(m,mn),(c,cn)])

area=[1.2,2.4,3.6,1.8,2.1,4.5,8.4,7.1,2.6,5.6,1.6]
price=[210,280,340,190,260,460,780,650,280,480,200]
area=numpy.array(area).astype('float32')
price=numpy.array(price).astype('float32')

for i in range(20000):
    costval=train(area,price)
    print(costval)
    
print(m.get_value())
print(c.get_value())

pred=area*m.get_value()+c.get_value()
import matplotlib.pyplot as plt
plt.scatter(area,price)
plt.plot(area,pred,'r')
plt.show()
