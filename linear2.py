import numpy
area=[1.2,2.4,3.6,1.8,2.1,4.5,8.4,7.1,2.6,5.6,1.6]
price=[210,280,340,190,260,460,780,650,280,480,200]
area=numpy.array(area).reshape(11,1)
price=numpy.array(price)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(area,price)
model.coef_
model.intercept_
