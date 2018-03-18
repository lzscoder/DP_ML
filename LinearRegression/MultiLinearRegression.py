import numpy as np
from numpy import genfromtxt
from sklearn import datasets,linear_model

filename = "Delivery_Dummy.csv"

datas = genfromtxt(filename,delimiter = ',')

X_data = datas[1:-1,:-1]
Y_data = datas[1:-1,-1]


NLR = linear_model.LinearRegression()

#print(X_data,Y_data)
NLR.fit(X_data,Y_data)
pre_test = X_data[-1,:-1].reshape(1,4)

print(NLR.predict([[90,2,0,0,1]]))
