import numpy as np
import random

def GradientDescent(X,Y,theta,alpha,m,numIterations):
    X_tran = np.transpose(X)
    for i in range(numIterations):
        #X_tran为测试数据  theta为学习参数
        #hypothesis为结果向量
        #loss为标准与预测之间的误差
        #cost为一种损失，不知道为什么要这样做，好像没有用到sigmoid函数
        hypothesis = np.dot(theta,X_tran)
        loss = hypothesis - y
        cost = np.sum(loss**2)/(2*m)
        #这是求梯度的一个过程
        gradient  = np.dot(X_tran, loss)/m
        #theta的更新步骤
        theta = theta - alpha*gradient
        print ("Iteration %d | cost :%f" %(i,cost))
        
    #最后返回的theta为直线方程中的参数b0,b1,b2.....
    return theta


def genData(numPoints,bias,variance):
    X = np.zeros(shape = (numPoints,2))
    Y =np.zeros(shape = numPoints)
    
    for i in range(numPoints):
        X[i][0] = 1
        X[i][1] = i
        
        Y[i] = i+bias+random.uniform(0,1)*variance
        
    return X,Y

x,y = genData(100,25,10)
numIterations  = 100000
alpha = 0.0005
theta = np.ones(2)
theta = GradientDescent(x,y,theta,alpha,100,numIterations)
print(theta)

