import numpy as np

def SLR(X,Y):
    #y=b1*x+b0
    #这个函数就是求取b0与b1的值
    
    b1_sum_son = 0
    b1_sum_mom = 0
    
    x = np.array(X)
    y = np.array(Y)
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    for i in range(len(x)):
        b1_sum_son += (x[i]-x_mean)*(y[i]-y_mean)
        b1_sum_mom += (x[i]-x_mean)*(x[i]-x_mean)
        
        
    b1 = b1_sum_son/b1_sum_mom
    b0 = y_mean - b1*x_mean
    
    return b0,b1

def predict(x,b0,b1):
    return b1*x+b0
x = [1, 3, 2, 1, 3]
y = [14, 24, 18, 17, 27]    

b0,b1 = SLR(x,y)

test_x = 6
print("predict:",predict(test_x,b0,b1))
    