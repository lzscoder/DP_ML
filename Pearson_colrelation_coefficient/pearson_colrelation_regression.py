import numpy as np

import math

from astropy.units import Ybarn


#这个是计算pearson相关系数的函数
def computerCorrelation(X,y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    #直接使用API计算相关系数
    
    SSR = 0
    X_var = 0
    y_var = 0
    
    for i in range(0,len(X)):
        X_diff = X[i] - X_mean
        y_diff = y[i] - y_mean
        
        SSR += (X_diff*y_diff)
        X_var += X_diff**2
        y_var += y_diff**2
        
    SST = math.sqrt(X_var*y_var)
    
    return SSR/SST
        
def polyfit(X,y,degree):
    """处理线性回归的函数"""
    results = {}
    
    #进行线性回归的预处理
    coeffs = np.polyfit(X,y,degree)
    #get回归方程的参数
    results['polynomial'] = coeffs.tolist()
    
    #由回归方程参数得到回归方程直线
    p = np.poly1d(coeffs)
    
    """下面=计算R**2"""
    #由回归方程得到预测值
    yhat = p(X)
    y_mean = np.sum(y)/len(y)
    
    ssreg = np.sum((yhat - y_mean)**2)
    sstot = np.sum((y - y_mean)**2)
    
    results['determination'] = ssreg / sstot
    
    return results

X_test = [1,5,8,7,9]
y_test = [10,12,24,21,34]

print(computerCorrelation(X_test,y_test))


print(polyfit(X_test,y_test,1))

    
    