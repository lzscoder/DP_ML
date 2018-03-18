import numpy as np

#定义两个激励函数
def tanh(x):
    return np.tanh(x)
def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)


def logistic(x):
    return 1.0/(1 + np.exp(-x))
def logistic_deriv(x):
    a=logistic(x)
    return a*(1-a)


#定义神经网络
class NeuralNetwork():
    def __init__(self, layers, activation = 'tanh'):
        
        #定义神经网络内的激励函数
        if activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        elif activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        
        #定义神经节点间的权重
        self.Weights = []
        
        #下面的layers[i-1]可能不用加1
        #这里有一个偏置bias参数的问题，所以有加一的问题
        for i in range(1,len(layers) -1 ):
            self.Weights.append((2*np.random.random((layers[i - 1]+1, layers[i]+1))-1)*0.25)
            self.Weights.append((2*np.random.random((layers[i]+1, layers[i + 1]))-1)*0.25)
            
    def fit(self, X, y, learning_rate = 0.2, epochs = 10000):

        X = np.atleast_2d(X)#确保是一个2维的数据集
        temp = np.ones([X.shape[0],X.shape[1] + 1])
        temp[:,0:-1] = X
        X = temp
        y = np.array(y)
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]#这里的a是一个列表，第一个元素是随机从训练集中随机抽取的一行数据
            
            #对每一层实行前馈神经网络算法
            for l in range(len(self.Weights)):
                a.append(self.activation(np.dot(a[l], self.Weights[l])))#使用激励方程把点积转换为一定区间的数
                
            error = y[i] - a[-1]  #计算输出层的损失值
            deltas = [error * self.activation_deriv(a[-1])]   #计算更新后的输出层error
            
            #反向计算所有隐藏层的误差
            #为什么len(a)-1是输出层的位置
            #再减一是最靠近输出层的隐藏层
            #这里的范围为[len(a)-2,0)
            for i in range(len(a)-2,0,-1):
                #这里的weights的值在*的前面，需要将其转置
                deltas.append(deltas[-1].dot(self.Weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            
            for i in range(len(self.Weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                # 这里为什么需要将a[i]给转置呢
                self.Weights[i] += learning_rate * layer.T.dot(delta)
                

    def predict(self,x):
        #转化测试数据,使它多出一列元素出来
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        #开始每一层的测试，使用点层和激励函数来计算出每一层的输出向量
        for l in range(0, len(self.Weights)):
            a = self.activation(np.dot(a, self.Weights[l]))
        #返回的是最后一层的计算结果
        return a