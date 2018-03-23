import random
import numpy as np
class Network(object):
    def __init__(self,sizes):
        """传进来的一维矩阵的值分别为神经网络每一层上面神经元的个数"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        """设置的参数个数为除输入层外，每一层神经元的个数的矩阵"""
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        """每俩层的之间的权重设置为一个矩阵shape为(前一层的神经元个数，后一层神经元个数)"""
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        
    """前馈算法，主要是使用sigmoid函数"""
    def feedforward(self,a):
        """a由输入值变成了输出值"""
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a)+b)
		return a
            
    """随机下降算法的函数"""
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        """不一定会有测试数据"""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        """epochs是循环的次数，每一次都使用绝大部分的traing_data"""
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                           for k in range(0,n,mini_batch_size)]
            """每一次的只使用一个mini_batch进行更新"""
            """一个mini_batch中有一定数目的测试数据"""
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
                
            if test_data:
                """用来测试准确率"""
                print("Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test))

            else:
                print("Epoch {0} complete".format(j))
    
    """使用每一个mini_minibatch进行更新参数的函数"""
    def update_mini_batch(self,mini_batch,eta):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x,y in mini_batch:
            
            """利用BP函数求得权重和偏差的偏导数"""
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)
            
            """重新计算nabla_b以及nabla_w的值"""
            nabla_b = [nb +dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw +dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
            
        self.weights = [w - (eta/len(mini_batch))*nm for w,nm in zip(self.weights,nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b ,nb in zip(self.biases,nabla_b)]
        
    """BP算法实现的细节"""
    def backprop(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation = x
        activations = [x]
        
        zs = []
        """输入层好像没有使用sigmoid函数"""
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        """这是输出层的delta更新方式"""
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """原理很简单
           使用feedforward进行测试数据
           np.argmax()这个判断预测的是哪一个节点
           以tuple()的形式存储在test_results中
           
           最后一x是否等于y的形式来统计有多少个预测结果是对的"""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """计算loss的函数"""
        return (output_activations-y)
		
		