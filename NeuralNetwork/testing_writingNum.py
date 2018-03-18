#识别手写数字，用自己手写的两层神经网络

import numpy as np

from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split


digits = load_digits()
X = digits.data
Y = digits.target
#1797个图片，也就是1797个数字,8*8个维度

#在sklearn中所有的值都要在0-1之间
X -= X.min()
X /= X.max()

nn = NeuralNetwork([64,100,10],'logistic')
#划分的比例为0.75
X_train,X_test ,Y_train ,Y_test = train_test_split(X,Y)

label_train = LabelBinarizer().fit_transform(Y_train)
label_test = LabelBinarizer().fit_transform(Y_test)

print("start fiting:")

nn.fit(X_train,label_train)


predictions = []

for i in range(X_test.shape[0]):
    pre = nn.predict(X_test[i])
    pre_sort_index = np.argsort(pre)
    predictions.append(pre_sort_index[-1])
    
print (confusion_matrix(Y_test, predictions))
print (classification_report(Y_test, predictions))