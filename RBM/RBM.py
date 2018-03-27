from __future__ import print_function

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


"""Setting up"""

def nudge_dataset(X,Y):
	direction_vectors = [
		[[0, 1, 0],
		[0, 0, 0],
		[0, 0, 0]],

		[[0, 0, 0],
		[1, 0, 0],
		[0, 0, 0]],

		[[0, 0, 0],
		[0, 0, 1],
		[0, 0, 0]],

		[[0, 0, 0],
		[0, 0, 0],
		[0, 1, 0]]
	]
	#ravel是降维的用途
	shift = lambda x,w: convolve(x.reshape((8,8)),mode = 'constant',weights = w).ravel()
	
	X = np.concatenate([X]+[np.apply_along_axis(shift,1,X,vector) for vector in direction_vectors])
	Y = np.concatenate([Y for _ in range(5)],axis = 0)
	
	return  X,Y
	
	
	
digits = datasets.load_digits()
X = np.asarray(digits.data,'float32')
X,Y = nudge_dataset(X,digits.target)
X = (X - np.min(X,0))/(np.max(X,0) + 0.0001)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state = 0,verbose = True)

classifier = Pipeline(steps = [('rbm',rbm),('logisitc',logistic)])


#training


rbm.learning_rate = 0.06
rbm.n_iter = 20

rbm.n_components = 100

logistic.C = 6000.0


classifier.fit(X_train,Y_train)

logistic_classifier = linear_model.LogisticRegression(C = 100.0)
logistic_classifier.fit(X_train,Y_train)


print("Logistic regression using RBM features:\n%s\n" % (metrics.classification_report(Y_test,classifier.predict(X_test))))
print("Logistic regression using raw pixel features:\n%s\n" % (metrics.classification_report(Y_test,logistic_classifier.predict(X_test))))



plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
	plt.subplot(10, 10, i + 1)
	plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,interpolation='nearest')
	plt.xticks(())
	plt.yticks(())
plt.suptitle('100 components extracted by RBM', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()

