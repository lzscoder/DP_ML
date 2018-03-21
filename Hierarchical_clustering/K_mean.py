import numpy as np

def Kmeans(X,k,MaxIt):
	"""输入的X为全部的数据k为分类个数的意向,Maxit为最多多少次中心点的交换"""
	numPoints,numDim = X.shape  
	dataSets = np.zeros((numPoints,numDim+1))
	dataSets[:,:-1] = X
	
	#随机找到k个点作为中心点
	centroids = dataSets[np.random.randint(numPoints,size = k),:]
	#对中心点进行赋值操作
	#这里有K个数据也就是说中心点有k行，数据有numdim列
	centroids = dataSets[0:k,:]
	#对中心点按顺序进行分类
	centroids[:,-1] = range(1,k+1)
	
	
	iterations = 0
	oldCentroids = None
	
	#开始进行k-means算法
	while not shouldStop(oldCentroids,centroids,iterations,MaxIt):
		
		oldCentroids = np.copy(centroids)
		iterations += 1
		#根据数据值以及中心点来更新点的分类
		updateLabel(dataSets,centroids)
		#根据新的分类来更新中心点的位置
		centroids = getCentroids(dataSets,k)
		
	return dataSets
def shouldStop(oldCentroids,centroids,iterations,MaxIt):
	#抵达循环次数
	if iterations>MaxIt:
		return true
	#前一个分类和这个分类的结果相同 
	return np.array_equal(oldCentroids,centroids)

def updateLabel(dataSets,centroids):
	
	numPoints,numDim = dataSets.shape
	for i in range(0,numPoints):
		#赋值为分类的结果
		#参数为所有的第i个数据，以及重心点的位置
		dataSets[i,-1] = getLabelFromClosestCentriod(dataSets[i,:-1],centroids)
		
def getLabelFromClosestCentriod(dataSetRow,centroids):
	#获取所有的中心点的值
	#centroids是一个二维数组
	label = centroids[0,-1]
	#找到最小的那一个dist
	minDist = np.linalg.norm(dataSetRow - centroids[0,:-1])
	for i in range(1,centroids.shape[0]):
		dist = np.linalg.norm(dataSetRow - centroids[i,:-1])
		if dist <minDist:
			dist = minDist
			label = centroids[i,-1]
	#print("minDist：”,minDist)
	return label

def getCentroids(dataSets,k):
	result = np.zeros((k,dataSets.shape[1]))
	for i in range(1,k+1):
		oneCluster = dataSets[dataSets[:,-1] == i,:-1]
		result[i-1,:-1] = np.mean(oneCluster,axis = 0)
		result[i-1,-1] = i
		
	return result
	
x1 = np.array([1,1])
x2 = np.array([2,1])
x3 = np.array([4,1])
x4 = np.array([5,4])

testX = np.vstack((x1,x2,x3,x4))

result = Kmeans(testX,2,10)

print(result)
	