from Hierarchical_clustering import hcluster
from Hierarchical_clustering import getheight
from Hierarchical_clustering import printclust
import os 
from PIL import Image , ImageDraw
import numpy as np



def drawdendrogram(clust,imlist,jpeg = 'cluster.jpg'):
	h = getheight(clust)*20
	w = 1200
	depth = getheight(clust)
	
	
	scaling = float(w-150)/depth
	
	img = Image.new('RGB',(w,h),(255,255,255))
	draw = ImageDraw.Draw(img)
	
	draw.line((0,h/2,10,h/2),fill = (255,0,0))
	
	drawnode(draw,clust,10,int(h/2),scaling,imlist,img)
	
	img.save(jpeg)
	
	
def drawnode(draw,clust,x,y,scaling,imlist,img):
	if clust.id < 0:
		h1 = getheight(clust.left)*20
		h2 = getheight(clust.right)*20
		top = y - (h1+h2)/2
		bottom = y + (h1+h2)/2
		
		l1 = clust.distance*scaling
		draw.line((x,top+h1/2,x,bottom - h2/2),fill = (255,0,0))
		draw.line((x,top+h1/2,x+l1,top+h1/2),fill = (255,0,0))
		draw.line((x,bottom - h2/2,x+l1,bottom - h2/2),fill = (255,0,0))
		
		drawnode(draw,clust.left,x+l1,top + h1/2,scaling,imlist,img)
		drawnode(draw,clust.right,x+l1,bottom - h2/2,scaling,imlist,img)
		
	else:
		nodeim = Image.open(imlist[clust.id])
		nodeim.thumbnail((20,20))
		ns = nodeim.size
		#print( x,y-ns[1])
		#print(x+ns[0])
		
imlist = []
folderPath = r"F:\TrackLearn2018\03DPMZfountation_1\hierarchical_clustering\picture"
for filename in os.listdir(folderPath):
	if os.path.splitext(filename)[1] == '.jpg':
		imlist.append(os.path.join(folderPath,filename))
n = len(imlist)

print(n)

features = np.zeros((n,3))

for i in range(n):
	im = np.array(Image.open(imlist[i]))
	R = np.mean(im[:,:,0].flatten())
	G = np.mean(im[:,:,1].flatten())
	B = np.mean(im[:,:,2].flatten())
	
	features[i] = np.array([R,G,B])
	
tree = hcluster(features)
#print(tree)
#drawdendrogram(tree,imlist,jpeg = 'sunset.jpg')

printclust(tree)