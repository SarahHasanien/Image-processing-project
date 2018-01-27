# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import cv2;
import numpy as np
from skimage import measure
from PIL import Image
import sys

class DSU:
	parents = list()
	GroupSize = list()
	#or: parents = []
	def __init__(self,num,initgroupsize):
		for i in range(0,num):
			self.parents.append(i)
			self.GroupSize.append(initgroupsize)
	def findParent(self,i):
		if (self.parents[i] == i):
			return i
		self.parents[i]= self.findParent(self.parents[i])
		return self.parents[i]
	def sameSet(self,x,y):
		flag=(self.findParent(x) == self.findParent(y))
		return flag
	def unionSets(self,x,y):
		maxSize=0
		parent1 = self.findParent(x)
		parent2 = self.findParent(y)
		if (parent1 == parent2):
			return self.GroupSize[parent1]
		if (self.GroupSize[parent1] >= self.GroupSize[parent2]):
			self.parents[parent2] = parent1
			self.GroupSize[parent1] += self.GroupSize[parent2]
			maxSize = self.GroupSize[parent1]
		else:
			self.parents[parent1] = parent2
			self.GroupSize[parent2] += self.GroupSize[parent1]
			maxSize = self.GroupSize[parent2]
		return maxSize
	def getSize(self,x):
		return self.GroupSize[self.findParent(x)]
	
	def clearClass(self):
		self.parents[:]=[]
		self.GroupSize[:]=[]


im_in = cv2.imread("colors.jpg",cv2.IMREAD_ANYCOLOR)
rows, cols,j = im_in.shape
edges=cv2.Canny(im_in,100,200)
h , w = edges.shape[:2]
im_floodfill= edges.copy()
mask = np.zeros((h+2, w+2),np.uint8)
cv2.floodFill(im_floodfill, mask, (0,0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
thresh = cv2.threshold(im_floodfill_inv, 200, 255, cv2.THRESH_BINARY)[1]
L = measure.label(thresh, neighbors=8, background=0)
print ("Number of components:", np.max(L))
shapesDict={'square':0,'rectangle':0,'triangle':0,'circle':0,'rhombus':0,'hexagon':0,'pentagon':0,'ellipse':0}
label_number = 1
while True:
	temp = np.uint8(L==label_number) * 255
	if not cv2.countNonZero(temp):
		break
	img = Image.new('1', (h, w))
	mynewimg = np.zeros((h, w, 3), dtype=np.uint8)
	for x in range(0,w):
		for y in range(0,h):
            		oldcol=temp[y,x]
            		mynewimg[y,x]=(oldcol,oldcol,oldcol)
	img = Image.fromarray(mynewimg, 'RGB')
	name=str(label_number)+".jpg"
	img.save(name)    
	label_number += 1

label_number = 1
while True:
	temp = np.uint8(L==label_number) * 255
	name=str(label_number)+".jpg"
	print ("the shape no: ", label_number)
	img = cv2.imread(name,0)
	if not cv2.countNonZero(temp):
		break    
	contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	c = max(contours, key=cv2.contourArea)
	#print dst
	#print corners
	#print label_number
	# Now draw them
	((centerx,centery),(width,height),angle)=rect = cv2.minAreaRect(c)
	#print(centerx)
	#print(centery)
	#print(height*width)
	n_white_pix = np.sum(temp == 255)
	#print(n_white_pix)
	ratio=(height*width)/n_white_pix
	ratio2=height/width
	if width>height:
		width,height=height,width
	print(ratio)
	print(ratio2)
	if ratio>0.9 and ratio<1.2:
        
		#if height/width>0.9 and height/width<1.1:
		if height/width>0.8 and height/width<1.2:
			print("square")
			shapesDict['square']=shapesDict['square']+1
		else:
			print("rectangle")
			shapesDict['rectangle']=shapesDict['rectangle']+1

	else:
		#to detect whether it was rhombus,pentagon,hexagon or triangle
		#hough lines
		#print "else"
		#first take an image only of the edges of the current shape using canny
		tempedges = cv2.Canny(temp, 100, 200,apertureSize =3)
		#then get the matrix containing all rho and theta of these edges
		rho_resolution = 1
		theta_resolution = np.pi/180
		threshold =70
        
		hough_lines = cv2.HoughLines(tempedges, rho_resolution , theta_resolution , threshold)
		count=0
		if hough_lines is None:
			if height/width>0.8 and height/width<1.2:
				print("circle")
				shapesDict['circle']=shapesDict['circle']+1
			else:
				shapesDict['ellipse'] = shapesDict['ellipse']+1
				print ("ellipse")
		if hough_lines is not None:
			dim1=len(hough_lines)
			count=0
			#print "hough lines size:", dim1
			houghDSU= None
			houghDSU = DSU(dim1,1)
			for i1 in range(0,dim1):
				rho=hough_lines[i1,0,0]
				theta=hough_lines[i1,0,1]
				for j1 in range(0,dim1):
					rho1=hough_lines[j1,0,0]
					theta1=hough_lines[j1,0,1]
					diff=rho-rho1
					diff2=theta-theta1
					if(abs(diff)<13 and abs(diff2) < 0.02):
						houghDSU.unionSets(i1,j1)
			for i in range(0,dim1):
				if(houghDSU.findParent(i)==i):
					count = count +1
			houghDSU.clearClass()
		#to increase the number of shapes in the dictionary of shapes:
			if(count == 6):
				shapesDict['hexagon'] = shapesDict['hexagon']+1
				print ("hexagon")
			elif(count == 4):
				shapesDict['rhombus'] = shapesDict['rhombus']+1
				print ("rhombus")
			elif(count == 5):
				shapesDict['pentagon'] = shapesDict['pentagon']+1
				print ("pentagon")
			else:
				shapesDict['triangle'] = shapesDict['triangle']+1
				print ("triangle")
			print ("number of edges is:", count,"while it was",dim1)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(img,[box],0,(255,0,0),2)
	#cv2.imshow(str(label_number),img)
	label_number += 1
	temp=[]


print ("Square: ",shapesDict['square'])
print ("Rectangle: ",shapesDict['rectangle'])
print ("Triangle: ",shapesDict['triangle'])
print ("Circle: ",shapesDict['circle'])
print ("Rhombus: ",shapesDict['rhombus'])
print ("Hexa: ",shapesDict['hexagon'])
print ("Pentagon: ",shapesDict['pentagon'])
print ("Ellipse: ",shapesDict['ellipse'])
#cv2.imshow("corners",im_in)
#cv2.imshow("img1",im_floodfill_inv)
#cv2.waitKey(0)
sys.exit()

