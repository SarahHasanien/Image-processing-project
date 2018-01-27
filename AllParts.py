# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import time
from random import randint
from skimage import measure
from PIL import Image
import sys

#video processing
MIN_RADIUS = 2
cnt1=0
cnt2=0
cnt3=0
cnt4=0
aOld=0
f=0
flag=0
mp = {}

count=0   #to count how many times it enters if (if count == 2 then calculate the number=num1*10^0 + num2*10^1 and then count=0)
num1=0   #least digit
num2=0   #most digit
result=0 #total number
score=0
Lives=3
i=1  #for naming images

answersCount=0
over=cv2.imread('over.jpg',1)
second=cv2.imread('colors2.png',1)

skin_min=np.array((0,133,77))
skin_max=np.array((255,173,127))
cap=cv2.VideoCapture(0) #catch first webcam
while (1):
	cnt0=0
	#Capture the frame
	font=cv2.FONT_HERSHEY_SIMPLEX
	_, frame=cap.read()
	#change the color space
	ycrcb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)

	#threshold based on the upper and lower bound
	mask=cv2.inRange(ycrcb_image,skin_min,skin_max)
	cv2.imshow("Thresholding",mask)
	
	#Dilate
	kernel = np.ones((5,5), np.uint8)
	img_dilation1 = cv2.dilate(mask, kernel, iterations=2)
	#img_erosion1 = cv2.erode(img_dilation1, kernel, iterations=2)
	img_binary=img_dilation1
	cv2.imshow("Dilation",img_dilation1)
	#cv2.imshow("Ersion",img_erosion1)
	#cv2.imshow("Final",img_binary)
	#time counting part
	cnt1=cnt1+1
	if(cnt1%20==0):
		cnt2=cnt2+1
	cv2.putText(frame,str(cnt2),(0,40),font,1,(100,200,0),3)
	#here we need to take 10 frames and check the change in x :D
	if((cnt2>=6) & (cnt3<4)):
		cv2.putText(frame,"Start moving",(200,40),font,1,(0,0,255),3)
		img_contours = img_binary.copy()
		contours = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, \
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	 	#Find the largest contour and use it to compute the min enclosing circle
		center = None
		radius = 0
		if len(contours) > 0:
			c = max(contours, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			a,b,w,h = cv2.boundingRect(c)
			if M["m00"] > 0:
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				if radius < MIN_RADIUS:
					center = None
		if(cnt4==0):
			aOld=a
		else:
			mp[cnt4]=a-aOld#if negative means moving right
			aOld=a
		#Draw a green circle around the largest enclosed contour
		#print(a)
		cv2.rectangle(frame,(a,b),(a+w,b+h),(0,255,0),2)
		if center != None:
			cv2.circle(frame, center, int(round(radius)), (0, 0, 255),2)
		cnt4+=1
		if(cnt4%10==0):
			cnt3+=1
		cv2.putText(frame,str(cnt3),(300,70),font,1,(100,200,0),3)
#####################################################################################################################################
	else:
		img_contours = img_binary.copy()
		contours = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, \
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	 	#Find the largest contour and use it to compute the min enclosing circle
		center = None
		radius = 0
		if len(contours) > 0:
			c = max(contours, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			a,b,w,h = cv2.boundingRect(c)
			if M["m00"] > 0:
				center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				if radius < MIN_RADIUS:
					center = None
		if(cnt4==0):
			aOld=a
		else:
			mp[cnt4]=a-aOld#if negative means moving right
			aOld=a
		#Draw a green circle around the largest enclosed contour
		#print(a)
		cv2.rectangle(frame,(a,b),(a+w,b+h),(0,255,0),2)
		if center != None:
			cv2.circle(frame, center, int(round(radius)), (0, 0, 255),2)
#####################################################################################################################################
	#check the map
	for i in mp:#ymen el x bt2el, law msh negative yb2a mshet left
		if(mp[i]>0):
			cnt0+=1
	if((cnt3>=4) & (cnt0<=10)):
		if((cnt2<11)):
			print("You moved your hand right!")
			cv2.destroyWindow("original")
			cv2.destroyWindow("Dilation")
			cv2.destroyWindow("Ersion")
			cv2.destroyWindow("Thresholding")
			cv2.destroyWindow("Final")
			cv2.putText(frame,"You moved your hand right!",(100,100),font,1,(0,0,255),3)
#####################################################################first game starts################################################################
			
			startTime = time.time()
			while True:
				num1=randint(0, 9)
				num2=randint(0, 9)
				#print(num1)
				#print(num2)
				res=num1+num2
				firstDigit=res % 10
				res /=10
				res=int(res)
				secondDigit=res
				#print(firstDigit)
				#print(secondDigit)
				found1=False
				found2=False

				if firstDigit == 0:
					found1=True
				elif firstDigit == 1:
					found1=True
				elif firstDigit == 2:
					found1=True
				elif firstDigit == 3:
					found1=True
				elif firstDigit == 4:
					found1=True
				elif firstDigit == 5:
					found1=True
				if secondDigit == 0:
					found2=True
				elif secondDigit == 1:
					found2=True
				elif secondDigit == 2:
					found2=True
				elif secondDigit == 3:
					found2=True
				elif secondDigit == 4:
					found2=True
				elif secondDigit == 5:
					found2=True
				if((found1) and (found2)):
					break
			while(cap.isOpened()):

				endTime = time.time()
				#print(time.time())
				elapsed = int(endTime - startTime)
				#print("endtime")
				#print(endTime)	    
				ret, frame = cap.read()

				ycrcb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
			 
				skin_min=np.array((0,133,77))
				skin_max=np.array((255,173,127))
				mask=cv2.inRange(ycrcb_image,skin_min, skin_max)     #detect for hand
				median = cv2.medianBlur(mask,9) 
			 
				kernel = np.ones((5,5), np.uint8)
				img_dilation1 = cv2.dilate(median, kernel, iterations=1)
				img_erosion1 = cv2.erode(img_dilation1, kernel, iterations=1)
				img_dilation2 = cv2.dilate(img_erosion1, kernel, iterations=2)
				img_erosion2 = cv2.erode(img_dilation2, kernel, iterations=1) 
			 
				cv2.imshow('image', img_erosion2)
				_,contours,_ = cv2.findContours(img_dilation1,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
				      
				#print len(contours)
				drawing = np.zeros(img_dilation1.shape,np.uint8)     #draw contour in it
				cv2.drawContours(drawing, contours, -1, (255,255,255), 2)
			  
			# find contour with max area  as it is the contour of the hand (may be there another contours due to noise)
				if len(contours) > 0:
					cnt = max(contours, key = lambda x: cv2.contourArea(x))   #list of coordinates of max contour
					# finding convex hull
					hull = cv2.convexHull(cnt, returnPoints=False)    #false to return indeces of hull not coordinates
					#draw bounding box
					x,y,w,h = cv2.boundingRect(cnt)
					img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
					defects = cv2.convexityDefects(cnt, hull)
					count_defects = 0    #to know which number
					if elapsed == 10: # sec passed translate the number 
						startTime=endTime
						count += 1
			    
			    # applying Cosine Rule to find angle for all defects (between fingers)
			    # with angle > 90 degrees and ignore defects
						if defects is not None:
							for i in range(defects.shape[0]):
								s,e,f,d = defects[i,0]

								start = tuple(cnt[s][0])
								end = tuple(cnt[e][0])
								far = tuple(cnt[f][0])

								# find length of all sides of triangle
								a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
								b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
								c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

								# apply cosine rule here
								angle = math.degrees(math.acos((b**2 + c**2 - a**2)/(2*b*c)))

								# ignore angles > 105 and highlight rest with red dots
								if ((angle <= 86) and ( angle>26 )) :
									count_defects += 1
									cv2.circle(frame,far, 4, [255,0,0], -1)  
			     
								cv2.line(frame,start, end, [0,255,0], 2)
								cv2.imwrite("hand-" + str(i) + ".jpg",frame)
			    
								i += 1
			    
			 
			    # define actions required
							if h<150: 
								result += 0
			       # print (0)
			    # define actions required
							elif count_defects == 1:
								if count == 1:
									result += 20
								else:
									result += 2
							elif count_defects == 2:
								if count == 1:
									result += 30
								else:
									result += 3
							elif count_defects == 3:
								if count == 1:
			 						result += 40
								else:
									result += 4
							elif count_defects == 4:
								if count == 1:
									result += 50
								else:
									result += 5
							else:
								if count == 1:
									result += 10
								else:
									result += 1              
							print(result)
			#new if with 3 tabs
						if count == 2:
							cv2.putText(frame,str(result), (50, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2,(0,0,255), 2)
			       
							if result == num1+num2:
								score += 1
							else:
								Lives -= 1
			       #print results
							count=0
							result=0
							while True:
								num1=randint(0, 9)
								num2=randint(0, 9)
								res=num1+num2
								firstDigit=res % 10
								res /=10
								secondDigit=int(res)

								found1=False
								found2=False

								if firstDigit == 0:
									found1=True
								elif firstDigit == 1:
									found1=True
								elif firstDigit == 2:
									found1=True
								elif firstDigit == 3:
									found1=True
								elif firstDigit == 4:
									found1=True
								elif firstDigit == 5:
									found1=True
								if secondDigit == 0:
									found2=True
								elif secondDigit == 1:
									found2=True
								elif secondDigit == 2:
									found2=True
								elif secondDigit == 3:
									found2=True
								elif secondDigit == 4:
									found2=True
								elif secondDigit == 5:
									found2=True
								if((found1) and (found2)):
									break
			 ###################################
				if elapsed != 10 :
					if elapsed != 9 :
						cv2.putText(frame,str(elapsed), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2)
					if elapsed == 9 :
						cv2.putText(frame,"Go..", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 2)  
			##########################################
				cv2.putText(frame,str(num1) +" + " +str(num2)+" = ?", (370, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(0,0,250), 3)
				cv2.putText(frame,"SCORE : "+str(score), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 3)
				cv2.putText(frame,"LIVES : "+str(Lives), (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 3)
			################################
				if(Lives==0):
					cv2.putText(over,"FINAL SCORE : "+str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(150,0,200), 3)
					cv2.putText(over,"GAME OVER", (450, 350), cv2.FONT_HERSHEY_SIMPLEX, 1,(150,0,200), 3)  
					cv2.destroyWindow("frme")
					cv2.destroyWindow("contours")
					cv2.destroyWindow("image")
					cv2.imshow('GAME OVER',over)
					cv2.waitKey(0)
					flag=1
					break
			###############################
				cv2.imshow('frme',frame)
				if cv2.waitKey(1) & 0xFF == ord('c'):
					break
					cap.release()
					cv2.destroyAllWindows()
					flag=1
#####################################################################first game ends####################################################################
	if((cnt3>=4) & (cnt0>10) ):
		if((cnt2<11)):
			cv2.putText(frame,"You moved your hand left!",(100,100),font,1,(0,0,255),3)
			print("You moved your hand left!")
			cv2.destroyWindow("original")
			cv2.destroyWindow("Dilation")
			cv2.destroyWindow("Ersion")
			cv2.destroyWindow("Thresholding")
			cv2.destroyWindow("Final")
			cap.release()
			flag=1
			cv2.imshow('Second Game',second)
			cv2.waitKey(0)
#####################################################################second game starts################################################################
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


			im_in = cv2.imread("colors.png",cv2.IMREAD_ANYCOLOR)
			rows, cols,j = im_in.shape
			edges=cv2.Canny(im_in,100,200)
			h , w = edges.shape[:2]
			im_floodfill= edges.copy()
			mask = np.zeros((h+2, w+2),np.uint8)
			cv2.floodFill(im_floodfill, mask, (0,0), 255)
			im_floodfill_inv = cv2.bitwise_not(im_floodfill)
			thresh = cv2.threshold(im_floodfill_inv, 200, 255, cv2.THRESH_BINARY)[1]
			L = measure.label(thresh, neighbors=8, background=0)
			#print ("Number of components:", np.max(L))
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
				#print ("the shape no: ", label_number)
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
						#print("square")
						shapesDict['square']=shapesDict['square']+1
					else:
						#print("rectangle")
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
							#print("circle")
							shapesDict['circle']=shapesDict['circle']+1
						else:
							shapesDict['ellipse'] = shapesDict['ellipse']+1
							#print ("ellipse")
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
						print("size of hough: %d and count %d",dim1,count)
						houghDSU.clearClass()
					#to increase the number of shapes in the dictionary of shapes:
						if(count == 6):
							shapesDict['hexagon'] = shapesDict['hexagon']+1
							#print ("hexagon")
						elif(count == 4):
							shapesDict['rhombus'] = shapesDict['rhombus']+1
							#print ("rhombus")
						elif(count == 5):
							shapesDict['pentagon'] = shapesDict['pentagon']+1
							#print ("pentagon")
						else:
							shapesDict['triangle'] = shapesDict['triangle']+1
							#print ("triangle")
						#print ("number of edges is:", count,"while it was",dim1)
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
######################################################################################################################################################
			startTime = time.time()
			cap = cv2.VideoCapture(0)
			#shapesDict={'square':3,'rectangle':2,'triangle':4,'circle':0,'rhombus':1,'hexagon':1,'pentagon':2,'ellipse':2}
			while((cap.isOpened()) & (answersCount<8)):
				endTime = time.time()
				elapsed = int(endTime - startTime)
			    
				ret, frame = cap.read()
				if(answersCount==0):
					cv2.putText(frame,"Number of squares = ?", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
				elif (answersCount==1):
					cv2.putText(frame,"Number of rectangle = ?", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
				elif (answersCount==2):
					cv2.putText(frame,"Number of triangle = ?", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
				elif (answersCount==3):
					cv2.putText(frame,"Number of circle = ?", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
				elif (answersCount==4):
					cv2.putText(frame,"Number of rhombus = ?", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
				elif (answersCount==5):
					cv2.putText(frame,"Number of hexagon = ?", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
				elif (answersCount==6):
					cv2.putText(frame,"Number of pentagon = ?", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
				elif (answersCount==7):
					cv2.putText(frame,"Number of ellipse = ?", (140, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
				ycrcb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
			 
				skin_min=np.array((0,133,77))
				skin_max=np.array((255,173,127))
				mask=cv2.inRange(ycrcb_image,skin_min, skin_max)     #detect for hand
				median = cv2.medianBlur(mask,9) 
			 
				kernel = np.ones((5,5), np.uint8)
				img_dilation1 = cv2.dilate(median, kernel, iterations=1)
				img_erosion1 = cv2.erode(img_dilation1, kernel, iterations=1)
				img_dilation2 = cv2.dilate(img_erosion1, kernel, iterations=2)
				img_erosion2 = cv2.erode(img_dilation2, kernel, iterations=1) 
			 
				cv2.imshow('image', img_erosion2)
				_,contours,_ = cv2.findContours(img_dilation1,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
				      
				#print len(contours)
				drawing = np.zeros(img_dilation1.shape,np.uint8)     #draw contour in it
				cv2.drawContours(drawing, contours, -1, (255,255,255), 2)
				cv2.imshow('contours',drawing)
			  
			# find contour with max area  as it is the contour of the hand (may be there another contours due to noise)
				if len(contours) > 0:
					cnt = max(contours, key = lambda x: cv2.contourArea(x))   #list of coordinates of max contour
					# finding convex hull
					hull = cv2.convexHull(cnt, returnPoints=False)    #false to return indeces of hull not coordinates
					#draw bounding box
					x,y,w,h = cv2.boundingRect(cnt)
					img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
					defects = cv2.convexityDefects(cnt, hull)
					count_defects = 0    #to know which number
					if elapsed == 10: # sec passed translate the number 
						startTime=endTime
						count += 1
			    
			    # applying Cosine Rule to find angle for all defects (between fingers)
			    # with angle > 90 degrees and ignore defects
						if defects is not None:
							for i in range(defects.shape[0]):
								s,e,f,d = defects[i,0]

								start = tuple(cnt[s][0])
								end = tuple(cnt[e][0])
								far = tuple(cnt[f][0])

								# find length of all sides of triangle
								a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
								b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
								c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

								# apply cosine rule here
								angle = math.degrees(math.acos((b**2 + c**2 - a**2)/(2*b*c)))

								# ignore angles > 105 and highlight rest with red dots
								if ((angle <= 100) and ( angle>25 )) :
									count_defects += 1
									cv2.circle(frame,far, 4, [255,0,0], -1)  
			     
								cv2.line(frame,start, end, [0,255,0], 2)
								cv2.imwrite("hand-" + str(i) + ".jpg",frame)
			    
								i += 1
			    
							if h<150: 
								result += 0
							elif count_defects == 1:
								result += 2
							elif count_defects == 2:
								result += 3
							elif count_defects == 3:
								result += 4
							elif count_defects == 4:
								result += 5
							else:
								result += 1              
							print("result",result)

						cv2.putText(frame,str(result), (50, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2,(0,0,255), 2)
						#result=0
						print(answersCount)
						if(answersCount==0):
							if(result==shapesDict['square']):
								print("true")
							else:
								print("false")
					
						elif (answersCount==1):
							if(result==shapesDict['rectangle']):
									print("true")
							else:
									print("false")
						elif (answersCount==2):
							if(result==shapesDict['triangle']):
									print("true")
							else:
									print("false")
						elif (answersCount==3):
							if(result==shapesDict['circle']):
									print("true")
							else:
									print("false")
						elif (answersCount==4):
							if(result==shapesDict['rhombus']):
									print("true")
							else:
									print("false")
						elif (answersCount==5):
							if(result==shapesDict['hexagon']):
									print("true")
							else:
									print("false")
						elif (answersCount==6):
							if(result==shapesDict['pentagon']):
									print("true")
							else:
									print("false")
						elif (answersCount==7):
							if(result==shapesDict['ellipse']):
									print("true")
							else:
									print("false")
						answersCount=answersCount+1
						count=0
						if result == 5:
							score += 1
						result=0
			 ###################################
				if elapsed != 10 :
					if elapsed != 9 :
						cv2.putText(frame,str(elapsed), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(140,100,0), 1)
					if elapsed == 9 :
						cv2.putText(frame,"Go..", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(140,100,0),1)  
			##########################################
				#cv2.putText(frame,"Number of triangles = ?", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0), 2)
				#cv2.putText(frame,"SCORE : "+str(score), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
			################################
			###############################
				cv2.imshow('frme',frame)
				if cv2.waitKey(1) & 0xFF == ord('c'):
					break

######################################################################################################################################################
#####################################################################second game ends##################################################################
	#end:
	if(flag==1):
		break
		cap.release()
		cv2.destroyAllWindows()
	cv2.imshow("original",frame)
	if cv2.waitKey(1) & 0xFF == ord('c'):
		break
