import cv2
import numpy as np
#video processing
MIN_RADIUS = 2
cnt1=0
cnt2=0
cnt3=0
cnt4=0
aOld=0
f=0
mp = {}
skin_min=np.array((0,133,77))
skin_max=np.array((255,173,127))
cap=cv2.VideoCapture(0) #catch first webcam
while (1):
	count=0
	#Capture the frame
	font=cv2.FONT_HERSHEY_SIMPLEX
	_, frame=cap.read()
	#change the color space
	ycrcb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)

	#threshold based on the upper and lower bound
	mask=cv2.inRange(ycrcb_image,skin_min,skin_max)
	cv2.imshow("Thresholding",mask)
	
	#erode and dilate
	kernel = np.ones((5,5), np.uint8)
	img_dilation1 = cv2.dilate(mask, kernel, iterations=2)
	img_erosion1 = cv2.erode(img_dilation1, kernel, iterations=2)
	img_binary=img_erosion1
	cv2.imshow("Dilation",img_dilation1)
	cv2.imshow("Ersion",img_erosion1)
	cv2.imshow("Final",img_binary)
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
	#print("out")
	for i in mp:#ymen el x bt2el, law msh negative yb2a mshet left
		if(mp[i]>0):
			count+=1
	if((cnt3>=4) & (count>10) ):
		if((cnt2<11)):
			cv2.putText(frame,"You moved your hand left!",(100,100),font,1,(0,0,255),3)
	if((cnt3>=4) & (count<=10)):
		if((cnt2<11)):
			cv2.putText(frame,"You moved your hand right!",(100,100),font,1,(0,0,255),3)
	cv2.imshow("original",frame)
	if cv2.waitKey(1) & 0xFF == ord('c'):
		break
cap.release()
cv2.destroyAllWindows()
