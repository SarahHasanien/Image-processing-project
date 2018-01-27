import cv2
import numpy as np
import math
import time
from random import randint

cap = cv2.VideoCapture(0)

startTime = time.time()
count=0   #to count how many times it enters if (if count == 2 then calculate the number=num1*10^0 + num2*10^1 and then count=0)
num1=0   #least digit
num2=0   #most digit
result=0 #total number
score=0
Lives=3
i=1  #for naming images

over=cv2.imread('over.jpg',1)
while True:
  num1=randint(0, 9)
  num2=randint(0, 9)
  '''print(num1)
  print(num2)'''
  res=num1+num2
  firstDigit=res % 10
  res /=10
  res=int(res)
  secondDigit=res
  '''print(firstDigit)
  print(secondDigit)'''
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
 elapsed = int(endTime - startTime)
    
 ret, frame = cap.read()
 #hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

 ycrcb_image=cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
 '''lower = np.array([2, 36, 127], dtype = "uint8")   #low h,s,v
 upper = np.array([12, 255, 243], dtype = "uint8") #high h,s,v'''
 
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
 
 if elapsed != 10 :
   if elapsed != 9 :
    cv2.putText(frame,str(elapsed), (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(140,100,0), 2)
   if elapsed == 9 :
    cv2.putText(frame,"Go..", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(140,100,0), 2)  
 cv2.putText(frame,str(num1) +" + " +str(num2)+" = ?", (370, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(100,100,250), 3)
 cv2.putText(frame,"SCORE : "+str(score), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(20,10,120), 3)
 cv2.putText(frame,"LIVES : "+str(Lives), (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(20,10,120), 3)

 if(Lives==0):
   cv2.putText(over,"FINAL SCORE : "+str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(150,0,200), 3)
   cv2.putText(over,"GAME OVER", (450, 350), cv2.FONT_HERSHEY_SIMPLEX, 1,(150,0,200), 3)  
   cv2.destroyWindow("frme")
   cv2.destroyWindow("contours")
   cv2.destroyWindow("image")
   cv2.imshow('GAME OVER',over)
   cv2.waitKey(0)
   break
 cv2.imshow('frme',frame)
 if cv2.waitKey(1) & 0xFF == ord('c'):
  break
	
