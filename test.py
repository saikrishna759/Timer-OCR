import cv2
import os
import numpy as np 
import imutils
from pytesseract import Output
import pytesseract
# Reading the input image
for i,imo in enumerate(os.listdir('3')):
        image = cv2.imread('3/'+str(imo)) 
          
        image = imutils.resize(image, height=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200, 255)
        #cv2.imshow("edge",edged)
        #cv2.waitKey(0)


        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


        for cnt in cnts:
                # idx += 1
                # peri = cv2.arcLength(cnt, True)
                # approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                # print(len(approx))
                # if(len(approx)==6):
                # 	displayCnt=approx
                #if cv2.contourArea(conto)
                x,y,w,h = cv2.boundingRect(cnt)
                roi=image[y:y+h,x:x+w]
                if(len(roi)==326):
                        print(x,y,w,h)
                        image=roi[y:x,:y+w]
                        break
                elif(len(roi)==215 or len(roi)==144):
                        print(x,y,w,h)
                        image=roi[w//2:y-w,:]
                        break
                elif(len(roi)==151):
                        print(x,y,w,h)
                        image=roi[w-(h//8):-w//2,-h//2:y+h]
                        break
        print(image.shape)

        #cv2.imshow("img",image)
        #cv2.waitKey(0)
        image=cv2.resize(image,(150,150))
        kernel = np.ones((4,4), np.uint8) 
        img_erosion = cv2.erode(image, kernel, iterations=1)
        #cv2.imshow("erode",img_erosion)
        #cv2.waitKey(0)

        image = cv2.cvtColor(img_erosion, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(image, 0, 255,cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Taking a matrix of size 5 as the kernel 

        # # The first parameter is the original image, 
        # # kernel is the matrix with which image is  
        # # convolved and third parameter is the number  
        # # of iterations, which will determine how much  
        # # you want to erode/dilate a given image.  
        # img_erosion = cv2.erode(thresh, kernel, iterations=2) 
        # img_dilation = cv2.dilate(img, kernel, iterations=1) 

        # cv2.imshow('Input', img) 
        # cv2.imshow('Erosion', img_erosion) 
        # cv2.imshow('Dilation', img_dilation) 
          
        #cv2.waitKey(0)

        cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
        # print(len(cnts))
        contours=[]
        for c in cnts:
            print(cv2.contourArea(c))
            if cv2.contourArea(c) > 1000 or cv2.contourArea(c) < 50:
                    continue
            (x, y, w, h) = cv2.boundingRect(c)
            # print(x,y,w,h)
            contours.append([x,y,w,h])
            # print(contours)
            # print('*'*10)
            #cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+h)),(255,0,0),2)
            #cv2.imshow("ch",image)
            #cv2.waitKey(0)
        contours=sorted(contours)
        print(contours)
        c=0
        j = 0
        while(c<len(contours)-1):
            

            
            if(abs(contours[c][0]-contours[c+1][0])<=1):
                print('con1',contours[c])
                (x, y, w, h) = contours[c][0],contours[c][1],contours[c][2],contours[c][3]
                c+=1
                cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+(2*h))),(255,0,0),2)
                im=image[y:y+2*h,x:x+w]
                #cv2.imshow("ch",im)
                #cv2.waitKey(0)
            # elif((contours[c][0]+contours[c][2])==(contours[c+1][0]+contours[c+1][2])):
            elif(abs((contours[c][0]+contours[c][2])-(contours[c+1][0]+contours[c+1][2]))<=1):
                 #(contours[c+1][1] > contours[c][1]) and  ((contours[c][1] + contours[c][3]) < contours[c+1][1])):
                
                print('con2',contours[c],contours[c+1])
                (x, y, w, h) = contours[c][0],contours[c][1],contours[c][2],contours[c][3]
                c+=1
                cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+(2*h))),(255,0,0),2)
                im=image[y:y+2*h,x:x+w]
                #cv2.imshow("ch",im)
                #cv2.waitKey(0)
            else:
                print('con3',contours[c])
                (x, y, w, h) = contours[c][0],contours[c][1],contours[c][2],contours[c][3]
                cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+h)),(255,0,0),2)
                im=image[y:y+h,x:x+w]
                #cv2.imshow("ch",im)
                #cv2.waitKey(0)
            cv2.imwrite("C:\\Users\\saikrishna\\Desktop\\octa digit ocr\\train\\cropped_3\\"+str(i)+"_"+str(j)+".jpg",im)
            j += 1
            c+=1
        
                                            

