
from tools import *
image_path = r"C:\Users\saikrishna\Desktop\octa digit ocr\train\1\image_0.jpg"
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np 
import matplotlib.pyplot as plt

# def find_display_contour(edge_img_arr):
#   display_contour = None
#   edge_copy = edge_img_arr.copy()
#   contours,hierarchy = cv2.findContours(edge_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#   top_cntrs = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

#   for cntr in top_cntrs:
#     peri = cv2.arcLength(cntr,True)
#     approx = cv2.approxPolyDP(cntr, 0.02 * peri, True)

#     if len(approx) == 4:
#       display_contour = approx
#       break

#   return display_contour
# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

image = cv2.imread(image_path)
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)
cv2.imshow("edge",edged)
cv2.waitKey(0)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
# print(cnts)
displayCnt=None
for cnt in cnts:
    # idx += 1
    # peri = cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    # print(len(approx))
    # if(len(approx)==6):
    #   displayCnt=approx
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
# print(displayCnt)
print("bye0")
cv2.imshow('img',image)
cv2.waitKey(0)


img = image

    # Apply image segmentation and extract digits
digits = histogram_of_pixel_projection(img)


for i in range(len(digits)):

    # Get digit
    digit = digits[i]

    # Make the image squared
    squared_digit = square(digit)

    # Resize the image
    resized_digit = cv2.resize(squared_digit, (20, 20), interpolation=cv2.INTER_AREA)
    cv2.imshow("im",resized_digit)























'''
image=cv2.resize(image,(150,150))

# print('size',image.shape)
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(image, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# cv2.imshow('thresh',thresh)
# cv2.waitKey(0)   
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# print(cnts)
digitCnts = []
print("ji")
# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+h)),(255,0,255),1)
    cv2.imshow("ch",image)
    cv2.waitKey(0)
    # if the contour is sufficiently large, it must be a digit
    if w >= 7 and h>=20:
        
        digitCnts.append(c)
    
    # digitCnts = contours.sort_contours(digitCnts[0][0])
    digits = []
    print('len',len(digitCnts))
    for c in digitCnts:
    # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)
        # define the set of 7 segments
        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)), # top-left
            ((w - dW, 0), (w, h // 2)), # top-right
            ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
            ((0, h // 2), (dW, h)), # bottom-left
            ((w - dW, h // 2), (w, h)), # bottom-right
            ((0, h - dH), (w, h))   # bottom
        ]
        on = [0] * len(segments)
        # print(segments)
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        #   # print(i,((xA, yA), (xB, yB)))
        # #     # extract the segment ROI, count the total number of
        # #     # thresholded pixels in the segment, and then compute
        # #     # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            # cv2.imshow("segroi",segROI)
            # cv2.waitKey(0)
            total = cv2.countNonZero(segROI)
        #   # print(total,'tot')
            area = (xB - xA) * (yB - yA)
        #   # print(area,'area')
        # #     # if the total number of non-zero pixels is greater than
        # #     # 50% of the area, mark the segment as "on"
            if total / float(area) > 0.5:
                on[i]= 1
            print('on',on)
            try:
                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(digit)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(image, str(digit), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.imshow("Output", image)
                cv2.waitKey(0)
            except Exception as e:
                print(e,'e')
'''
