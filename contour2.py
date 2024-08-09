from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure

import cv2

import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec

image_path = r"C:\Users\saikrishna\Desktop\octa digit ocr\train\1\image_7.jpg"
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
img2 = image
img2 = cv2.resize(img2,(100,100))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow('imj',img2)
img2 
cv2.waitKey(0)
print(img2)
print(img2.shape)

'''
imging = image


def load_img(path):

    bgr = cv2.imread(path)

    img = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))[2]

    return img,bgr

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def thresh_img(img,size,offset):

    T = threshold_local(img, size, offset=offset, method="gaussian")
    thresh = (img > T).astype("uint8") * 255

    return thresh


def correct_thresh(img,l,b,r,u):

    #replaces white(255) pixels in background of binarized image
    #with black(0) pixels
    #l,b,r,u are the widths along the left,bottom,up,right

    h,w = img.shape

    #left
    img[0:h,0:l] = 0
    
    #right
    img[0:h,w-1-r:-1] = 0

    #up
    img[0:u,0:w] = 0

    #bottom
    img[h-1-b:-1,0:w] = 0

    return img


def upper_border_cut(horizontal_profile):

    #finds the upper bound row index of an edge image
    #from its horizontal profile

    start_span = horizontal_profile[:5]
    min_px = min(start_span)

    if min_px == 0:

        start = len(start_span) - 1 - start_span[::-1].index(0)

    else:

        return 0



    

    #check_span = horizontal_profile[start:]

    for row_index in range(start,len(horizontal_profile)):

        if horizontal_profile[row_index+1] > 0:

            upper_bound =  row_index

            break

    return upper_bound


def left_border_cut(vertical_profile):

    #finds the left bound row index of an edge image
    #from its vertical profile

    start_span = vertical_profile[:5]
    min_px = min(start_span)

    if min_px == 0:

        start = len(start_span) - 1 - start_span[::-1].index(0)

    else:

        return 0



    

    #check_span = vertical_profile[start:]

    for column_index in range(start,len(vertical_profile)):

        if vertical_profile[column_index+1] > 0:

            left_bound =  column_index

            break

    return left_bound



def choose_candidates(cut_candidates,row_width_thresh):

    #chooses the best upper bound out of proposed upper bound
    #chooses the one nearest to the central index

    cand_above_mid = max(list(filter(lambda x: x <= row_width_thresh, cut_candidates)),default=-1)

    cand_below_mid = min(list(filter(lambda x: x > row_width_thresh, cut_candidates)),default=-1)

    if (cand_above_mid !=-1 and cand_below_mid !=-1):

        dist_above = row_width_thresh - cand_above_mid
        dist_below = cand_below_mid - row_width_thresh

        if (dist_above >= dist_below):

            u = cand_below_mid
            l = cand_below_mid + 1

        else:

            u = cand_above_mid 
            l = cand_above_mid + 1

    elif (cand_above_mid == -1):

        u = cand_below_mid
        l = cand_below_mid + 1

    elif (cand_below_mid == -1):

        u = cand_above_mid
        l = cand_above_mid + 1


    return (u,l)







def mid_cut(img,horizontal_profile,row_width_thresh):

    #returns 2 line separation row indices
    #first one corresponds to the lower bound of the 1st line of text
    #second index corresponds to the upper bound of the 2 line of text

    #row_width is the num of pixels we're checking for each half

    height = img.shape[0]
    mid = height//2

    scale_factor = mid - row_width_thresh

    check_span = horizontal_profile[mid-row_width_thresh:mid+row_width_thresh+1]

    
    

    min_px = min(check_span)

    if (min_px == 0):

        #index corresponding to lower bound of 1st line
        up = check_span.index(min_px) + scale_factor

        #check whether there is any continuous stride of black rows

        for i in range(up,len(horizontal_profile)):

            if horizontal_profile[i+1] > 0:

                #index corresponding to upper bound of 2nd line
                low = i

                break

        up = up - scale_factor
        low = low - scale_factor

    else:

        #checking for valleys
        cut_candidates = []

        for i in range(1,len(check_span)-1):

            if (check_span[i]<check_span[i+1] and check_span[i]<check_span[i-1]):
                
                cut_candidates.append(i)

        if len(cut_candidates) > 0:
            #atleast one valley

            up,low = choose_candidates(cut_candidates,row_width_thresh)

        else:

            #if no valley,check for pseudo-valleys

            for i in range(1,len(check_span)-1):

                if (check_span[i]<=check_span[i+1] and check_span[i]<=check_span[i-1]):
                    
                    cut_candidates.append(i)

            if len(cut_candidates) > 0:
                #atleast one valley

                up,low = choose_candidates(cut_candidates,row_width_thresh)

            else:

                #if no pseudo valley ,the row with min intensity is our candidate

                up = check_span.index(min_px)
                low = up + 1
    
    up = up + scale_factor
    low = low + scale_factor

    return (up,low)            


def eliminate_vertical_borders(thresh,vertical_profile):

    #given an image, this subroutine eliminates the vertical boundaries

    #rotated_edges = cv2.flip( img, 1 )

    height,width = thresh.shape


    
    #left bound index
    left = left_border_cut(vertical_profile)

    #vertical profile of rotated edge img
    vertical_profile.reverse()
    #vertical_profile = cv2.reduce(rotated_edges, 0, cv2.REDUCE_SUM, dtype=cv2.CV_64F).flatten()
    #vertical_profile = list(map(int,vertical_profile.tolist()))

    #right bound index
    right = width - left_border_cut(vertical_profile) - 1

    #cropping img,removing right and left borders
    print('left,right',left,right)
    
    vertical_crop = thresh[0:height,left:right]

    return vertical_crop



def eliminate_horizontal_borders(thresh,horizontal_profile):

    #given an image, this subroutine eliminates the horizontal boundaries

    #rotated_edges = cv2.flip( img, 1 )

    height,width = thresh.shape


    
    #up bound index
    up = upper_border_cut(horizontal_profile)

    #horizontal profile of rotated edge img
    horizontal_profile.reverse()
    #vertical_profile = cv2.reduce(rotated_edges, 0, cv2.REDUCE_SUM, dtype=cv2.CV_64F).flatten()
    #vertical_profile = list(map(int,vertical_profile.tolist()))

    #bottom bound index
    bottom = height - upper_border_cut(horizontal_profile) - 1

    #cropping img,removing up and bottom borders
    print('up,bottom',up,bottom)
    
    horizontal_crop = thresh[up:bottom,0:width]

    return horizontal_crop


def cut_img_into_two_halves(horizontal_crop,horizontal_profile):

    #takes in horizontally cropped edge img
    # and returns two image halves




    up,low = mid_cut(horizontal_crop,horizontal_profile,2)

    print('up,low',up,low)
    upper_half = horizontal_crop[0:up,0:horizontal_crop.shape[1]]

    lower_half = horizontal_crop[low:horizontal_crop.shape[0],0:horizontal_crop.shape[1]]

    return (upper_half,lower_half)



def eliminate_horizontal_borders(upper_half,lower_half,horizontal_profile):

    #upper bound of upper half
    up = upper_border_cut(horizontal_profile)

    upper_half = upper_half[up:,0:]
    #lower bound of lower half
    horizontal_profile.reverse()
    print(horizontal_profile)

    lower_half_ht = lower_half.shape[0]
    bottom = lower_half_ht - left_border_cut(horizontal_profile) - 1

    horizontal_profile = lower_half[0:lower_half_ht - bottom,0:]

    return (upper_half,lower_half)


def remove_boundaries_and_cut_in_half(edges,thresh):

    #main driver script
    #given an grayscale image, trims its borders and divides the img into 2 halves

    #img_ = img.copy()
    #edges = cv2.Canny(img,30,90)

    #computing the vertical profile of edge img
    #vertical_profile = cv2.reduce(edges, 0, cv2.REDUCE_SUM, dtype=cv2.CV_64F).flatten()
    #vertical_profile = list(map(int,vertical_profile.tolist()))

    #vertical borders trimmed img
    #vertical_crop = eliminate_vertical_borders(edges,vertical_profile)


    #computing horizontal profile of vertically trimmed edge img
    horizontal_profile = cv2.reduce(edges, 1, cv2.REDUCE_SUM, dtype=cv2.CV_64F).flatten()
    horizontal_profile = list(map(int,horizontal_profile.tolist()))

    #trimming horizontal borders
    horizontal_crop_thresh = eliminate_horizontal_borders(thresh,horizontal_profile)
    horizontal_crop_edge = eliminate_horizontal_borders(edges,horizontal_profile)

    #dividing the horizontally cropped img into 2 halves
    upper_half_thresh,lower_half_thresh = cut_img_into_two_halves(horizontal_crop_thresh,horizontal_profile)
    upper_half_edge,lower_half_edge = cut_img_into_two_halves(horizontal_crop_edge,horizontal_profile)

    #now trim vertical borders of each half

    #computing the vertical profile of upper half
    up_vertical_profile = cv2.reduce(upper_half_edge, 0, cv2.REDUCE_SUM, dtype=cv2.CV_64F).flatten()
    up_vertical_profile = list(map(int,up_vertical_profile.tolist()))

    #trimming vertical borders of upper half
    upper_half_thresh = eliminate_vertical_borders(upper_half_thresh,up_vertical_profile)
    upper_half_edge = eliminate_vertical_borders(upper_half_edge,up_vertical_profile)

    #computing the vertical profile of lower half
    low_vertical_profile = cv2.reduce(lower_half_edge, 0, cv2.REDUCE_SUM, dtype=cv2.CV_64F).flatten()
    low_vertical_profile = list(map(int,low_vertical_profile.tolist()))

    #trimming vertical borders of lower half
    lower_half_thresh = eliminate_vertical_borders(lower_half_thresh,low_vertical_profile)
    lower_half_edge = eliminate_vertical_borders(lower_half_edge,up_vertical_profile)



    

    #upper_half,lower_half = eliminate_horizontal_borders(upper_half,lower_half,horizontal_profile)

    return (upper_half_thresh,lower_half_thresh,upper_half_edge,lower_half_edge)



if __name__=="__main__":

    #path = "./imgs/positive08910.bmp"
    img = imging

    for imname in range(1):
        
        img = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
        img = image_resize(img,height=90)
        #bgr = image_resize(bgr,height=90)
        
        print(img.shape)

        img_gray = img.copy()

        edges = cv2.Canny(img,90,200)
        thresh = thresh_img(img_gray,5,1)

        upper_half_thresh,lower_half_thresh,upper_half_edge,lower_half_edge = remove_boundaries_and_cut_in_half(edges,thresh)

        print(upper_half_thresh.shape,lower_half_thresh.shape)

        #cv2.imshow('edge',edges)
        #cv2.imshow('thresh',thresh)
        #cv2.imshow('up',upper_half_thresh)
        cv2.imshow('low',lower_half_thresh)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        up_cnts,histr = cv2.findContours(upper_half_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #print(cnts)
        #up_cnts = sorted(up_cnts, key=cv2.contourArea, reverse=True)
        
        up_cnts_areas = list(map(cv2.contourArea,up_cnts))

        #up_cnts_areas_mean = sum(up_cnts_areas)/len(up_cnts_areas)

        low_cnts,histr = cv2.findContours(lower_half_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in low_cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),1)
        cv2.imshow("ch",img)
        for c in up_cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),1)
        cv2.imshow("ch",img)


        low_cnts_areas = list(map(cv2.contourArea,low_cnts))

        cnts_areas = up_cnts_areas+low_cnts_areas

        cnts_areas_mean = sum(cnts_areas)/len(cnts_areas)
        
        print(cnts_areas)
        print(cnts_areas_mean)
        #print(cnts)
        low_cnts = sorted(low_cnts, key=cv2.contourArea, reverse=True)
        print("len",len(low_cnts))
        for low_cnt in low_cnts:
            
            
            area = cv2.contourArea(low_cnt)
            print(area)
            
            if(True):
                
                print('area',area)

                #epsilon = 0.02*cv2.arcLength(cnt,True)
                #approx = cv2.approxPolyDP(cnt,epsilon,True)

                
                x,y,w,h = cv2.boundingRect(low_cnt)
                print('h,w',h,w)
                cropped_char = lower_half_thresh[y:y+h,x:x+w]
                
                #cropped_char = cv2.cvtColor(cropped_char, cv2.COLOR_BGR2GRAY)
                
                #print(img.shape)
                #img = thresh_img(img,11,1)
                
                cv2.imshow('cnt',cropped_char)
                cv2.waitKey(0)

                cv2.destroyAllWindows()

        for up_cnt in up_cnts:
            
            
            area = cv2.contourArea(up_cnt)
            print(area)
            
            if(area > cnts_areas_mean):
                
                print('area',area)

                #epsilon = 0.02*cv2.arcLength(cnt,True)
                #approx = cv2.approxPolyDP(cnt,epsilon,True)

                
                x,y,w,h = cv2.boundingRect(up_cnt)
                print('h,w',h,w)
                cropped_char = upper_half_thresh[y:y+h,x:x+w]
                
                #cropped_char = cv2.cvtColor(cropped_char, cv2.COLOR_BGR2GRAY)
                
                #print(img.shape)
                #img = thresh_img(img,11,1)
                
                cv2.imshow('cnt',cropped_char)
                cv2.waitKey(0)

                cv2.destroyAllWindows()




    


'''

















        





