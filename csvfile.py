import os
import cv2
import csv
l = []
import pandas as pd
l2 = []
f =  {48:0,49:1,50:2,51:3,52:4,53:5,54:6,55:7,56:8,57:9}
s = dict()
for filename in os.listdir(r'C:\Users\saikrishna\Desktop\octa digit ocr\train\new_cropped_3'):
    
    print(filename)
print(s)
for k,v in s.items():
    print("k: "+str(k)+" v: "+str(v))
'''
pairs = {'filename': l}#, 'digit': l2}

df = pd.DataFrame.from_dict(pairs)
print(df.head())

df.to_csv('mycsv.csv')

'''
