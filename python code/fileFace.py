import cv2
# import pandas as pd
import numpy as np
import os
import re
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
file1 = open('identity_CelebA.txt',"r")
# print(file1.read())
# detection=str(file1.read()).split(' ')
d=file1.read().split('\n')
file1.close()
# file = open("re/" + m.split("\\")[-1],"w")
# try:
for f in d:
    # print(f)
#     center_x=0
#     center_y=0
#     w=0
#     h=0
#     x=0
#     y=0
    name, index=f.split(' ')
    print(name,"==============",index)
    img=cv2.imread("./img_align_celeba/"+str(name))
    print(img.shape)
    print("img_align_celeba/"+str(name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if not os.path.exists('class/'+str(index)):
            os.mkdir('class/'+str(index))
        cv2.imwrite("{fname}/{index}/{imgname}".format(fname = "class", index = index,imgname=name),img[y:y+h,x:x+w])
        
    #     label=int(detection[0])
    #     if label == 2 or label == 1 or label == 6:
    #         file.write("0 "+" "+str(detection[1])+" "+str(detection[2])+" "+str(detection[3])+" "+str(detection[4]))
    #         file.write("\n")
    # file.close() 
# except :
#     # file.close()
#     print("e")