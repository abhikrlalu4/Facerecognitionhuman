# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sqlite3
def UpdateOrInsert(Id,l):
    con=sqlite3.connect('face.db')
    cmd='SELECT * FROM face WHERE id='+str(Id)
    cursor=con.execute(cmd)
    isExist=0
    for row in cursor:
        isExist=1
    if isExist==1:
        if len(l)==1:
            cmd="UPDATE face SET emotion='"+l[0]+"' WHERE id="+str(Id)
        elif len(l)>1:
            cmd="UPDATE face SET emotion='"+l[0]+"',"+"age='"+l[1]+"',ethencity='"+l[2]+"',gender='"+l[3]+"' WHERE id="+str(Id)
    else:
        if len(l)==1:
            cmd="INSERT INTO face(id,emotion) Values("+str(Id)+",'"+l[0]+"')"
        elif len(l)>1:
            cmd="INSERT INTO face(id,emotion,age,ethencity,gender) Values("+str(Id)+",'"+l[0]+"'"+",'"+l[1]+"'"+",'"+l[2]+"'"+",'"+l[3]+"')"
    con.execute(cmd)
    #print(cmd)
    con.commit()
    con.close()

import cv2

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
    


import json
data=open('Face_Recognition.json','r')
c=1
l=[]
n=1
f=1
NoneType=type(None)
for i in data:
    
    a=json.loads(i)
    a['contant']='a'+str(n)+'.jpg'
    print(a['annotation'])
    #print(a)
    if len(a['annotation'])>0:
        faces=[]
        for k in range(len(a['annotation'])):
            jepg=0
            l=a['annotation'][k]['label']
            if len(l)==0:
                continue
            while len(l)!=4:
                if len(l)>4:
                    l.pop(-1)
                else:
                    l.append('NULL')
            print(a['annotation'][k]['label'])
            UpdateOrInsert(c,l)
            
            path='face/'
            frame=cv2.imread(path+'a'+str(n)+'.jpg')
            if type(frame)==NoneType:
                jepg=1
                frame=cv2.imread(path+'a'+str(n)+'.jpeg')
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            point=a['annotation'][k]['points']
            faces.append([point[0]['x'],point[0]['y'],point[1]['x'],point[1]['y']])
            #faces=face_cascade.detectMultiScale(gray,1.05,10)
            print(faces)
            w=a['annotation'][k]['imageWidth']
            h=a['annotation'][k]['imageHeight']
            
            c+=1
        for x1,y1,x2,y2 in faces:
            x1,y1,x2,y2=int(x1*w),int(y1*h),int(x2*w),int(y2*h)
            if x2-x1>=40 or y2-y1>=40:
                if jepg==1:
                    cv2.imwrite('data/user.'+str(n)+'.'+str(f)+'.jpeg',gray[y1:y2,x1:x2])
                else:
                    cv2.imwrite('data/user.'+str(n)+'.'+str(f)+'.jpg',gray[y1:y2,x1:x2])
                gray=cv2.rectangle(gray,(x1,y1),(x2,y2),(255,0,0),3)
                #print(x1*w,y1*h,x2*w,y2*h)
            f+=1
                
        cv2.imshow('capturing',gray)
            
    n+=1
    
# making LBPH Model and training with face data

from PIL import Image
import numpy as np
import cv2
import os
recognizer=cv2.face.LBPHFaceRecognizer_create()
path='data/'
faces=[]
IDs=[]
def getImageWithID(path):
    imgPaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imgPath in imgPaths:
        faceImg=Image.open(imgPath)
        faceNP=np.array(faceImg,'uint8')
        faces.append(faceNP)
        cv2.imshow('face',faceNP)
        cv2.waitKey(1)
        ID=int(os.path.split(imgPath)[-1].split('.')[2])
        IDs.append(ID)
        #print(IDs)
    return faces,IDs
faces,ids=getImageWithID(path)
recognizer.train(faces,np.array(ids))
recognizer.save('trainingData.yml')
cv2.destroyAllWindows()


