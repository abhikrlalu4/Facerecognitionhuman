# -*- coding: utf-8 -*-

import cv2
import sqlite3
import numpy as np

rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('trainingData.yml')
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
def getProfile(Id):
    con=sqlite3.connect('face.db')
    cmd='select * from face where id='+str(Id)
    d=con.execute(cmd)
    profile=None
    for row in d:
        profile=row
    con.close()
    return profile
        
def predict(path,n,a):
    frame=cv2.imread(path+'a'+str(n)+'.jpg')
    if type(frame)==NoneType:
        jepg=1
        frame=cv2.imread(path+'a'+str(n)+'.jpeg')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    for k in range(len(a['annotation'])):
            jepg=0
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
    for x1,y1,x2,y2 in faces:
        x1,y1,x2,y2=int(x1*w),int(y1*h),int(x2*w),int(y2*h)
        print(x1,x2,y1,y2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0.255,0),1)
        Id,conf=rec.predict(gray[y1:y2,x1:x2])
        profile=getProfile(Id)
        k=0
        print(Id)
        if profile is not None:
            for i in range(1,len(profile)):
                if profile[i]!='NULL':
                    cv2.putText(frame,profile[i],(x1,y2+k),font,1,(0,255,0),3)
                    k+=20
    cv2.imshow('face',frame)

import json
data=open('Face_Recognition.json','r')
faces=[]
c=1
NoneType=type(None)
for i in data:
    a=json.loads(i)
    if c==100:
        break
    c+=1
predict('face/',c,a)


facedetect=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
video=cv2.VideoCapture(0)
while True:
    check,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.2,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0.255,0),4)
        Id,conf=rec.predict(gray[y:y+h,x:x+w])
        profile=getProfile(Id)
        k=0
        print(Id)
        if profile is not None:
            for i in range(1,len(profile)):
                if profile[i]!='NULL':
                    cv2.putText(frame,profile[i],(x,y+h+k),font,1,(0,255,0),3)
                    k+=20
    cv2.imshow('face',frame)
    if cv2.waitKey(1)==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
