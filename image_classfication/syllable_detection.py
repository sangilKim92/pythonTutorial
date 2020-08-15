import numpy as np
import json
import os
import cv2
import re
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

path ='./01_handwriting_syllable_images/1_syllable/'

def readImg():
    train=[]
    for index,x in enumerate(os.listdir('./01_handwriting_syllable_images/1_syllable')):
        img=cv2.imread(path+x)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img = img.reshape(1,-1)
        img = img[:,:30000]
        print(img.shape)
        train.append(img)
        if index==100:
            train= np.array(train)
            train=train.reshape(101,30000)
            df=pd.DataFrame(train)
            df.to_csv('이미지파일.csv',sep=',')
            break

#readImg()

train=pd.read_csv('./이미지파일.csv', sep=',', header=0,index_col=0) #0 번째 행을 칼럼으로, 0번째 열을 인덱스로
Y= pd.read_csv('./target.csv',sep=',', header=0, index_col=0)
train=train.values
train=train.reshape(-1,300,100,1)
Y= Y.values.reshape(101)
Y= Y.tolist()
target={}
time=1
check=[]
for num in Y:
    if num not in target:
        target[num]=time
        check.append(time)
        time=time+1
target=to_categorical(check)

model= Sequential()
model.add(Conv2D(filters=32,kernel_size=(5,5),input_shape=(300,100,1)))
model.add(Flatten())
model.add(Dense(102, activation='softmax'))
model.compile(optimizer=RMSprop(lr=0.001),
 loss='categorical_crossentropy',
 metrics=['accuracy'])
model.fit(train,target,epochs=5)
test= train[1].reshape(1,300,100,1)
print(model.predict(test))
test= train[2].reshape(1,300,100,1)
print(model.predict(test))
test= train[3].reshape(1,300,100,1)
print(model.predict(test))
test= train[4].reshape(1,300,100,1)
print(model.predict(test))


def readJson():
    train=[]
    with open('./handwriting_data_info1.json' ,encoding='UTF8') as json_file:
        json_data=json.load(json_file)

    for index,x in enumerate(os.listdir('./01_handwriting_syllable_images/1_syllable')):
        for num in json_data['annotations']:
            if num['id']==re.sub(r'.png','',x):

                train.append(num['text'])

        if index==100:
            df=pd.DataFrame(train)
            df.to_csv('target.csv',sep=',')
            break

#readJson()