#!/usr/bin/python
# -*- coding:utf8 -*-
'''
Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import os
import csv
import keras
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import cv2

batch_size = 16
num_classes = 4
epochs = 1

# pretreatment 预处理
# input image dimensions
img_rows, img_cols = 256, int(256*1.5)

def change(data, t, x, y):
    for i in range(3):
        for j in range(3):
            data[x+i][y+j] = t * 255
def changeImage(img, row):
    age = int(row[1])/100
    her = int(row[2])/3  # 0...3
    p = 0 if row[3] == 'FALSE' else 1
    change(img, age, 0, 0)
    change(img, her, 0, 3)
    change(img, p, 0, 6)


root_dir = '/Users/xuwang/project/AMLTC/HW03/train/'
rows = []
with open(f'{root_dir}feats.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        rows.append(row)
rows = rows[1:]
# 'id', 'age', 'HER2', 'P53', 'molecular_subtype'
x_train = []
y_train = []
for p in rows:
    name = p[0]
    y_index = int(p[4])-1
    y = [0, 0, 0, 0]
    y[y_index] = 1
    for pi in os.listdir(f'{root_dir}images/{name}'):
        pic = cv2.imread(f'{root_dir}images/{name}/{pi}')
        pic_rs = cv2.resize(pic, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
        pic = cv2.cvtColor(pic_rs, cv2.COLOR_RGB2GRAY)
        changeImage(pic, row)
        print(pic.shape)
        x_train.append(pic.tolist())
        y_train.append(y)

input_shape = (img_rows, img_cols, 1)

tmp = []
for i in range(len(x_train)):
    tmp.append((x_train[i], y_train[i]))
random.shuffle(tmp)
x_train = [i[0] for i in tmp]
y_train = [i[1] for i in tmp]
x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_train /= 255
x_test = x_train
print('x_train shape:', x_train.shape)

# build the neural net 建模型(卷积—relu-卷积-relu-池化-relu-卷积-relu-池化-全连接)
model = keras.models.load_model('/Users/xuwang/project/AMLTC/HW03/m.h5')

# train the model 训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

model.save('m1.h5')

# test the model 测试模型
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
