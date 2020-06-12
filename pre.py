import keras
import os
import csv
import cv2
import numpy as np


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

img_rows, img_cols = 256, int(256*1.5)

root_dir = '/Users/xuwang/project/AMLTC/HW03/test/'
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
    for pi in os.listdir(f'{root_dir}images/{name}'):
        pic = cv2.imread(f'{root_dir}images/{name}/{pi}')
        pic_rs = cv2.resize(pic, (img_cols, img_rows), interpolation=cv2.INTER_AREA)
        pic = cv2.cvtColor(pic_rs, cv2.COLOR_RGB2GRAY)
        changeImage(pic, row)
        print(pic.shape)
        x_train.append(pic.tolist())

x_train = np.array(x_train).astype('float32')
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

model = keras.models.load_model('/Users/xuwang/project/AMLTC/HW03/m.h5')
y_pred = model.predict(x_train, batch_size=1)

y_pred = y_pred.tolist()
import csv

with open("test.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(rows)):
        writer.writerow([rows[i][0], y_pred[i].index(max(y_pred[i]))+1])
