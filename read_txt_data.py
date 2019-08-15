import numpy as np
import os
import pandas as pd


import math
import codecs

import cv2
data_path = "rightdata1"





def read_data():
    Line = []
    Label = []
    for i in os.listdir(data_path)[:1]:
        line = []
        file_name, _ = os.path.splitext(i)
        #print(file_name)
        params = file_name.split("-")
        #print("parmas:{}".format(params))
        label = list(map(float, params[1:9:2]))
        #print("label:{}".format(label))
        file_path = os.path.join(data_path, i)
        Label.append(label)

        x,y,ux, uy, uz, w = [], [],[], [], [], []
        dx = 0.05
        dy = 0.05
        dz = 0.2
        gx = 80
        gy = 100
        gz = 3
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:
                #print("line:{}".format(line))
                b = line.split("\t")
                b = list(map(float, b))
                x.append(b[0]),y.append(b[1]),ux.append(b[2]),uy.append(b[3]),uz.append(b[4]),w.append(b[5])







        pic0 = np.zeros((gx, gy), dtype=np.float32)
        for j in range(len(x)):
            ix = int(math.floor(x[j] / dx + dx / 2) + gx / 2)
            iy = int(math.floor(y[j] / dy + dy / 2) + gy / 2)
            #print("ix:{}".format(ix))
            if (ix >= 1 and ix <= gx and iy >= 1 and iy <= gy):
                pic0[ix,iy] = pic0[ix,iy] + w[j]
        #print("pic0:{}".format(pic0))
        sum_pic = np.zeros((int(gx), ), dtype=np.float32)
        for n in range(int(gx)):
            num = np.mean(sum(pic0[n, :]))
            round_num = round(number=num,ndigits=3)
            sum_pic[n] = round_num
            print("sum_pic.shape:{}".format(sum_pic.shape))
        #Line.append(sum_pic)
        pic0 = sum_pic
        pic1 = sum_pic
        pic2 = sum_pic
        pic3 = sum_pic
        pic4 = sum_pic
        pic5 = sum_pic
        line.append(pic0)
        line.append(pic1)
        line.append(pic2)
        line.append(pic3)
        line.append(pic4)
        line.append(pic5)
        Line.append(line)




    return Line, Label



if __name__ == "__main__":
    read_data()

















