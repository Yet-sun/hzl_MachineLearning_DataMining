# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 03_生成和查看网格数据点.py
@time: 2019/3/18
@desc: 生成和查看网格数据点
'''

#生成[0, 1]之间的二维网格，水平方向5个点，数值方向3个点。作图查看效果
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,1,5)
y = np.linspace(0,1,3)

X, Y = np.meshgrid(x, y)  # meshgrid函数用两个坐标轴上的点在平面上画格

plt.plot(X, Y,
         color='red',  # 全部点设置为红色
         marker='.',  # 点的形状为圆点
         linestyle='')  # 线型为空，也即点与点之间不用线连接
plt.grid(True)
plt.show()