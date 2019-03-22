# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 06_向量.py
@time: 2019/3/22
@desc: 向量基本操作
'''


from numpy import *
import numpy as np

# （1）数组(行向量)
# 创建及基本运算；
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([5, 4, 3, 2, 1])
print("x1:", x1, "\nx2:", x2)
print("x1 * 5 + 2 =:", x1 * 5 + 2)
print("x2 * 5 + 2 =:", x2 * 5 + 2)
# 各元素分别计算x1 * 5 + 2
# 对应元素分别相乘

# （2）创建列向量（一维数组可视为一个行向量；如果要生成列向量，必须使用mx1的二维数组）；
# 直接创建列向量
x3 = np.array([[1], [2], [3], [4], [5]])
print("x3:\n", x3)
# 将列向量转成1xm的行向量
x4 = x3.reshape(1, len(x3))
print("x4:", x4)

# 将一个一维数组转成列向量
x5 = x1.reshape(len(x1), 1)
print("x5:\n", x5)
# 借助mat类，然后通过转置实现列向量
x6 = mat([1, 1, 0, 0])
print("x6:", x6)
print("x6_T:\n", x6.T)

# 使用reshape方法, 将原来的一维数组变成了mx1形式的二维数组
print("x7:\n",x2.reshape(len(x2),1))