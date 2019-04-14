# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 09_矩阵进阶.py
@time: 2019/3/23
@desc: 求解线性方程组
'''

import numpy as np

A = np.mat("2 1 3;6 6 10;2 7 6")
a = np.array([2, 7, 6])

# 调用solve函数求解线性方程
x = np.linalg.solve(A, a)
print(x)

# 使用dot函数检查求得的解是否正确
print(np.dot(A, x))
