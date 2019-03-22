# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 04_三角形斜边长.py
@time: 2019/3/18
@desc: 设计一个求直角三角形斜边长的函数: 两条直角边为参数，求斜边长。
'''

import math


def line_lenth(a, b):
    c = math.sqrt(a ** 2 + b ** 2)
    return c


if __name__ == '__main__':
    a=float(input("please input a:"))
    b=float(input("please input b:"))

    c=line_lenth(a,b)
    print("c =",c)