# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 06_向量.py
@time: 2019/3/22
@desc: 向量基本操作
'''

# （1）数组(行向量)
# 创建及基本运算；
# x1 = np.array([1, 2, 3, 4, 5])
# x2 = np.array([5, 4, 3, 2, 1])
# 各元素分别计算x1 * 5 + 2
# 对应元素分别相乘

# （2）创建列向量（一维数组可视为一个行向量；如果要生成列向量，必须使用mx1的二维数组）；
# 直接创建列向量
# 将列向量转成1xm的行向量
# 将一个一维数组转成列向量
# 借助mat类，然后通过转置实现列向量
# 使用reshape方法, 将原来的一维数组变成了mx1形式的二维数组
