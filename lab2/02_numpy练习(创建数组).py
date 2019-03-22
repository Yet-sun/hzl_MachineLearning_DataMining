# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 02_numpy练习(创建数组).py
@time: 2019/3/18
@desc: numpy练习(创建数组)
'''

import numpy as np

# (1)使用单一值创建数组
# 创建具有10个元素的全0值数组
my_array1 = np.zeros((10))
print("my_array1:\n", my_array1)

# 创建2x3的全0值二维数组
my_array2 = np.zeros((2, 3))
print("my_array2:\n", my_array2)

# 创建2x3的全0值二维整数数组
my_array3 = np.zeros((2, 3), dtype=np.int)
print("my_array3:\n", my_array3)

# 创建2x3的全1值二维数组
my_array4 = np.ones((2, 3))
print("my_array4:\n", my_array4)

# 创建2x3的二维数组，每个元素值都是5
my_array5 = np.full((2, 3), 5)
print("my_array5:\n", my_array5)

# 创建3x3的二维数组，并且主对角线上元素都是1
my_array6 = np.eye((3))
print("my_array6:\n", my_array6)

# 创建mxn的二维数组，并且主对角线上元素都是1
m, n = eval(input())  # 用逗号分割，一次读入两个值存入m,n
print(m, n)
my_array7 = np.eye(m, n, k=1)  # mxn的二维数组,主对角线上元素都是1
print("my_array7:\n", my_array7)

# 创建2x3的二维数组，不指定初始值
my_array8 = np.random.random((2, 3))
print("my_array8:\n", my_array8)

# (2)从现有数据初始化数组；
# 创建5个元素的一维数组，初始化为1, 2, 3, 4, 5
my_array9 = np.array(5)
my_array9 = [1, 2, 3, 4, 5]
print("my_array9:\n", my_array9)

# 创建2x3的二维数组，用指定的元素值初始化
my_array10 = np.array([[1, 2, 3], [1, 2, 3]])
print("my_array10:\n", my_array10)

# a是mxn数组，根据a的维度生成mxn的全0值数组b
a = np.array([[1, 2, 3], [4, 2, 3], [5, 2, 3], [3, 2, 3]])
print(a.shape)
b = np.zeros(a.shape)
print("array b:\n", b)

# 以指定的主对角线元素创建对角矩阵
my_array12 = np.diag((5, 4, 3))  # 指定主对角线元素分别为5、4、3
print("my_array12:\n", my_array12)

# (3)将指定数值范围切分成若干份，形成数组；
# 根据指定的间距，在[m, n)区间等距切分成若干个数据点，形成数组
my_array13 = np.arange(2, 14, 2)
print("my_array13:\n", my_array13)

# 根据指定的切分点数量，在[m, n]区间等距切分成若干个数据点，形成数组
my_array14 = np.linspace(1, 10, 10)
print("my_array14:\n", my_array14)

# 生成指数间隔(而非等距间隔)的数组
import math

my_array15 = np.arange(2, 20, math.exp(2))  # math.exp(x)表示以e为底，x为指数
print("my_array15:\n", my_array15)

# 生成网格数据点
import matplotlib.pyplot as plt

x = np.array([0, 1, 2])
y = np.array([0, 1])

X, Y = np.meshgrid(x, y)  # meshgrid函数用两个坐标轴上的点在平面上画格

plt.plot(X, Y,
         color='red',  # 全部点设置为红色
         marker='.',  # 点的形状为圆点
         linestyle='')  # 线型为空，也即点与点之间不用线连接
plt.grid(True)
plt.show()

# （4）数组的引用与拷贝
# 使数组b与数组a共享同一块数据内存(数组b引用数组a)
array_a = np.array([1, 2, 3, 4])
array_b = array_a
print("id of array_a:", id(array_a))
print("id of array_b:", id(array_b))
print(array_a is array_b)

# 将数组a的值做一份拷贝后再赋给b，a和b各自保留自己的数据内存
array_a = np.array([1, 2, 3, 4])
array_b = array_a.copy()  # copy()：深拷贝
print("id of array_a:", id(array_a))
print("id of array_b:", id(array_b))
print(array_a is array_b)
