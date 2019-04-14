# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 07_矩阵.py
@time: 2019/3/23
@desc: 矩阵基础操作
'''

from numpy import *
import numpy as np

# （1）mat形式的矩阵
# 创建矩阵
a1 = mat([[2, 3, 4], [5, 8, 2]])
print("a1:\n", a1)

# 矩阵与向量乘积
a2 = mat([2, 1, 6]).reshape(3, 1)
print("a2:\n", a2)
print("a1*a2=:\n", a1 * a2)

# 矩阵与矩阵乘积
a3 = mat([[2, 1], [1, 0], [6, 7]])
print("a3:\n", a3)
print("a1*a3=:\n", a3 * a1)

# （2）数组(array)形式的矩阵
# 创建矩阵和向量（列向量、行向量）
a4 = np.array([1, 2, 3, 4, 5])  # 行向量
a5 = a4.reshape(5, 1)  # 列向量
print("a4:\n", a4)
print("a5:\n", a5)

# 生成特殊矩阵
a6 = np.zeros((2, 3))  # 全零矩阵
a7 = np.ones([2, 3])  # 全一矩阵
a8 = np.identity(3)  # 主对角线全为以1
a9 = np.diag([1, 2, 3, 4])  # 生成完整的奇异值矩阵
print("a6:\n", a6, "\na7:\n", a7, "\na8:\n", a8, "\na9:\n", a9)

# （3）关于"*"操作符
# 数组/矩阵与标量进行计算：数组/矩阵中的每个元素与标量分别计算
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([[1, 2, 3], [4, 5, 6]])
x3 = np.mat([[1, 2, 3], [4, 5, 6]])
r1 = x1 * 2
r2 = x2 * 2
r3 = x3 * 2
print("x1:\n", x1, "\nx2:\n", x2, "\nx3:\n", x3)
print("r1 = x1 * 2 =\n", r1, "\nr2 = x2 * 2 =\n", r2, "\nr3 = x3 * 2 =\n", r3)

# mat形式矩阵/向量之间的计算：完全遵照矩阵乘法进行
A1 = np.mat([[1, 2, 3], [4, 5, 6]])
A2 = np.mat([[1, 2], [3, 4], [5, 6]])
x1 = np.mat([[1], [2], [3]])
r1 = A1 * A2
r2 = A1 * x1
print("A1:\n", A1, "\nA2:\n", A2, "\nx1:\n", x1)
print("r1 = A1 * A2 =\n", r1, "\nr2 = A1 * x1 =\n", r2)

# 数组形式矩阵/向量之间的计算：不能当成矩阵乘法来处理，而是按照对应元素相乘，判断r1、r2、r3、r4、r5、r6、r7的正误，给出原因。
A1 = np.array([[1, 2, 3], [4, 5, 6]])
A2 = np.array([[1, 2], [3, 4], [5, 6]])
x1 = np.array([[1], [2], [3]])
x2 = np.array([1, 2, 3])
print("A1:\n", A1, "\nA2:\n", A2, "\nx1:\n", x1, "\nx2:\n", x2)

r1 = A1 * x2
r2 = x2 * A1
print("r1 = A1 * x2 =\n", r1, "\nr2 = x2 * A1 =\n", r2)  # r1,r2正确，(2,3)×(3,2)

# r3 = A1 * A2
# r4 = A1 * x1
# r5 = x1 * A1
# r3,r4,r5错误,错误原因：数组大小“不一致”，且r5即使变成矩阵运算也是不对的，(3,1) (2,3)的两个矩阵无法相乘，只能是r4的形式
# 错误,错误原因：数组大小“不一致”
# r3:operands could not be broadcast together with shapes (2,3) (3,2)
# r4:operands could not be broadcast together with shapes (2,3) (3,1)
# r5:operands could not be broadcast together with shapes (3,1) (2,3)
# 正确的乘法操作应该是：
r3 = A1.dot(A2)
r4 = A1.dot(x1)
print("r3 = A1.dot(A2)=\n", r3, "\nr4 = A1.dot(x1)\n", r4)

r6 = x1 * x2
r7 = x2 * x1
print("r6 = x1 * x2 =\n", r6, "\nr7 = x2 * x1 =\n", r7)  # r6,r7正确

# （4）关于dot函数
# 数组/矩阵与标量进行计算：数组/矩阵中的每个元素与标量分别计算，对于a.dot(b)：
# •若a, b都是一维数组，执行内积和操作(即对应元素乘积再求和)，同时要求a和b的维数相同
# •若a, b都是二维数组，执行矩阵乘法操作。如果a、b在执行矩阵乘法时行列数不匹配，则产生错误
# •若a为mxn数组，b是一维数组，则b必须是n个元素。a的每行和b分别进行内积和
# •若a为具有m个元素的一维数组，b是二维数组，则b必须具有m行。a和b的每列分别进行内积和
# •np.dot(a, b)和a.dot(b)具有相同的效果

# 判断r1、r2、r3、r4、r5、r6、r7的正误，结论错误的给出原因，结论正确的给出结果，结果放到一个数组中。
A1 = np.array([[1, 2, 3], [4, 5, 6]])
A2 = np.array([[1, 2], [3, 4], [5, 6]])
x1 = np.array([[1], [2], [3]])
x2 = np.array([1, 2, 3])
x3 = np.array([1, 2])

r1 = A1.dot(A2)
r2 = A1.dot(x1)
r3 = A1.dot(x2)
print("r1 = A1.dot(A2) =\n", r1, "\nr2 = A1.dot(x1) =\n", r2, "\nr3 = A1.dot(x2) = ", r3)  # r1, r2, r3正确

# r4 = A1.dot(x3)
# r5 = x2.dot(A1)
# r4,r5错误,错误原因：
# r4:(2,3) (1,2)的矩阵无法相乘，只能是(1,2) (2,3)的两个矩阵相乘
# r4:shapes (2,3) and (2,) not aligned: 3 (dim 1) != 2 (dim 0)
# r5:(1,3) (2,3)的矩阵无法相乘
# r5:shapes (3,) and (2,3) not aligned: 3 (dim 0) != 2 (dim 0)

r6 = x3.dot(A1)
print("r6 = x3.dot(A1) = ", r6)  # r6正确
