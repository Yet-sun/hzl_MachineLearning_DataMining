# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 04_数组的修改操作.py
@time: 2019/3/18
@desc: 数组的修改操作
'''

import numpy as np

# （1）使用单一值创建数组
# 查看数组的维度尺寸
arr1 = np.zeros((3, 4))
print("arr1:\n", arr1.shape)

# 一维数组变形为mxn二维数组
arr2 = np.arange(9)
print("arr2:\n", arr2)
arr3 = np.array(arr2).reshape(3, 3)
print("arr3:\n", arr3)

# 将二维数组调整为一行或一列
arr4 = np.array(arr3).reshape(1, -1)  # 一行
print("arr4:\n", arr4)
arr5 = np.array(arr3).reshape(-1, 1)  # 一列
print("arr5:\n", arr5)

# 行数组转成列数组
arr6 = np.array(arr4).reshape(-1, 1)
print("arr6:\n", arr6)

# 二维数组展成连续的一维数组
arr7 = np.array(arr3).flatten()
print("arr7:\n", arr7)

# 二维数组展成连续的一维数组(拷贝)
arr8 = arr3.flatten()
print("arr8:\n", arr8)

# 将原有数组调整为新指定尺寸的数组(拷贝)
arr9 = arr3.reshape(1, -1)
print("arr9:\n", arr9)

# 生成转置数组(矩阵)
arr10 = arr3.transpose()
print("arr10:\n", arr10)

# （2）数组的组合、拼接及拆分
# 以竖直方向叠加两个数组
a = np.arange(1, 7).reshape(2, 3)
b = np.arange(7, 13).reshape(2, 3)
arr11 = np.vstack((a, b))  # 竖直方向叠加
print("arr11:\n", arr11)

# 以水平方向叠加两个数组
arr12 = np.hstack((a, b))  # 水平方向叠加
print("arr12:\n", arr12)

# 竖直方向将二维数组拆分成若干个数组
# 根据axis所指定的轴向（0,1）进行多维数组的组合
# 如果待组合的两个数组都是二维数组
# axis=0:垂直方向
# axis=1:水平方向
arr13 = np.split(arr11, 4, axis=0)
print("arr13:\n", arr13)

# 水平方向将二维数组拆分成若干个数组
arr14 = np.split(arr11, 3, axis=1)
print("arr14:\n", arr14)

# （3）访问及修改元素
# 访问二维数组
arr15 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=int)
print("arr15:\n", arr15)

# 访问一维数组的部分元素
arr16 = np.array([1, 1, 2, 2, 3, 3, 4, 4])
e = arr16[3]  # 第4个单元元素
print("element:", e)

# 访问二维数组的部分元素
e = arr15[0, 1]  # 1行 第2个单元元素
print("element:", e)

# 删除元素
arr17 = np.delete(arr16, 1, axis=0)
print("arr17:\n", arr17)

# 删除行或列
arr18 = np.delete(arr15, 1, axis=0)  # 删除第二行（列不变）
print("arr18:\n", arr18)
arr19 = np.delete(arr15, 1, 1)  # 删除第二列（行不变）
print("arr19:\n", arr19)
# 第三参数axis为1删除的是列，为0删除的是行,可以只写数字不写axis，第二个参数决定删除的行数或列数

# 插入元素、行或列
b = np.array([[0, 0, 0]])
arr20 = np.insert(arr15, 1, values=b, axis=0)  # 在第一行后面插入一行
print("arr20:\n", arr20)
c = np.array([[0, 0, 0, 0]])
arr21 = np.insert(arr15, 0, values=c, axis=1)  # 在第一列前面插入一列
print("arr21:\n", arr21)

# 追加元素、行或列
d = np.array([0, 1, 2, 3])
arr22 = np.append(d, values=4)  # 追加元素
print("arr22:\n", arr22)
arr23 = np.append(d, values=b)  # 追加一行
print("arr23:\n", arr23)

# 在一个二维数组后添加一列
c = np.array([[0, 0, 0, 0]])
arr24 = np.insert(arr15, 3, values=c, axis=1)
print("arr24:\n", arr24)
