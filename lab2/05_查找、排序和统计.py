# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 05_查找、排序和统计.py
@time: 2019/3/19
@desc: 查找、排序和统计
'''

import numpy as np

# （1）检索符合条件的元素（请注意，所有返回结果与原数组都是独立的数据空间）
# 一维数组中，查找不为0的元素
arr1 = np.arange(10)
print("arr1:", arr1)
print("result:")
for i in arr1:
    if arr1[i] != 0:
        print(arr1[i], end=" ")

# 二维数组中，查找不为0的元素
arr2 = arr1.reshape(2, 5)
print("\narr2:\n", arr2)
print("result:", )
for element in arr2.flat:  # 对数组中每个元素都进行处理，可以使用flat属性，该属性是一个数组元素迭代器：
    if (element != 0):
        print(element, end=" ")

# 查找指定条件的元素
x = arr2
x = arr2[x > 5]  # 直接筛选出大于5的元素
print("\nresult:", x)

# 返回指定索引的若干个元素
arr3 = arr2[:, 2:]  # 返回每行第3到最后一列的元素
print("arr3:\n", arr3)

# （2）数组排序
# 将数组倒序
arr4 = np.array([1, 2, 6, 8, 6, 4, 3, 1, 9])
print("arr4:", arr4)
print("arr4_reverse", arr4[::-1])

# 一维数组排序
print("arr4_sort:", np.sort(arr4))

# 二维数组排序
arr5 = np.array([[3, 7, 6], [9, 1, 4]])
print("arr5:\n", arr5)
print("arr5_sort:\n", np.sort(arr5))

# 以指定索引位置作为分界线，左边元素都小于分界元素，右边元素都大于分界元素
print("arr6:", np.partition(arr4, arr4[4]))  # arr4[4]=6,小于6的排在前面，大于6的放在后面

# （3）数组统计
# 查找一维数组中的最大、最小值
arr7 = np.arange(3, 13)
print("arr7:", arr7)
print("arr7_min:", np.amin(arr7))
print("arr7_max", np.amax(arr7))

# 查找二维数组总的最大、最小值
arr8 = arr7.reshape(2, 5)
print("arr8:\n", arr8)
minindex = np.argmin(arr8)
maxindex = np.argmax((arr8))
print("arr8_min", arr8.flatten()[minindex])
print("arr8_max:", arr8.flatten()[maxindex])

# 查找极值元素的索引
print("arr8_min_index", minindex)
print("arr8_max_index:", maxindex)

# 统计数组中非零元素个数
print("非零个数：", np.sum(arr8 != 0))

# 计算数组算数平均值
print("arr7_means:", np.mean(arr7))

# 计算数组的加权平均值
print("arr7_average:", np.average(arr7, weights=np.arange(12, 2, -1)))
