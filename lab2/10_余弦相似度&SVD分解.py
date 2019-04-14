# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 10_余弦相似度&SVD分解.py
@time: 2019/3/23
@desc: 计算各（d2,d3）文档的相似度，试用余弦相似度和SVD分解分别计算，并比较两种方法得出结果的合理性。
'''

import numpy as np

d1 = np.array([[1, 0, 1, 1, 0]])
d2 = np.array([[0, 1, 1, 0, 0]])
d3 = np.array([[1, 0, 0, 0, 0]])
d4 = np.array([[0, 0, 0, 1, 1]])
d5 = np.array([[0, 0, 0, 1, 0]])
num = float(d2.dot(d3.T))
denom = np.linalg.norm(d2) * np.linalg.norm(d3)
cos = num / denom  # 余弦值
sim = 0.5 + 0.5 * cos  # 根据皮尔逊相关系数归一化
print("余弦相似度为：", sim)

C = np.vstack((d1, d2, d3, d4, d5)).T
U, sigma, VT = np.linalg.svd(C)
# 按照前k个奇异值的平方和占总奇异值的平方和的百分比来确定k的值,后续计算SVD时需要将原始矩阵转换到k维空间
percentage = 0.9
sigma2 = sigma ** 2  # 对sigma求平方
sumsgm2 = sum(sigma2)  # 求所有奇异值sigma的平方和
sumsgm3 = 0  # sumsgm3是前k个奇异值的平方和
k = 0
for i in sigma:
    sumsgm3 += i ** 2
    k += 1
    if sumsgm3 >= sumsgm2 * percentage:
        break
d = np.diag(sigma[:k]).dot(VT.T[:k])  # 文档坐标

d2 = d[1]
d3 = d[2]
# 用新的文档坐标计算余弦相似度
num = float(d2.dot(d3.T))
denom = np.linalg.norm(d2) * np.linalg.norm(d3)
cos = num / denom  # 余弦值
sim = 0.5 + 0.5 * cos  # 根据皮尔逊相关系数归一化
print("应用SVD后的相似度：", sim)

d = np.diag(sigma[:k]).dot(VT.T[:k])  # 得到文档的坐标

U, Sigma, VT = np.linalg.svd([d2, d3])
V1 = np.eye(2) * Sigma
new = np.matmul(U, np.matmul(V1, VT[:2, :]))
result = 1.0 / (1.0 + np.linalg.norm(new[0] - new[1]))  # 欧式距离求相似度
print("SVD欧式距离求相似度:", result)
