# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: 08_矩阵运算.py
@time: 2019/3/23
@desc: 矩阵运算
'''

import numpy as np

# numpy.linalg模块包含线性代数的函数。使用这个模块，可以计算逆矩阵、求特征值、解线性方程组以及求解行列式等。
# （1）矩阵的逆
# 用numpy包中的linalg.inv方法计算矩阵的逆。
A1 = np.array([[2, 2, 3], [1, -1, 0], [-1, 2, 1]])
A1_inv = np.linalg.inv(A1)
print("A1:\n", A1, "\nA1_inv:\n", A1_inv)

# （2）转置矩阵
A1 = np.array([[2, 2, 3], [1, -1, 0], [-1, 2, 1]])
A1_T = A1.T
print("A1:\n", A1, "\nA1_T:\n", A1_T)

# （3）特征分解
# 用numpy包中的linalg.eig函数对矩阵进行特征分解。
A = np.array([[2, 2, 3], [1, -1, 0], [-1, 2, 1]])
print("A:\n", A, "\n特征分解:")

vals, vecs = np.linalg.eig(A)
print("vals：", vals, "\nvecs:\n", vecs)

# （4）SVD分解
# numpy.linalg模块中的svd函数可以对矩阵进行奇异值分解。
# 该函数返回3个矩阵——U、Sigma和V，其中U和V是正交矩阵，Sigma包含输入矩阵的奇异值。
A1 = np.array([[2, 2, 3], [1, -1, 0], [-1, 2, 1]])
A2 = np.array([[2, 2, 3], [4, 4, 6], [-1, 2, 1]])
A3 = np.array([[2, 2, 3], [1, -1, 0], [-1, 2, 1], [-3, 1, 2]])
print("A1:\n", A1, "\nA2:\n", A2, "\nA3:\n", A3, "\nSVD分解:")

U1, Sigma1, V1 = np.linalg.svd(A1, full_matrices=False)
print("U1:\n", U1)
print("Sigma1:", Sigma1)
print("V1:\n", V1)
U2, Sigma2, V2 = np.linalg.svd(A2, full_matrices=False)
print("U2:\n", U2)
print("Sigma2:", Sigma2)
print("V2:\n", V2)
U3, Sigma3, V3 = np.linalg.svd(A3, full_matrices=False)
print("U3:\n", U3)
print("Sigma3:", Sigma3)
print("V3:\n", V3)
# 结果包含等式中左右两端的两个正交矩阵U和V，以及中间的奇异值矩阵Sigma

# （5）矩阵的秩
# 用numpy包中的linalg.matrix_rank方法计算矩阵的秩。
A1 = np.array([[2, 1, 3], [6, 6, 10], [2, 7, 6]])
A2 = np.array([[2, 1, 3], [4, 2, 6], [2, 7, 6]])
A3 = np.array([[2, 1, 3], [6, 6, 10], [2, 7, 6], [1, 3, 5]])
A1_matrix_rank = np.linalg.matrix_rank(A1)
A2_matrix_rank = np.linalg.matrix_rank(A2)
A3_matrix_rank = np.linalg.matrix_rank(A3)
print("A1:\n", A1, "\nA2:\n", A2, "\nA3:\n", A3, "\n矩阵的秩:")
print("A1_matrix_rank:", A1_matrix_rank, "\nA2_matrix_rank:", A2_matrix_rank, "\nA3_matrix_rank:", A3_matrix_rank)
