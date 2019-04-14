# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: problem_B.py
@time: 2019/4/1
@desc:
现有鸢尾花数据集iris.csv。Iris数据集是常用的分类实验数据集，由Fisher, 1936收集整理。
Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性。
可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。
具体要求：
	使用逻辑回归模型训练鸢尾花数据集，测试集取20%，训练集取80%。
	先对数据进行标准化后，分别采用多项式的次数为1-9进行训练，solver和multi_class请自行选择。
	分别在控制台打印出多项式次数为1-9时，该模型在测试集上预测出准确分类的正确率。
根据你的输出，请回答多项式次数为几时，预测效果最优。
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    path = './data/iris.csv'

    data = pd.read_csv(path, header=None)

    # 用np.split按列（axis=1）进行分割
    # (4,):分割位置，前4列作为x的数据，第4列之后都是y的数据
    X, Y = np.split(data, (4,), axis=1)
    # 对标签进行编码
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # 标准化特征值
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    for degree in range(1, 10):
        # 训练逻辑回归模型
        # solver: 对于多分类任务， 使用‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ 来解决多项式loss
        # multi_class: 默认值‘ovr’适用于二分类问题，对于多分类问题，用‘multinomial’在全局的概率分布上最小化损失
        logreg = make_pipeline(PolynomialFeatures(degree),
                               LogisticRegression(solver='sag', multi_class='multinomial', max_iter=10000))
        logreg.fit(X_train, Y_train)

        # 预测
        # prepro = logreg.predict_proba(X_test_std)
        acc = logreg.score(X_test_std, Y_test)
        print("accuracy of degree{}:".format(degree), format(acc * 100, '.2f'), "%")
