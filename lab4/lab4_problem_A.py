# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: lab4_problem_A.py.py
@time: 2019/4/8
@desc: 现有鸢尾花数据集iris.data。Iris数据集是常用的分类实验数据集，由Fisher, 1936收集整理。Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。数据集包含150个数据集，分为3类，每类50个数据，每个数据包含4个属性。可通过花萼长度(sepal length)、花萼宽度(sepal width)、花瓣长度(petal length)、花瓣宽度(petal width),4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。
现需要进行如下实验：
1、使用pandas库读取数据集，得到相应矩阵，并进项相应的数据预处理：包括数据标准化与鸢尾花类别编码等。
2、采用决策树分类模型(DecisionTreeClassifier)训练鸢尾花数据集，测试集取30%，训练集取70%。
3、特征选择标准criterion请分别选择"gini"与“entropy”，在控制台分别打印出其测试集正确率。请问在iris.data数据集上，选择不同的特征选择标准，结果有无区别？
4、为了提升模型的泛化能力，请分别使用十折交叉验证，确定第三小问中两个决策树模型的参数max_depth（树的最大深度，该特征为最有效的预剪枝参数）与max_features（划分时考虑的最大特征数）的最优取值。max_depth取值范围为1-5，max_features的取值范围为1-4。请在控制台输出这两个参数的最优取值。
5、分别使用最优取值替换模型的参数设置。
6、为了更好的反应模型的预测能力，请在所有数据上使用sklearn的cross_val_score进行十折交叉验证，输出两个模型采用最优参数设置后的平均预测准确率，并在控制台输出。
根据你的输出，请回答在鸢尾花数据集分类时，特征选择标准是采用信息增益还是Gini系数更有效。

'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

if __name__ == '__main__':
    data = pd.read_csv('iris.data', header=None)

    # 用np.split按列（axis=1）进行分割
    # (4,):分割位置，前4列作为x的数据，第4列之后都是y的数据
    features, classes = np.split(data, (4,), axis=1)
    # 对标签进行编码
    le = LabelEncoder()
    le.fit(classes)
    classes = le.transform(classes).ravel()
    features_train, features_test, classes_train, classes_test = train_test_split(features, classes, test_size=0.3,
                                                                                  random_state=1)

    # 标准化特征值
    sc = StandardScaler()
    sc.fit(features_train)
    features_train_std = sc.transform(features_train)
    features_test_std = sc.transform(features_test)

    criterions = ['gini', 'entropy']

    print('--------------------测试集正确率--------------------')
    for cri in criterions:
        decision_tree_classifier = DecisionTreeClassifier(criterion=cri)
        decision_tree_classifier.fit(features_train_std, classes_train.ravel())
        score = decision_tree_classifier.score(features_test_std, classes_test.ravel())
        print("{}正确率:".format(cri), score)

    print('--------------------十折交叉验证准确率--------------------')
    for cri in criterions:
        # ShuffleSplit迭代器产生指定数量的独立的train/test数据集划分，首先对样本全体随机打乱，然后再划分出train/test对，可以使用随机数种子random_state来控制数字序列发生器使得讯算结果可重现
        cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=1)

        parameter_grid = {'max_depth': range(1, 6),
                          'max_features': range(1, 5)}
        decision_tree_classifier = DecisionTreeClassifier(criterion=cri)
        grid_search = GridSearchCV(decision_tree_classifier, param_grid=parameter_grid, cv=cv)
        # GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。
        grid_search.fit(features_train_std, classes_train.ravel())
        print('Best score of {0}: {1}'.format(cri, grid_search.best_score_))
        print('Best parameters of {0}: {1}'.format(cri, grid_search.best_params_))

    print('--------------------用最优值替代模型参数--------------------')
    # cross_val_score进行十折交叉验证
    decision_tree_classifier = DecisionTreeClassifier(criterion='gini', max_features=4, max_depth=3)
    decision_tree_classifier.fit(features_train_std, classes_train.ravel())
    score = cross_val_score(decision_tree_classifier, features_test_std, classes_test.ravel(), cv=cv).mean()
    print("gini正确率:", score)

    decision_tree_classifier = DecisionTreeClassifier(criterion='entropy', max_features=4, max_depth=3)
    decision_tree_classifier.fit(features_train_std, classes_train.ravel())
    score = cross_val_score(decision_tree_classifier, features_test_std, classes_test.ravel(), cv=cv).mean()
    print("entropy正确率:", score)
