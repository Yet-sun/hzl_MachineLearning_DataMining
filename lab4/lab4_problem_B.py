# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: lab4_problem_B.py
@time: 2019/4/13
@desc: 
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    data = pd.read_csv('iris.data', header=None)

    # 用np.split按列（axis=1）进行分割
    # (4,):分割位置，前4列作为x的数据，第4列之后都是y的数据
    features, classes = np.split(data, (4,), axis=1)
    # 对标签进行编码
    le = LabelEncoder()
    le.fit(classes)
    classes = le.transform(classes)
    features_train, features_test, classes_train, classes_test = train_test_split(features, classes, test_size=0.3,
                                                                                  random_state=1)

    # 标准化特征值
    sc = StandardScaler()
    sc.fit(features_train)
    features_train_std = sc.transform(features_train)
    features_test_std = sc.transform(features_test)

    clf = RandomForestClassifier(n_estimators=10, criterion='entropy')  # n_estimators:决策树的个数，一定条件下越多越好，但是性能就会越差
    rf_clf = clf.fit(features_train_std, classes_train.ravel())
    score = rf_clf.score(features_test_std, classes_test.ravel())
    print("随机森林准确率:", score)

    print('--------十折交叉验证准确率--------')
    # ShuffleSplit迭代器产生指定数量的独立的train/test数据集划分，首先对样本全体随机打乱，然后再划分出train/test对，可以使用随机数种子random_state来控制数字序列发生器使得讯算结果可重现
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=1)
    parameter_grid = {'max_depth': range(1, 6),
                      'n_estimators': range(1, 21)}
    clf = RandomForestClassifier(criterion='entropy')
    grid_search = GridSearchCV(clf, param_grid=parameter_grid, cv=cv)
    # GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。
    grid_search.fit(features_train_std, classes_train.ravel())
    print('Best score {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
