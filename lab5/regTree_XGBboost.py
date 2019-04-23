# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: regTree_XGBboost.py
@time: 2019/4/21
@desc:
分别使用回归树与XGBoost回归，预测给出的Advertising.csv数据集，并与传统线性回归预测方法进行比较。
具体要求：
1、	首先进行数据标准化。
2、	测试集和训练集比例分别为30%和70%。
3、	使用均方误差来评价预测的好坏程度。
4、	对于XGBoost请尝试使用交叉验证找到n_estimators的最优参数值。n_estimators的取值范围为[100-1000]。
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit


def data(path):
    '''
    读取数据并标准化，切分数据集
    :param path:
    :return:
    '''
    data = pd.read_csv(path)

    X = data[['TV', 'Radio', 'Newspaper']]
    Y = data[['Sales']]
    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    return X_train, X_test, Y_train, Y_test


def xgboost(X_train, X_test, Y_train, Y_test):
    '''
    xgboost模型训练
    :param X_train:
    :param X_test:
    :param Y_train:
    :param Y_test:
    :return:
    '''
    param = {
        'max-depth': 5,
        'min_child_weight': 0,
        'eta': 1,
        'verbosity': 1,
        'objective': 'reg:gamma',
        'nthread': 4,
        'gamma': 0,
        'n_estimators': 160,
        'learning_rate': 0.1,
    }

    bst = xgb.XGBRegressor(**param)
    bst.fit(X_train, Y_train)
    preds = bst.predict(X_test)

    mse = mean_squared_error(preds, Y_test)  # 均分误差
    print("xgboost mse:", mse)

    '''交叉验证找到n_estimators的最优参数值'''
    # ShuffleSplit迭代器产生指定数量的独立的train/test数据集划分，首先对样本全体随机打乱，然后再划分出train/test对，可以使用随机数种子random_state来控制数字序列发生器使得讯算结果可重现
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=1)

    parameter_grid = {'n_estimators': range(100, 1000 + 1)}
    grid_search = GridSearchCV(bst, param_grid=parameter_grid, cv=cv)
    # GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。
    grid_search.fit(X_train, Y_train)
    print('Best score of xgboost: {}'.format(grid_search.best_score_))
    print('Best parameters of n_estimators: {}'.format(grid_search.best_params_))

    # 绘图
    plt.figure()
    num = 30
    x = np.arange(1, num + 1)
    plt.plot(x, Y_test[:num], '-', label='real')
    plt.plot(x, preds[:num], '--', label='preds')
    plt.title("xgboost")
    plt.legend(loc='best')
    plt.show()


def reg_tree(X_train, X_test, Y_train, Y_test):
    '''
    回归树模型
    :param X_train:
    :param X_test:
    :param Y_train:
    :param Y_test:
    :return:
    '''
    reg = tree.DecisionTreeRegressor()
    reg.fit(X_train, Y_train)
    preds = reg.predict(X_test)
    mse = mean_squared_error(preds, Y_test)
    print("regression tree mse:", mse)

    plt.figure()
    num = 30
    x = np.arange(1, num + 1)
    plt.plot(x, Y_test[:num], '-', label='real')
    plt.plot(x, preds[:num], '--', label='preds')
    plt.title("regression tree")
    plt.legend(loc='best')
    plt.show()


def line_reg(X_train, X_test, Y_train, Y_test):
    '''
    线性回归模型
    :param X_train:
    :param X_test:
    :param Y_train:
    :param Y_test:
    :return:
    '''
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(preds, Y_test)
    print("linear regression mse:", mse)

    plt.figure()
    num = 30
    x = np.arange(1, num + 1)
    plt.plot(x, Y_test[:num], '-', label='real')
    plt.plot(x, preds[:num], '--', label='preds')
    plt.title("linear regression")
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    path = './Advertising.csv'
    X_train, X_test, Y_train, Y_test = data(path)

    reg_tree(X_train, X_test, Y_train, Y_test)
    line_reg(X_train, X_test, Y_train, Y_test)
    xgboost(X_train, X_test, Y_train, Y_test)
