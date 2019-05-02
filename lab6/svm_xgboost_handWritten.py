# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: svm_xgboost_handWritten.py
@time: 2019/4/29
@desc:
分别使用支持向量机与一个你喜欢的分类模型（任选）对手写体数字光学识别数据集(UCI)进行训练，并对比其正确率与时间效率。
具体要求：
1、	训练集采用optdigits.tra。测试集采用optdigits.tes。
2、	使用分类准确率评价分类器的好坏程度。
3、	需要记录训练所花费时间，以及在训练集上预测分类的所花费时间。
4、	（选做，有加分）各模型的超参数选择交叉验证得出。
5、	（选做，有加分）分析你的结果，并简单探讨两个分类器好坏程度、效率两个方面的原因。
'''

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
import xgboost as xgb
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings

path = './data/'
# 训练过程中后台一直报一个警告，经查阅后知道是一个numpy的问题，已经修复，但未在最新版本中发布，故这里选择直接忽略所有的警告
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


def loadData():
    print('training data is loading...')
    data = pd.read_csv(path + '/optdigits.tra', header=None)
    x_train, y_train = data[list(range(64))], data[64]
    x_train, y_train = x_train.values, y_train.values  # 转换为numpy形式
    images_train = x_train.reshape(-1, 8, 8)  # 每一行为一个8*8的矩阵,对应图片
    print('images.shape = ', images_train.shape)
    y_train = y_train.ravel().astype(np.int)  # 将数组降维一维后数据格式转化为np.int类型

    print('test data is loading...')
    data = pd.read_csv(path + 'optdigits.tes', header=None)
    x_test, y_test = data[list(range(64))], data[64]
    x_test, y_test = x_test.values, y_test.values
    images_test = x_test.reshape(-1, 8, 8)
    y_test = y_test.ravel().astype(np.int)

    print('data is ready...')
    return images_train, images_test, x_train, y_train, x_test, y_test


def svm_train(x_train, y_train, x_test, y_test):
    params = {'C': np.logspace(0, 3, 7),
              'gamma': np.logspace(-5, 0, 11)}
    model = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=params, cv=3)
    # model = svm.SVC(C=1, kernel='rbf', gamma=0.001)
    print('[svm]training start...')
    t_start = time()
    model.fit(x_train, y_train)
    t_end = time()
    t = t_end - t_start
    print('[svm]训练+CV耗时：%d分钟%.3f秒' % (int(t / 60), t - 60 * int(t / 60)))
    print('[svm]best parameters of C:', model.best_params_)
    print('[svm]training completed...')
    print('[svm]训练集准确率：', accuracy_score(y_train, model.predict(x_train)))
    print('[svm]测试集准确率：', accuracy_score(y_test, model.predict(x_test)))
    y_hat = model.predict(x_test)
    mse = mean_squared_error(y_hat, y_test)  # 均分误差
    print("[svm]mse:", mse)
    show_err(y_hat, 'svm')


def xgboost_train(x_train, y_train, x_test, y_test):
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

    print('[xgboost]training start...')
    t_start = time()
    bst = xgb.XGBClassifier(**param)
    bst.fit(x_train, y_train)
    '''交叉验证找到n_estimators的最优参数值'''
    # ShuffleSplit迭代器产生指定数量的独立的train/test数据集划分，首先对样本全体随机打乱，然后再划分出train/test对，可以使用随机数种子random_state来控制数字序列发生器使得讯算结果可重现
    cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=1)
    parameter_grid = {'n_estimators': range(100, 300)}
    grid_search = GridSearchCV(bst, param_grid=parameter_grid, cv=cv)
    # GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。
    grid_search.fit(x_train, y_train)
    t_end = time()
    t = t_end - t_start
    print('[xgboost]训练+CV耗时：%d分钟%.3f秒' % (int(t / 60), t - 60 * int(t / 60)))
    print('[xgboost]training completed...')
    print('[xgboost]Best score of xgboost: {}'.format(grid_search.best_score_))
    print('[xgboost]Best parameters of n_estimators: {}'.format(grid_search.best_params_))

    y_hat = bst.predict(x_test)
    mse = mean_squared_error(y_hat, y_test)  # 均分误差
    print("[xgboost]mse:", mse)
    print('[xgboost]训练集准确率:', accuracy_score(y_train, bst.predict(x_train)))
    print('[xgboost]测试集准确率:', accuracy_score(y_test, bst.predict(x_test)))

    show_err(y_hat, 'xgboost')


def show_err(y_hat, modelName):
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    err_images = images_test[y_test != y_hat]
    err_y_hat = y_hat[y_test != y_hat]
    err_y = y_test[y_test != y_hat]
    # print(err_y_hat)
    # print(err_y)
    plt.figure(figsize=(10, 8), facecolor='w')
    for index, image in enumerate(err_images):
        if index >= 12:
            break
        plt.subplot(3, 4, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('%s 错分为：%i，真实值：%i' % (modelName, err_y_hat[index], err_y[index]))
    plt.tight_layout()
    plt.show()


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(tip + '分类正确率：%.2f%%' % (100 * np.mean(acc)))


if __name__ == '__main__':
    images_train, images_test, x_train, y_train, x_test, y_test = loadData()
    svm_train(x_train, y_train, x_test, y_test)
    xgboost_train(x_train, y_train, x_test, y_test)
