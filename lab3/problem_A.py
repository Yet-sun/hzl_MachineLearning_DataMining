# encoding: utf-8

'''
@author: sunyue
@contact: sunyue@mail.ynu.edu.cn
@software: pycharm
@file: lab4_problem_A.py
@time: 2019/4/1
@desc:
1、使用pandas库读取数据集，得到相应矩阵。使用matplotlib库画出：TV、Radio、Newspaper与产品销售额的数据散点图。
具体要求：
	结果为一张图，TV, radio, 和 newspaper需要用不同形状的点表示。
	图的X轴为广告花费、Y轴为销售额的值。
	需要画出虚线形式的网格参考线。
2、 再次使用matplotlib库分别画出：TV与产品销售额、 Radio与产品销售额、Newspaper与产品销售额的数据散点图。
具体要求：
	结果为三张子图组成的一个大图，分为三行。从上到下的顺序依次为：TV与产品销售额、 Radio与产品销售额、Newspaper与产品销售额的数据散点图。
	图的X轴为广告花费、Y轴为销售额的值。
	需要画出虚线形式的网格参考线。
根据图示，请回答，哪一个广告媒介的投入与产品销售额最无关系？仅根据图的结果回答，请给出你的理由。
3、先对数据进行标准化后，建立线性回归中的多项式拟合模型，分别采用多项式的次数为1-9进行训练。最后根据预测结果与真实结果绘图。
具体要求：
	测试集取20%，训练集取80%。因为数据特征有三个（TV,Radio,NewsPaper），无法绘制特征与预测结果的二维图形。因此X轴换为测试样本下标，Y轴为产品销售额。
	分别画出9个图，在图中使用绿色线条代表模型针对测试集得出的预测销售额，使用红色线条代表测试集对应的实际产品销售额。图的标题表明线性模型多项式次数。
根据图示，请回答多项式次数为几时，预测效果最优，请给出你的理由。

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.preprocessing import PolynomialFeatures  # 导入多项式回归模型
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import sklearn.preprocessing

# 在代码中动态设置字体,否则无法显示中文
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

if __name__ == '__main__':
    path = './data/Advertising.csv'

    # 1.
    data = pd.read_csv(path)
    X = data[['TV', 'Radio', 'Newspaper']]
    Y = data['Sales']
    print(X.shape, Y.shape)

    plt.figure(facecolor='white')
    plt.plot(data['TV'], Y, 'ro', label='TV', mec='k')
    plt.plot(data['Radio'], Y, 'r+', label='Radio')
    plt.plot(data['Newspaper'], Y, 'b^', label='Newspaper', mec='k')
    plt.legend(loc='best')
    plt.xlabel('广告花费', fontsize=16, fontproperties=font)
    plt.ylabel('销售额', fontsize=16, fontproperties=font)
    plt.title('广告花费与销售额对比数据', fontsize=18, fontproperties=font)
    plt.grid(b=True, linestyle=':')  # 是否显示网格线,b设置为True。linestyle=':',设置为虚线
    plt.show()

    # 2.
    plt.figure(facecolor='w', figsize=(9, 10))
    plt.subplot(311)  # 311:即三行一列，占用第1行
    plt.plot(data['TV'], Y, 'b<', mec='k')
    plt.xlabel('广告花费', fontsize=16, fontproperties=font)
    plt.ylabel('销售额', fontsize=16, fontproperties=font)
    plt.title('TV与产品销售额', fontproperties=font)
    plt.grid(b=True, ls=':')

    plt.subplot(312)  # 312:即三行一列，占用第2行
    plt.plot(data['Radio'], Y, 'ro', mec='k')
    plt.xlabel('广告花费', fontsize=16, fontproperties=font)
    plt.ylabel('销售额', fontsize=16, fontproperties=font)
    plt.title('Radio与产品销售额', fontproperties=font)
    plt.grid(b=True, ls=':')

    plt.subplot(313)  # 313:即三行一列，占用第3行
    plt.plot(data['Newspaper'], Y, 'g^', mec='k')
    plt.xlabel('广告花费', fontsize=16, fontproperties=font)
    plt.ylabel('销售额', fontsize=16, fontproperties=font)
    plt.title('Newspaper与产品销售额', fontproperties=font)
    plt.grid(b=True, ls=':')
    plt.tight_layout(pad=2)
    plt.show()

    # 3.
    for degree in range(1, 10):
        # 基于不同的次数生成多项式模型
        model = make_pipeline(PolynomialFeatures(degree),
                              LinearRegression(normalize=False))  # normalize：默认为False，是否对数据归一化。
        # 对数据进行标准化
        X = preprocessing.scale(X)
        Y = preprocessing.scale(Y).reshape(-1, 1)

        # 划分训练集和测试集
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                            random_state=1)  # 切分数据集,测试集取20%,训练集取80%,random_state：随机种子

        model.fit(X_train, Y_train)  # 训练

        y_pred = model.predict(X_test)  # 预测
        plt.subplot('33' + str(degree))
        plt.plot(y_pred, 'g-', label='predict sales')
        plt.plot(Y_test, 'r-', label='real sales')
        plt.title('degree {}'.format(degree))
        plt.xlabel('广告花费', fontsize=16, fontproperties=font)
        plt.ylabel('产品销售额', fontsize=16, fontproperties=font)
        plt.grid(b=True, ls=':')
        plt.legend(loc='best',fontsize=5)
        plt.tight_layout(pad=2)

    plt.show()
