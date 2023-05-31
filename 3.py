import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
data = pd.read_excel('附件 2.xlsx',index_col='NO')
X = data['Time']
Y1 = data['1# Temperature']
Y2 = data['2# Temperature']

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn

matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
data = pd.read_csv('feture.csv',index_col='Unnamed: 0')
wendu = pd.read_excel('附件 2.xlsx',index_col='NO')
X = wendu['Time']
Y1 = wendu['1# Temperature']
Y2 = wendu['2# Temperature']
plt.figure(figsize=(10, 4))
plt.plot(data.iloc[40:-1, 28])
plt.show()


#python 的多项式拟合
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

poly_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=3)),
    ("std_sclar", StandardScaler()),
    ("lin_reg", LinearRegression())
])
x = np.array([i for i in range(150, 671, 1)])
X = x.reshape(-1, 1)
y = np.array(data.iloc[40:-1, 28])
poly_reg.fit(X, y)
y_predict = poly_reg.predict(X)
plt.figure(figsize=(10, 4))
plt.plot(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)], color='r')
plt.figure(figsize=(10, 4))
plt.plot(np.sort(x),1-(y_predict[np.argsort(x)]-np.min(y_predict))/(np.max(y_predict)-np.min(y_predict)), color='r')
plt.xlabel('时 间', fontsize=15)
plt.ylabel('进 度', fontsize=15)
plt.show()
y_pre=1-(y_predict[np.argsort(x)]-np.min(y_predict))/(np.max(y_predict)-np.min(y_predict))
pd.DataFrame(y_pre).to_csv('进度曲线数据.csv')
##ACF 和 PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def ACF_and_PCAF(data):
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title('Ice Cream Sales', fontsize=18)
    plt.ylabel('Sales')
    acf_plot = plot_acf(data, lags=100)
    plt.xlabel('时 间', fontsize=15)
    plt.ylabel('进 度', fontsize=15)
    pacf_plot = plot_pacf(data)
    plt.show(block=True)
    return 0

ACF_and_PCAF(data.iloc[:, 28])

for i in range(47):
    x = data.iloc[:, i]
    ACF_and_PCAF(x)
    ACF_and_PCAF(data.iloc[:, 28])

from statsmodels.tsa.stattools import acf,pacf

plt.figure(figsize=(10, 4))
lag_acf = acf(data.iloc[:, 28], nlags=100)

plt.plot(-np.log(lag_acf)/np.max(-np.log(lag_acf))*100)
plt.plot(X, Y1)
plt.show()

#白噪声检验结果
from statsmodels.stats.diagnostic import acorr_ljungbox

print(u'白噪声检验结果：',acorr_ljungbox(data.iloc[:, 28], lags=2))#返回统计量和 p 值 lags 为检验的延迟数
PCC = []
p_value = []
for i in range(len(data)-1):
    x = data.iloc[i, :]
    y = data.iloc[i+1, :]
PCC.append(pearsonr(x, y)[0])
p_value.append(pearsonr(x, y)[1])
#
# ACF_and_PCAF(PCC)
# 绘制特征值图形
for i in range(47):
    x = data.iloc[:, i]
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(x))
    plt.show()
# #绘制特征值图形
PCC = []
p_value = []
for i in range(len(data)-1):
    x = data.iloc[i, :]
    y = data.iloc[i+1, :]
    PCC.append(pearsonr(x, y)[0])
    p_value.append(pearsonr(x, y)[1])
#
#
PCC_1 = []
p_value_1 = []
#
temp = [0, 30, 38, 39, len(data)-2]
#
for i in temp:
    x = data.iloc[i, :]
    y = data.iloc[i+1, :]
    PCC_1.append(pearsonr(x, y)[0])
    p_value_1.append(pearsonr(x, y)[1])
#
#
plt.figure(figsize=(8, 6))
plt.plot(PCC, zorder=1)
plt.scatter(temp, PCC_1, c='red', s=40, zorder=2)
plt.xlabel('时 间 序 列', fontsize=14)
plt.ylabel('PCC', fontsize=14)
#
plt.figure(figsize=(8, 6))
plt.plot(p_value, zorder=1)
plt.scatter(temp, p_value_1, c='red', s=40, zorder=2)
plt.xlabel('时 间 序 列', fontsize=14)
plt.ylabel('P-value', fontsize=14)
plt.show()
temp = [0, 30, 38, 39, -2]
#
for i in temp:
    x = data.iloc[i, :]
    y = data.iloc[i+1, :]
    PCC.append(pearsonr(x, y)[0])
    p_value.append(pearsonr(x, y)[1])
    plt.figure(figsize=(8,6))
    plt.plot(PCC)
#
plt.figure(figsize=(8,6))
plt.plot(p_value)
plt.show()