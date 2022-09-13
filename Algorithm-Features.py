######################################################
# Deep Learning Regression with Python               #
# Algorithm Features                                 #
# (c) Diego Fernandez Garcia 2015-2018               #
# www.exfinsis.com                                   #
######################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import statsmodels.regression.linear_model as rg
import statsmodels.tools.tools as ct
import sklearn.decomposition as fe
import matplotlib.pyplot as plt

#########

# 2. Data Downloading or Reading

# Yahoo Finance
# data = web.DataReader('SPY', data_source='yahoo', start='2007-01-01', end='2016-01-01')
# spy = data['Adj Close']
# spy.columns = ['SPY.Adjusted']
# spy = pd.DataFrame(spy)

# Data Reading
spy = pd.read_csv('Data//Deep-Learning-Regression-Data.txt', index_col='Date', parse_dates=True)

#########

# 3. Feature Creation

# 3.1. Target Feature
rspy = spy/spy.shift(1)-1
rspy.columns = ['rspy']

# 3.2. Predictor Features
rspy1 = rspy.shift(1)
rspy1.columns = ['rspy1']
rspy2 = rspy.shift(2)
rspy2.columns = ['rspy2']
rspy3 = rspy.shift(3)
rspy3.columns = ['rspy3']
rspy4 = rspy.shift(4)
rspy4.columns = ['rspy4']
rspy5 = rspy.shift(5)
rspy5.columns = ['rspy5']

# 3.3. All Features
rspyall = rspy
rspyall = rspyall.join(rspy1)
rspyall = rspyall.join(rspy2)
rspyall = rspyall.join(rspy3)
rspyall = rspyall.join(rspy4)
rspyall = rspyall.join(rspy5)
rspyall = rspyall.dropna()
print(rspyall)

# 3.4. Range Delimiting

# 3.4.1. Training Range
rspyt = rspyall['2007-01-01':'2014-01-01']

# 3.4.2. Testing Range
rspyf = rspyall['2014-01-01':'2016-01-01']

#########

# 4. Algorithm Features

# 4.2. Predictor Features Selection

# 4.2.1. Linear Regression 1
rspyt.loc[:, 'const'] = ct.add_constant(rspyt)
pfst1 = ['const', 'rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5']
pfregt1 = rg.OLS(rspyt['rspy'], rspyt[pfst1], hasconst=bool).fit()
print("")
print("== Linear Regression 1 ==")
print("")
print(pfregt1.summary())
print("")

# 4.2.2. Linear Regression 2
pfst2 = ['const', 'rspy1', 'rspy2', 'rspy5']
pfregt2 = rg.OLS(rspyt['rspy'], rspyt[pfst2], hasconst=bool).fit()
print("")
print("== Linear Regression 2 ==")
print("")
print(pfregt2.summary())
print("")

# 4.3. Predictor Features Extraction

# 4.3.1. Principal Component Analysis
pfet = ['rspy1', 'rspy2', 'rspy3', 'rspy4', 'rspy5']
pcat = fe.PCA().fit(rspyt[pfet], rspyt['rspy'])
print("== Principal Component Analysis ==")
print("")
print("Principal Component Analysis Explained Variance:")
print("['pc1', 'pc2', 'pc3', 'pc4', 'pc5']")
np.set_printoptions(precision=4)
print(pcat.explained_variance_ratio_)
print("")

# 4.3.2. Principal Component Analysis Bar Chart
fig, ax = plt.subplots()
ax.bar(x=list(range(1, 6)), height=pcat.explained_variance_ratio_)
ax.set_title('Principal Component Analysis Explained Variance')
ax.set_ylabel('Explained Variance')
ax.set_xlabel('Principal Component')
plt.show()