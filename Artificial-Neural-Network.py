######################################################
# Deep Learning Regression with Python               #
# Artificial Neural Network                          #
# (c) Diego Fernandez Garcia 2015-2018               #
# www.exfinsis.com                                   #
######################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import sklearn.preprocessing as pr
import sklearn.decomposition as fe
import sklearn.model_selection as cv
import sklearn.metrics as fa
import matplotlib.pyplot as plt
import keras.models as nn
import keras.layers as ml
from tensorflow import optimizers as opt
import keras.wrappers.scikit_learn as ks
import time

#########

# 2. Data Downloading or Reading

# Yahoo Finance
# data = web.DataReader('SPY', data_source='yahoo', start='2007-01-01', end='2016-01-01')
# spy = data['Adj Close']
# spy.columns = ['SPY.Adjusted']

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

# 3.4. Range Delimiting

# 3.4.1. Training Range
rspyt = rspyall['2007-01-01':'2014-01-01']

# 3.4.2. Testing Range
rspyf = rspyall['2014-01-01':'2016-01-01']

#########

# 4. Artificial Neural Network

# 4.1. ANN Regression Features

# 4.1.1. Features Pre-processing
# re-scaling range [0,1]
# prspyt = (rspyt-min(rspyall))/(max(rspyall)-min(rspyall))
scalet = pr.MinMaxScaler(feature_range=(0, 1)).fit(rspyall)
prspyt = scalet.transform(rspyt)
prspyf = scalet.transform(rspyf)

# 4.1.2. Target Feature
yt = prspyt[:, 0]
yf = prspyf[:, 0]

# 4.1.3. Predictor Features Selection
xta = prspyt[:, [1, 2, 5]]
xfa = prspyf[:, [1, 2, 5]]

# 4.1.4. Predictor Features Extraction
# Principal Components Analysis
xtb = fe.PCA().fit_transform(prspyt[:, 1:6])
xfb = fe.PCA().fit_transform(prspyf[:, 1:6])

# 4.2. ANN Regression Model
def annt_model(wdecay, inshape):
    # 4.2.1. Model Type
    annt = nn.Sequential()
    # 4.2.2. Model Layers
    # Number of Hidden Layers
    # Dense: units: number of output features, input_shape: 2D data dimension (samples and features)
    # Activation: activation function options: 'linear' activation function
    ###
    # Hidden Layer 1
    annt.add(ml.Dense(units=1, input_shape=inshape))
    annt.add(ml.Activation(activation='linear'))
    annt.add(ml.Dropout(rate=0.0))
    annt.add(ml.ActivityRegularization(l1=0.0, l2=wdecay))
    # 4.2.3. Model Optimization Algorithm
    # Back-propagation optimization options: 'SGD' stochastic gradient descent
    # Optimization regularization: learning rate 'lr', 'momentum' and learning rate 'decay'
    annoptt = opt.SGD(learning_rate=0.01, momentum=0, decay=0, nesterov=False)
    # 4.2.4. Model Compilation
    # Loss function options: 'mean_squared_error'
    annt.compile(loss='mean_squared_error', optimizer=annoptt, metrics=['accuracy'])
    return annt

# 4.3. ANN Regression Training Optimal Parameter Selection

# 4.3.1. Time Series Cross-Validation
# Exhaustive Grid Search Time Series Cross-Validation with Parameter Array Specification
# TimeSeriesSplit = anchored time series cross-validation with
# initial training subset = validating subset ~ number of observations per sample / (number of splits + 1) in size
anntsa = time.time()
cvannta = cv.GridSearchCV(ks.KerasRegressor(annt_model), cv=cv.TimeSeriesSplit(n_splits=5),
                          param_grid={"wdecay": [0.00, 0.10], "inshape": [(3,)]}).fit(xta, yt)
anntea = time.time()
anntsb = time.time()
cvanntb = cv.GridSearchCV(ks.KerasRegressor(annt_model), cv=cv.TimeSeriesSplit(n_splits=5),
                          param_grid={"wdecay": [0.00, 0.10], "inshape": [(5,)]}).fit(xtb, yt)
annteb = time.time()

# 4.3.2. Time Series Cross-Validation Optimal Parameter Selection
cvannpara = cvannta.best_params_
cvannparb = cvanntb.best_params_
print("")
print("== ANN Regression Training Optimal Parameter Selection ==")
print("")
print("ANN Regression A Optimal Weight Decay L2 Regularization: ", cvannpara)
print("ANN Regression B Optimal Weight Decay L2 Regularization: ", cvannparb)
print("")
print("ANN Regression A Training Time: ", (anntea-anntsa)/60, " minutes")
print("ANN Regression B Training Time: ", (annteb-anntsb)/60, " minutes")
print("")

# 4.4. ANN Algorithm Training
annta = cvannta.best_estimator_.model
anntb = cvanntb.best_estimator_.model

# 4.5. ANN Regression Testing
# re-scaling back to original scale, values.reshape(-1,1) for single feature, values.reshape(1,-1) for single sample
annfa = annta.predict(x=xfa)
annfb = anntb.predict(x=xfb)
scalef = pr.MinMaxScaler(feature_range=(0, 1)).fit(rspyall['rspy'].values.reshape(-1, 1))
annfa = scalef.inverse_transform(annfa)
annfb = scalef.inverse_transform(annfb)

# 4.5.1. Testing Charts

# ANN Regression Testing Chart A
annfadf = pd.DataFrame(annfa, index=rspyf.index)
fig1, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(annfadf, label='annfa')
plt.legend(loc='upper left')
plt.title('ANN Regression Testing Chart A')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# ANN Regression Testing Chart B
annfbdf = pd.DataFrame(annfb, index=rspyf.index)
fig2, ax = plt.subplots()
ax.plot(rspyf['rspy'])
ax.plot(annfbdf, label='annfb')
plt.legend(loc='upper left')
plt.title('ANN Regression Testing Chart B')
plt.ylabel('Arithmetic Returns')
plt.xlabel('Date')
plt.show()

# 4.6. ANN Regression Testing Forecasting Accuracy
annmaea = fa.mean_absolute_error(rspyf['rspy'], annfa)
annmaeb = fa.mean_absolute_error(rspyf['rspy'], annfb)
annmsea = fa.mean_squared_error(rspyf['rspy'], annfa)
annmseb = fa.mean_squared_error(rspyf['rspy'], annfb)
annrmsea = np.sqrt(annmsea)
annrmseb = np.sqrt(annmseb)
print("== ANN Regression Testing Forecasting Accuracy ==")
print("")
print("Mean Absolute Error ", "A:", round(annmaea, 6), "B:", round(annmaeb, 6))
print("Mean Squared Error ", "A:", round(annmsea, 6), "B:", round(annmseb, 6))
print("Root Mean Squared Error ", "A:", round(annrmsea, 6), "B:", round(annrmseb, 6))