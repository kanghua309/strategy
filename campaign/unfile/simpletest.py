
# Linear Regression
import pandas
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
print dataframe
print "========================"
print array

TRAIN_X = array[0:400,0:13]
TRAIN_Y = array[0:400,13]

TEST_X = array[400:,0:13]
TEST_Y = array[400:,13]


# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]
#
# # Split the targets into training/testing sets
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test = diabetes.target[-20:]



print TRAIN_X
print TRAIN_Y
print "----------------",np.shape(TRAIN_X),np.shape(TRAIN_Y),np.shape(TEST_X),np.shape(TEST_Y)
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, TRAIN_X, TRAIN_Y, cv=kfold, scoring=scoring)
print(results)
print(results.mean())

res = model.fit(TRAIN_X, TRAIN_Y)
print res

print('Coefficients: ', model.coef_)
print model.residues_
print("Residual sum of squares: %.2f"
      % np.mean((model.predict(TEST_X) - TEST_Y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(TEST_X, TEST_Y))

resid = model.predict(TEST_X) - TEST_Y


#http://www.statsmodels.org/devel/examples/notebooks/generated/regression_diagnostics.html
import numpy as np
import pandas as pd
from statsmodels import regression
import statsmodels.api as sms
import statsmodels.stats.diagnostic as smd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
from statsmodels.compat import lzip



from statsmodels.stats.stattools import jarque_bera
_, pvalue, _, _ = jarque_bera(resid)
print ("Test Residuals Normal",pvalue)

# name = ['Chi^2', 'Two-tail probability']
# test = sms.omni_normtest(results.resid)
# print lzip(name, test)



###############################################
from statsmodels import regression, stats

#xs_with_constant = sms.add_constant(np.column_stack((X1,X2,X3,X4)))
xs_with_constant = sms.add_constant(TEST_X)

_, pvalue1, _, _ = stats.diagnostic.het_breushpagan(resid, xs_with_constant)

print ("Test Heteroskedasticity",pvalue1)

############################################
ljung_box = smd.acorr_ljungbox(resid, lags = 10)
print "Lagrange Multiplier Statistics:", ljung_box[0]
print "Test Autocorrelation P-values:", ljung_box[1]

#############################################
from statsmodels.tsa.stattools import coint, adfuller
def check_for_stationarity(X, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print 'p-value = ' + str(pvalue) + ' The series is likely stationary.'
        return True
    else:
        print 'p-value = ' + str(pvalue) + ' The series is likely non-stationary.'
        return False
print check_for_stationarity(TEST_Y)