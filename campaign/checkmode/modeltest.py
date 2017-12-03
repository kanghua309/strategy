import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

#
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
# names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
# array = dataframe.values
# X = array[:,0:13]
# Y = array[:,13]
# print dataframe.head(10)



def model_cross_valid(X,Y):
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    def bulid_model(model_name):
        model = model_name()
        return model
    scoring = 'neg_mean_squared_error'
    # + random fest boost lstm gbdt

    for model_name in [LinearRegression,Ridge,Lasso,ElasticNet,KNeighborsRegressor,DecisionTreeRegressor,SVR,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor]:
        model = bulid_model(model_name)
        results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print(model_name,results.mean())

import numpy as np
def model_fit_and_test(TrainX,TrainY,TestX,TestY):
    def bulid_model(model_name):
        model = model_name()
        return model

    for model_name in [LinearRegression, Ridge, Lasso, ElasticNet, KNeighborsRegressor, DecisionTreeRegressor, SVR,
                       RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor]:
        model = bulid_model(model_name)
        model.fit(TrainX,TrainY)
        print("====================",model_name)
        resid = model.predict(TestX) - TestY
        #print resid
        print("Residual sum of squares: %f"% np.mean(resid ** 2))
        #print model.predict(TestX)
        #print TestY
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % model.score(TestX, TestY))

        from statsmodels.stats.stattools import jarque_bera
        _, pvalue, _, _ = jarque_bera(resid)
        print ("Test Residuals Normal", pvalue)

        from statsmodels import regression, stats
        import statsmodels.api as sms
        import statsmodels.stats.diagnostic as smd
        # xs_with_constant = sms.add_constant(np.column_stack((X1,X2,X3,X4)))
        xs_with_constant = sms.add_constant(TestX)
        _, pvalue1, _, _ = stats.diagnostic.het_breushpagan(resid, xs_with_constant)
        print ("Test Heteroskedasticity", pvalue1)
        ljung_box = smd.acorr_ljungbox(resid, lags=10)
        print "Lagrange Multiplier Statistics:", ljung_box[0]
        print "Test Autocorrelation P-values:", ljung_box[1]
# model_cross_valid(X,Y)
