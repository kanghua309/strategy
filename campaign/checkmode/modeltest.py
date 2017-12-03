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


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:,0:13]
Y = array[:,13]
print dataframe.head(10)

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
