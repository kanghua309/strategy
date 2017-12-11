import numpy as np
from sklearn import preprocessing
import pandas as pd
from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.classifiers import CustomClassifier,Latest
class quantiles(CustomClassifier):
    inputs = []
    window_length = 1
    window_safe =  True
    dtype = np.int64
    missing_value = -1
    params = {'bins': None, }
    def compute(self, today, assets,out,factor,bins):
        print assets
        out[:] = pd.qcut(factor,bins,labels=False)