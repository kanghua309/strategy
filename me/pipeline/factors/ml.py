# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor,Returns, Latest
from me.pipeline.factors.tsfactor import Fundamental
from sklearn import linear_model, decomposition, ensemble, preprocessing, isotonic, metrics

n_fwd_days = 5 # number of days to compute returns over



def BasicFactorRegress(inputs, window_length, mask, trigger_date=None):

    class BasicFactorRegress(CustomFactor):
        #params = {'trigger_date': None, }
        init = False

        def __shift_mask_data(self,X, Y, upper_percentile=70, lower_percentile=30, n_fwd_days=1):
            # Shift X to match factors at t to returns at t+n_fwd_days (we want to predict future returns after all)
            shifted_X = np.roll(X, n_fwd_days, axis=0)

            # Slice off rolled elements
            X = shifted_X[n_fwd_days:]
            Y = Y[n_fwd_days:]

            n_time, n_stocks, n_factors = X.shape

            # Look for biggest up and down movers
            upper = np.nanpercentile(Y, upper_percentile, axis=1)[:, np.newaxis]
            lower = np.nanpercentile(Y, lower_percentile, axis=1)[:, np.newaxis]

            upper_mask = (Y >= upper)
            lower_mask = (Y <= lower)

            mask = upper_mask | lower_mask  # This also drops nans
            mask = mask.flatten()

            # Only try to predict whether a stock moved up/down relative to other stocks
            Y_binary = np.zeros(n_time * n_stocks)
            Y_binary[upper_mask.flatten()] = 1
            Y_binary[lower_mask.flatten()] = -1

            # Flatten X
            X = X.reshape((n_time * n_stocks, n_factors))

            # Drop stocks that did not move much (i.e. are in the 30th to 70th percentile)
            X = X[mask]
            Y_binary = Y_binary[mask]

            return X, Y_binary

        def __get_last_values(self,input_data):
            last_values = []
            for dataset in input_data:
                last_values.append(dataset[-1])
            return np.vstack(last_values).T
        def compute(self, today, assets,out,returns,*inputs):
            #print "------------------------------- BasicFactorRegress:",today,trigger_date
            if trigger_date != None and today != pd.Timestamp(trigger_date,tz='UTC'):  #仅仅是最重的预测factor给定时间执行了，其他的各依赖factor还是每次computer调用都执行，也流是每天都执行！ 不理想
                return
            #if trigger_date != None:
            #    today != np.datetime64(trigger_date)
            #    return None
            if (not self.init) :
            #if (not self.init) or (today.weekday == 0):  # Monday
                # Instantiate sklearn objects
                self.imputer = preprocessing.Imputer()
                self.scaler = preprocessing.MinMaxScaler()
                self.clf = ensemble.AdaBoostClassifier(n_estimators=100)
                #print "debug factor regress inputs:",len(inputs),inputs
                # Stack factor rankings
                X = np.dstack(inputs)  # (time, stocks, factors)  按时间组织了
                Y = returns  # (time, stocks)
                #print "debug factor regress X:", np.shape(X),X
                #print "debug factor regress Y:", np.shape(Y),Y

                # Shift data to match with future returns and binarize
                # returns based on their
                X, Y = self.__shift_mask_data(X, Y, n_fwd_days)  #n天的数值被展开成1维的了- 每个factor 按天展开
                #print "debug factor regress aftershift X:", np.shape(X),X
                #print "debug factor regress aftershift Y:", np.shape(Y),Y

                X = self.imputer.fit_transform(X)  #缺失值处理
                X = self.scaler.fit_transform(X)   #缩放处理

                # Fit the classifier
                self.clf.fit(X, Y)

                self.init = True

                # Predict
                # Get most recent factor values (inputs always has the full history)
            last_factor_values = self.__get_last_values(inputs)
            last_factor_values = self.imputer.transform(last_factor_values)
            last_factor_values = self.scaler.transform(last_factor_values)
            #print "debug factor regress last_factor_values:", np.shape(last_factor_values),last_factor_values

            # Predict the probability for each stock going up
            # (column 2 of the output of .predict_proba()) and
            # return it via assignment to out.

            out[:] = self.clf.predict_proba(last_factor_values)[:, 1] #每行中的列1

    return BasicFactorRegress(inputs=inputs,window_length=window_length,mask=mask)



