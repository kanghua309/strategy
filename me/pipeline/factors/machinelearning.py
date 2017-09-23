# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor,Returns, Latest
from me.pipeline.factors.tsfactor import Fundamental
from sklearn import linear_model, decomposition, ensemble, preprocessing, isotonic, metrics

n_fwd_days = 5 # number of days to compute returns over

def shift_mask_data(X, Y, upper_percentile=70, lower_percentile=30, n_fwd_days=1):
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


def get_last_values(input_data):
    last_values = []
    for dataset in input_data:
        last_values.append(dataset[-1])
    return np.vstack(last_values).T


class FactorRegress(CustomFactor):
    init = False
    def compute(self, today, assets, out,returns,*inputs):
        if (not self.init) or (today.weekday == 0):  # Monday
            # Instantiate sklearn objects
            self.imputer = preprocessing.Imputer()
            self.scaler = preprocessing.MinMaxScaler()
            self.clf = ensemble.AdaBoostClassifier(n_estimators=100)

            # Stack factor rankings
            X = np.dstack(inputs)  # (time, stocks, factors)
            Y = returns  # (time, stocks)

            # Shift data to match with future returns and binarize
            # returns based on their
            X, Y = shift_mask_data(X, Y, n_fwd_days)

            X = self.imputer.fit_transform(X)
            X = self.scaler.fit_transform(X)

            # Fit the classifier
            self.clf.fit(X, Y)

            self.init = True

            # Predict
            # Get most recent factor values (inputs always has the full history)
        last_factor_values = get_last_values(inputs)
        last_factor_values = self.imputer.transform(last_factor_values)
        last_factor_values = self.scaler.transform(last_factor_values)

        # Predict the probability for each stock going up
        # (column 2 of the output of .predict_proba()) and
        # return it via assignment to out.
        out[:] = self.clf.predict_proba(last_factor_values)[:, 1]
