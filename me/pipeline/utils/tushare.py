# -*- coding: utf-8 -*-
"""
Created on Tue May 02 18:35:00 2017

@author: kanghua
"""



import tushare as ts
import pickle 



def load_tushare_df(df_type):
    file = 'ts.' + df_type + '.dat'
    try:
        obj = pickle.load(open(file,"rb"))
    except:
        print("---load in the fly",df_type)
        if df_type == "basic":
           obj = ts.get_stock_basics()
        elif df_type == "sme":
           obj = ts.get_sme_classified()
        elif df_type == "gem":
           obj=ts.get_gem_classified()
        elif df_type == "industry":
           obj = ts.get_industry_classified()
        elif df_type == "st":
           obj = ts.get_st_classified()
        else:
            raise Exception("Error TSshare Type!!!")
        pickle.dump(obj,open(file,"wb",0))
    else:
        print("----read from file",df_type)
        pass
    return obj