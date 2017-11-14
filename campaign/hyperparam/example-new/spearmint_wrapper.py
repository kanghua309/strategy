#!/usr/bin/python
import sys
from zipline.utils.run_algo import _run
import pandas as pd
import os
import datetime as datetime
from pandas.tslib import Timestamp
from zipline.data.bundles import register
from zipline.data.bundles.viadb import viadb

def main(job_id,D):
    print "job_id",job_id," params:",D

    equities1 = {}
    register(
        'my-db-bundle',  # name this whatever you like
        viadb(equities1),
        calendar='SHSZ'
    )

    parsed={}
    parsed['initialize']= None
    parsed['handle_data']= None
    parsed['before_trading_start']= None
    parsed['analyze']= None
    parsed['algotext']= None
    parsed['defines']= ()
    parsed['capital_base']= 1000000
    parsed['data']= None

    parsed['bundle']='my-db-bundle'
    #parsed['bundle']='YAHOO'
    #parsed['bundle_timestamp']=None
    parsed['bundle_timestamp']= pd.Timestamp.utcnow()
    parsed['start']= Timestamp('2017-03-01 13:30:00+0000', tz='UTC')
    parsed['end']  = Timestamp('2017-06-01 13:30:00+0000', tz='UTC')
    parsed['algofile']= open('/data/kanghua/workshop/strategy/campaign/hyperparam/example-new/zipline_strategy.py')
    parsed['data_frequency']='daily'

    parsed['print_algo']= False
    parsed['output']= 'os.devnull'
    parsed['local_namespace']= None
    parsed['environ']= os.environ
    parsed['bm_symbol']= None



    # Below what we expect spearmint to pass us
    # parsed['algo_params']=[47,88.7,7.7]
    # D={}
    # D['timeperiod']=10
    # D['nbdevup']=1.00
    # D['nbdevdn']=1.00
    parsed['algo_params']=D
    perf = _run(**parsed)
    StartV=perf['portfolio_value'][ 0]
    EndV=perf['portfolio_value'][-1]
    # spearmint wants to minimize so return negative profit
    OPTIM=(StartV-EndV)
    return OPTIM

if __name__ == "__main__":
    # Below will be overridden by spearmint when it runs
    main(101,{'leverage': 2,'history_depth':15,'top_num':2,'bottom_num':2})



