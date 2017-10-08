# -*- coding: utf-8 -*-

import tushare as ts
# from sqlalchemy import create_engine
import sqlite3
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime as dt
import threading
import datetime


# engine = create_engine('sqlite:///History.db', echo = False)

# conn=sqlite3.connect('History.db',check_same_thread = False)
# stocklist = []
# alreadylist = []
# stocksinfo = ts.get_stock_basics()
# print type(stocksinfo), stocksinfo.index[0],stocksinfo['timeToMarket'][0]

# query = "select name from sqlite_master where type='table' order by name"
# alreadylist = pd.read_sql(query, conn)
# conn.close()

def getAllStockSaved():
    conn = sqlite3.connect('History.db', check_same_thread=False)
    query = "select name from sqlite_master where type='table' order by name"
    alreadylist = pd.read_sql(query, conn)
    conn.close()
    return alreadylist


stocksinfo = ts.get_stock_basics()
errorlist = []
alreadylist = getAllStockSaved()

counter = 0
counter_lock = threading.Lock()
begin = datetime.datetime.now()


def process(stock):
    conn = sqlite3.connect('History.db')
    if stock not in list(alreadylist.name):
        _save(stock, conn)
    else:
        _update(stock, conn)
        #_clean(stock, conn)
    conn.close()
    global counter, counter_lock
    if counter_lock.acquire(): 
        counter += 1
        print "set counter:%s" % (counter)
    counter_lock.release() 

def process2(stock):
    conn = sqlite3.connect('History.db')
    if stock in list(alreadylist.name):
        _clean(stock, conn)
    conn.close()


def _save(stock, conn):
    try:
        print "save ----- :", stock
        marketday = stocksinfo.at[stock, 'timeToMarket']
        startday = pd.Timestamp(str(marketday))
        # print marketday,startday,str(startday)[:10]
        # df = ts.get_h_data(code, start=str(startday)[:10], retry_count = 5)
        df = ts.get_h_data(stock, start=str(startday)[:10], retry_count=5, pause=1)
        df = df.sort_index(ascending=True)
        # ma_list = [5,10,20,60]
        # for ma in ma_list:
        #    df['MA_' + str(ma)] = pd.rolling_mean(df.close, ma)
        # print df[['open','high','close','low','volume']].head(2)
        df[['open', 'high', 'close', 'low', 'volume']].to_sql(stock, conn, if_exists='append')
    except Exception, arg:
        print "exceptions:", stock, arg
        errorlist.append(stock)


def _update(stock, conn):
    try:
        print "update ----- :", stock
        query = "select * from '%s' order by date" % stock
        df = pd.read_sql(query, conn)
        df = df.set_index('date')

        print "sql saved:", df.tail(1),df.ix[-1],df.ix[-1].name
        if dt.now().weekday() == 5:
            today = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=1))[:10]
        elif dt.now().weekday() == 6:
            today = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=2))[:10]
        else:
            today = str(pd.Timestamp(dt.now()))[:10]
        print "today:",today
        if today != df.ix[-1].name[:10]:
            df = ts.get_h_data(stock, start=df.ix[-1].name[:10], retry_count=5, pause=1)
            print "read from tu:",df.head(1)
            df[['open', 'high', 'close', 'low', 'volume']].to_sql(stock, conn, if_exists='append')
            import time
            time.sleep(10)
    except Exception, arg:
        print "exceptionu:", stock, arg
        errorlist.append(stock)


def _clean(stock, conn):
    try:
        print "clean ------ :", stock
        query = "select * from '%s' order by date" % stock
        _ = pd.read_sql(query, conn)
        cur = conn.cursor()
        query = "delete from '%s' where rowid not in(select max(rowid) from '%s' group by date)" % (stock, stock)
        cur.execute(query)
        conn.commit()
    except Exception, arg:
        print "exceptionc:", stock, arg
        errorlist.append(stock)

        # conn.close()


def record():
    print "=======================ok====================="
    f = open('ready.txt', 'w')
    f.truncate()
    for code in list(set(alreadylist.name)):
        print code
        print >> f, code
    f.close()

    print "=======================error====================="
    f = open('error.txt', 'w')
    f.truncate()
    for err in list(set(errorlist)):
        print err
        print >> f, err
    f.close()


pool = ThreadPool(1)  # 4
pool.map(process, stocksinfo.index)
pool.close()
pool.join()

pool = ThreadPool(1)  # 4
pool.map(process2, stocksinfo.index)
pool.close()
pool.join()



alreadylist = getAllStockSaved()
record()

end = datetime.datetime.now()
print "run time:", end - begin
