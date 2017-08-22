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



def _check(stock, conn):
    print "update ----- :", stock
    query = "select * from '%s' order by date" % stock
    try:
    	df = pd.read_sql(query, conn)
    	df = df.set_index('date')
    except:
        print "stock '%s' read failed" % (stock)
        return False
    if dt.now().weekday() == 5:  
    	yesterday = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=1))[:10]
    elif dt.now().weekday() == 6: #sunday
        yesterday = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=2))[:10]
    elif dt.now().weekday() == 0:  
    	yesterday = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=3))[:10]
    else:
        yesterday = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=1))[:10]
    print "yesterday:",yesterday , "last day:",df.ix[-1].name[:10]
    if yesterday == df.ix[-1].name[:10]:
    	return 0
    elif yesterday > df.ix[-1].name[:10]: #stop exchange
    	return 1
    else :
        return -1

def _update(stock, conn):
    try:
        print "update ----- :", stock
        query = "select * from '%s' order by date" % stock
        df = pd.read_sql(query, conn)
        df = df.set_index('date')
        print "sql saved:", df.tail(1),df.ix[-1],df.ix[-1].name
        if dt.now().weekday() == 5:
            yesterday = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=1))[:10]
        elif dt.now().weekday() == 6:
            yesterday = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=2))[:10]
        #elif dt.now().weekday() == 0:  
    	#    yesterday = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=10))[:10]
        else:
            yesterday = str(pd.Timestamp(dt.now()))[:10]
        print "today:",today
        if yesterday != df.ix[-1].name[:10]:
            df = ts.get_h_data(stock, start=df.ix[-1].name[:10], retry_count=5, pause=1)
            print "read from tu:",df.head(1)
            df[['open', 'high', 'close', 'low', 'volume']].to_sql(stock, conn, if_exists='append')
            import time
            time.sleep(10)
    except Exception, arg:
        print "exceptionu:", stock, arg
        #errorlist.append(stock)


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


import pickle
begin = datetime.datetime.now()
today = str(pd.Timestamp(dt.now()))[:10]
conn = sqlite3.connect('History.db', check_same_thread=False)
df  = ts.get_today_all()
#df = pickle.load(open('mytest',"rb"))
ind = 0
for index,row in df.iterrows():  
    df = pd.DataFrame(row).T
    print df
    df.index = [today]
    df.index.name = 'date'
    print df.index
    stock = df.at[today,'code']
    print stock
    flag = _check(stock,conn)
    if flag == 0:
 	print ("%s stock %s check pass ,load today data" % (stock,today)) 
        if df.at[today,'trade'] == 0:
           ind +=1 
 	   print ("%s stock %s stop exchange" % (stock,today)) 
           continue;
        df.rename(
                  columns={
                   'trade': 'close',
                  },
                  inplace=True,
                  )
   	df[['open', 'high', 'close', 'low', 'volume']].to_sql(stock, conn, if_exists='append')
    elif flag < 0 :
        print ("stock %s miss data" % stock) 
        _update(stock,conn)
    else:
        print ("%s today has load yet" % stock)
    ind +=1
    print "---------------------------------------------------count ",ind

errorlist = []
alreadylist = getAllStockSaved()
for stock in alreadylist:
    _clean(stock,conn)

conn.close()
record()
end = datetime.datetime.now()
print "run time:", end - begin

