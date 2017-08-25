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


import datetime
def _check(stock, conn):
    print "check ----- :", stock
    query = "select * from '%s' order by date" % stock
    try:
    	df = pd.read_sql(query, conn)
    	df = df.set_index('date')
        stocklastsavedday = df.ix[-1].name[:10]
    except:
        print "stock '%s' read failed" % (stock)
        stocklastsavedday = '1970-01-01'

    if dt.now().weekday() == 5:  
    	lasttradeday = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=1))[:10]
    elif dt.now().weekday() == 6: #sunday
        lasttradeday = str(pd.Timestamp(dt.now()) - pd.Timedelta(days=2))[:10]
    else:
        lasttradeday = str(pd.Timestamp(dt.now()))[:10]
    print "lasttradeday %s,stocklastsaveday %s" % (lasttradeday,stocklastsavedday)
    gap = datetime.datetime.strptime(lasttradeday,'%Y-%m-%d') - datetime.datetime.strptime(stocklastsavedday,'%Y-%m-%d') 
    return gap.days,stocklastsavedday

def _update(stock,begin, conn):
    print "update ----- :", stock
    try:
        df = ts.get_h_data(stock, start=begin, retry_count=5, pause=1)
        df[['open', 'high', 'close', 'low', 'volume']].to_sql(stock, conn, if_exists='append')
        import time
        time.sleep(10)
    except Exception, arg:
        print "exceptionu:", stock, arg
        raise SystemExit(-1)
        #errorlist.append(stock)

def _clean(stock, conn):
    try:
        #print "clean ------ :", stock
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
'''
retry = 0
while True:
   try:
       df = ts.get_today_all()
       #pickle.dump(df,open(filename,"wb",0))
       break
   except:
       retry += 1
       if retry == 10:
           conn.close()
           raise SystemExit(-1)
       time.sleep(60)
'''
df = pickle.load(open('dbak/24-08-2017',"rb"))
ind = 0
for index,row in df.iterrows():  
    df = pd.DataFrame(row).T
    print df
    df.index = [today]
    df.index.name = 'date'
    print df.index
    stock = df.at[today,'code']
    print stock
    gapday,stocklastsavedday = _check(stock,conn)
        
    if  gapday == 1 : 
 	print ("to load today data" ,stock,today) 
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
    elif gapday > 1 :
        print ("stock %s miss data" % stock) 
        if gapday < 10:        
            _update(stock,stocklastsavedday,conn)
    else:
        print ("%s today has load yet" % stock)
    ind +=1
    print "---------------------------------------------------count ",ind

errorlist = []
alreadylist = getAllStockSaved()
for stock in alreadylist.name:
    _clean(stock,conn)

conn.close()

record()
end = datetime.datetime.now()
print "run time:", end - begin

