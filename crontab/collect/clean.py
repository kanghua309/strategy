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


def _modify(stock,conn):
     try:
        print "check and modify  ------ :", stock
        query = "update '%s' set date = datetime(strftime('%s',date),'unixepoch') where length(date) == 10" % (stock,'%s')
        #print query
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
     except Exception, arg:
        print "exceptionc:", stock, arg
        raise SystemExit(-1)

def _modify2(stock,conn):
     try:
        print "check and modify  ------ :", stock
        query = "update '%s' set date ='2017-08-28 00:00:00'  where date == '2017-08-29 00:00:00'" % (stock)
        #print query
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
     except Exception, arg:
        print "exceptionc:", stock, arg
        raise SystemExit(-1)




def _clean(stock, conn):
    try:
        print "clean ------ :", stock
        query = "select * from '%s' order by date" % stock
        df = pd.read_sql(query, conn)
        print "before",df.tail(5)
        cur = conn.cursor()
        query = "delete from '%s' where rowid not in(select max(rowid) from '%s' group by date)" % (stock, stock)
        cur.execute(query)
        conn.commit()
    except Exception, arg:
        print "exceptionc:", stock, arg
        raise SystemExit(-1)


def  xxx(stock,conn):
     try:
        print   "do check and modify  ------ :", stock
        query = "delete from '%s' where date='2017-08-25' and volume = 1518212" % (stock)
        #print query
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
     except Exception, arg:
        print "exceptionc:", stock, arg
        raise SystemExit(-1)



conn = sqlite3.connect('History.db', check_same_thread=False)
alreadylist = getAllStockSaved()
_modify('601326',conn)
_clean('601326',conn)

#xxx('603987',conn)
#_clean('603987',conn)
#_clean('603979',conn)

'''
for stock in alreadylist.name:
    _modify(stock,conn)
    #_clean('600363',conn)
    #_modify('600363',conn)
    _clean(stock,conn)
'''

conn.close()
