import tushare as ts
import pandas as pd

#info=ts.get_stock_basics()  

#print(type(info))
#print(len(info))

#x = info.at['603617','timeToMarket']


#print info.loc['603617']

#x = ts.get_h_data('300503')
#print x[['open','high','close','low','volume']]
#print x.index
#print(x.head(10))
#print x[['open','high']]

#x = ts.get_today_all()
#print x
import pickle 
import time
filename = time.strftime("%d-%m-%Y")
retry = 0
while True:
   try:
       df = ts.get_today_all()
       pickle.dump(df,open(filename,"wb",0))
       break
   except:
       retry += 1
       if retry == 10:
          break
      
     



