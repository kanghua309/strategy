# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:01:42 2017

@author: kanghua
"""


from zipline.api import (
    symbol,
    get_datetime,
)


class PortfolioManager:
    class BlackListManager:
         def __init__(self,expire):
             self.expire = expire #datetime.timedelta(hours=10)
             self.black_list = [];
         def addEquite(self,sym,dt):  #检查超时，自动移除
             self.black_list.insert(0, (sym, dt))
             for i in range(len(self.black_list) -1, -1, -1):
                 odt = self.black_list[i][1]
                 #print("debug print:",dt,odt,(dt - odt).days)
                 if (dt - odt).days > self.expire: 
                     self.black_list.pop(i)
                 else:
                     break
    
         def isExist(self,sym,dt):
             #print("-------blacklist:",self.blist)
             for i in range(len(self.black_list) -1, -1, -1):
                 odt = self.black_list[i][1]
                 #print("debug print:",dt,odt,(dt - odt).days)
                 if (dt - odt).days > self.expire: 
                     self.black_list.pop(i)
                 else:
                     break
             #print("debug print ------------BlackListManager isExist",sym,self.blist)
             return True if sym in self.black_list else False
              
         def getEquiteCount(self):
             return len(self.black_list)
         
         def getExpireTime(self):
             return len(self.expire)
         
    def __init__(self,context,blexpire,stopwin,stoploss,slotnum):
        self.context = context
        #self.blexpire = blexpire
        self.stopwin = stopwin
        self.stoploss = stoploss
        self.slotnum = slotnum;
        self.blmanager = self.BlackListManager(blexpire)
        self.free_slotnum = self.slotnum
        print(context,context.portfolio.positions)
        
    def getPositionCount(self):
        return len(self.context.portfolio.positions)
    
    def getBlackListCount(self):
        return self.blmanager.getEquiteCount()
    
    def getStopLossPosition(self):
        return self._getPositions(self.stoploss)
    
    def getStopWinPosition(self):
        return self._getPositions(self.stopwin)
    
    def _getPositions(self,ratio):
        poslist = []
        for stock in self.context.portfolio.positions:
            pos = self.context.portfolio.positions[stock]
            if ratio >= 0:
                if (pos.last_sale_price - pos.cost_basis) / pos.cost_basis > ratio:
                    poslist.append(pos)
            else:
                if (pos.last_sale_price - pos.cost_basis) / pos.cost_basis < ratio:
                    poslist.append(pos)
            #print("==",pos.last_sale_price,pos.cost_basis)
        #print("++",poslist)
        '''
        print("--",sorted(poslist,key=lambda pos:abs((pos.last_sale_price - pos.cost_basis)/pos.cost_basis), reverse=True))
        if len(poslist) == 0:
           return poslist
        else:
        '''
        return sorted(poslist,key=lambda pos:abs((pos.last_sale_price - pos.cost_basis)/pos.cost_basis), reverse=True) 
    
    def isInPositions(self,sym):
        #print("debug print ------------isInPositions",sym,len(self.context.portfolio.positions))
        #for stock in self.context.portfolio.positions:
        #    print(self.context.portfolio.positions[stock])
        return True if symbol(sym) in self.context.portfolio.positions else False
    
    def isStopLoss(self,sym):
        #print(self._getPositions(self.stoploss))
        return True if symbol(sym) in self._getPositions(self.stoploss) else False
    
    def isStopWin(self,sym):
        #print(self._getPositions(self.stopwin))
        return True if symbol(sym) in self._getPositions(self.stopwin) else False
    
    def addToBlackList(self,sym):
        self.blmanager.addEquite(sym,get_datetime())
        
    def isInBlackList(self,sym):
        return self.blmanager.isExist(sym,get_datetime())
    
    def getTotalSlotNum(self):
        return self.slotnum
    
    '''
    def getFreeSlotNum(self):
        return self.slotnum - len(self.portfolio.positions)
    '''
    def allocFreeSlot(self):
        self.free_slotnum -= 1
    
    def returnSlot(self):
        self.free_slotnum += 1
    
    def getFreeSlotNum(self):
        return self.free_slotnum
        
    
    def getPortfolioValue(self):
        return self.context.portfolio.profolio_value
    
    def getPositionsExposure(self):
        return self.context.portfolio.positions_exposure
    
    def getCurrentCash(self):
        return self.context.portfolio.cash
    