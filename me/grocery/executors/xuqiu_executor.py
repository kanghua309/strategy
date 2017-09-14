# -*- coding: utf-8 -*-
from executor import Executor
from me.grocery.broker.xueqiu import XueqiuLive
class XieqiuExecutor(Executor):
    def __init__(self,account,password,portfolio):
        Executor.__init__(self,'xq',account,password,[portfolio])
        self.broker = XueqiuLive(user='',account=account, password=password,portfolio_code=portfolio)
    def login(self):
        self.broker.login()

    def orders(self,targets):
        raise NotImplementedError()
    def balance(self):
        pass

    @property
    def portofolio(self):
        return self.broker.get_profolio_info()

if __name__ == '__main__':
   xqexec = XieqiuExecutor(account ='18618280998',password ='Threyear#3',protfolio='ZH1140387')
   xqexec.login()
   x= xqexec.portofolio
   print x