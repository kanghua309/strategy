# -*- coding: utf-8 -*-
from .executor import Executor
from me.grocery.broker.xueqiu import XueqiuLive
class XieqiuExecutor(Executor):
    def __init__(self,account,password,portfolio):
        Executor.__init__(self,'xq',account,password,[portfolio])
        self.broker = XueqiuLive(user='',account=account, password=password,portfolio_code=portfolio)
    def login(self):
        self.broker.login()

    def orders(self,targets):
        for stock in targets:
            self.broker.adjust_weight(stock,targets[stock] * 100)

    def balance(self):
        pass

    @property
    def portofolio(self):
        return self.broker.get_profolio_info()
