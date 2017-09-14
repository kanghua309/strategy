# -*- coding: utf-8 -*-


class Executor(object):
    def __init__(self,broker,account,password,commpasswd='',portfolio=[]):
        self.broker     = broker
        self.account    = account
        self.password   = password
        self.commpasswd = commpasswd
        self.portfolio = portfolio
        pass

    def login(self):
        pass

    def orders(self,target):
        raise NotImplementedError()

    def balance(self):
        pass

    def portofolio(self):
        pass
