# -*- coding: utf-8 -*-


class Executor(object):
    def __init__(self,broker,account,password,commpasswd='',protfolios=[]):
        self.broker     = broker
        self.account    = account
        self.password   = password
        self.commpasswd = commpasswd
        self.protfolios = protfolios
        pass

    def login(self):
        pass

    def orders(self,targets=[]):
        raise NotImplementedError()

    def balance(self):
        pass

    def portofolio(self):
        pass
