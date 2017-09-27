# -*- coding: utf-8 -*-

import logging
import datetime
import json
import codecs

from log import log



def file2dict(path):
    with open(path,'r') as f:
        s = f.read()
        return json.loads(s)

def read_config(path):
        try:
            config = file2dict(path)
            #print "config:",config
            for v in config:
                if type(v) is int:
                    log.warn('配置文件的值最好使用双引号包裹，使用字符串类型，否则可能导致不可知的问题')
            return config
        except ValueError,e:
            print e
            log.error('配置文件格式有误，请勿使用记事本编辑，推荐使用 notepad++ 或者 sublime text')



#