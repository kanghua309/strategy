#!/bin/bash
cd `dirname $0`
echo `date`
/data/kanghua/Envs/zipline/bin/python stock_fetcher.py
