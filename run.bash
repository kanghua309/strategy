#!/bin/bash
cd `dirname $0`
echo `date`

today=`date +%Y-%m-%d`
firstday=`date -d -100day +%Y-%m-%d`

~/Envs/zipline/bin/zipline run -f ./factortest2.py --bundle my-db-bundle --start $firstday --end $today -o /dev/null

