#!/bin/bash
today=`date +%Y-%m-%d`
firstday=`date -d -100day +%y-%m-%d`

zipline run -f ./factortest2.py --bundle my-db-bundle --start $firstday --end $today -o /dev/null

