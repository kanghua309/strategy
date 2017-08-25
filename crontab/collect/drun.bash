cd `dirname $0`
echo `date`

echo "daily fetcher stock data from tushare now"
/data/kanghua/Envs/zipline/bin/python stock_daily_fetcher.py

echo "daily update zipline bundle now"
bash import.bash 
