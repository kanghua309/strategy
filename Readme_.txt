zipline run -f ./factortest2.py --bundle my-db-bundle --start 2011-1-3 --end 2017-5-26 -o my_out.pickle



#0 0 * * * flock -xn /data/kanghua/workshop/collect/fetcher.lock -c 'bash -l /data/kanghua/workshop/collect/run.bash > /tmp/stock-fetcher.out 2>&1'

# run bak pickle data
0 16 * * 1-5 flock -xn /data/kanghua/workshop/collect/dbak/fetcher.lock -c 'bash -l /data/kanghua/workshop/collect/dbak/run.bash > /tmp/dbak-stock-fetcher.out 2>&1'

# run update stock data
0 17 * * 1-5 flock -xn /data/kanghua/workshop/collect/dfetcher.lock -c 'bash -l /data/kanghua/workshop/collect/drun.bash > /tmp/dstock-fetcher.out 2>&1'


# run train ...
0 20 * * 5 flock -xn /data/kanghua/workshop/Market2vector-GPU/do.lock -c 'bash -l /data/kanghua/workshop/Market2vector-GPU/run.bash > /tmp/train.out 2>&1'
# run strategy ....
0 18 * * 0 flock -xn /data/kanghua/workshop/strategy/do.lock -c 'bash -l /data/kanghua/workshop/strategy/run.bash > /tmp/strategy.out 2>&1'



