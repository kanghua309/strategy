#!/bin/bash
cd `dirname $0`
echo `date`

cp /data/kanghua/workshop/collect/History.db /tmp -rf
/data/kanghua/3part/sqlite3/bin/sqlite3 /tmp/History.db 'drop table predict'
rm /data/kanghua/workshop/large-repo/History.db.tar.gz -rf
tar zvcf /data/kanghua/workshop/large-repo/History.db.tar.gz /tmp/History.db
cd /data/kanghua/workshop/large-repo/
git add History.db.tar.gz
git commit -m "..........."
git push -u origin master

