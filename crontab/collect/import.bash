#!/bin/bash
cd `dirname $0`
echo `date`

set +H
export REPLACEMENT=`cat ~/workshop/collect/ready.txt |sed "s/[[:digit:]].*/\'&\',/"|tr -d '\n'`
sed -i "/BEGIN/{p;:a;N;/END/!ba;s/.*\n/$REPLACEMENT\n/}" ~/.zipline/extension.py
set -H

/data/kanghua/Envs/zipline/bin/python clean.py
~/Envs/zipline/bin/zipline ingest -b my-db-bundle
