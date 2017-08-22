#!/bin/bash
set +H
export REPLACEMENT=`cat ~/workshop/collect/ready.txt |sed "s/[[:digit:]].*/\'&\',/"|tr -d '\n'`
sed -i "/BEGIN/{p;:a;N;/END/!ba;s/.*\n/$REPLACEMENT\n/}" ~/.zipline/extension.py
set -H
