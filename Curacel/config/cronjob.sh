#!/bin/bash

echo cronjob.sh called: `date`
HOME=/Users/akinwande.komolafe
PYTHONPATH=/Users/akinwande.komolafe/opt/anaconda3/bin
cd /Users/akinwande.komolafe/Documents/Curacel
python ./curacel.py >>logs.out 2>&1 1>/dev/null