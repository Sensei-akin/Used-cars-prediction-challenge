#!/bin/bash

echo cronjob.sh called: `date`
HOME=/Users/akinwande.komolafe
PYTHONPATH=/Users/akinwande.komolafe/opt/anaconda3/bin
cd /Users/akinwande.komolafe/Documents/Curacel
python ./app.py >> api.out