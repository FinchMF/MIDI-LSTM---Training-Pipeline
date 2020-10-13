#!/bin/bash

if ! -d './data'
then 
    echo '[+] Configuring Build and Recieving'
    bash setup.sh

fi

echo '[i] Begin Pipeline... '
python composer_train.py 

echo '[i] Predictions and Results'
python composer_write.py 



