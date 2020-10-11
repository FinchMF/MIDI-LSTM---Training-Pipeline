#!/bin/bash

echo 'Directory and Data Build and Recieving'

mkdir run && mkdir data
mkdir run/generative_comps && mkdir data/melody

echo 'Retrieving Data'
python fetch_midi_data.py 