#!/bin/bash

for i in `seq 3050 3082`;
do
    python src/third2.py $i
done

for i in `seq 373 392`;
do
    python src/third.py $i
done