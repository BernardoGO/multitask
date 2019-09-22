#!/bin/bash
for i in `seq 300 335`;
do
    CUDA_VISIBLE_DEVICES='' python src/third3.py $i
done

for i in `seq 3000 3035`;
do
    CUDA_VISIBLE_DEVICES='' python src/third4.py $i
done