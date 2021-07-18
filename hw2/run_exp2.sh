#!/bin/bash

bb="1000"
lr="0.005"
seeds="1 2 3"
for b in $bb
do
for l in $lr
do
for s in $seeds
do
	echo "launching exp with batch $b and lr=$l"
	python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 1.0 -n 100 -l 2 -s 64 -b ${b} -lr ${l} -rtg --exp_name q2_b${b}_r${l} --seed $s &
done
done
done

