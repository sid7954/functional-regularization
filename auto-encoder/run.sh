#!/bin/bash

pred_size=(100 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)

for p in "${pred_size[@]}"; do
	python func_reg.py --pred_size $p --num_trials 10 
	python end_end.py --pred_size $p --num_trials 1000
done
