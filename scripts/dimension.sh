#! /bin/sh
set -eu

for k in 80 100 120 140 160 180 200; do
    python svm_flatten.py --train_amount 5000 --test_amount 2000 --projection_dimension ${k} --nonlinear_mult True --sigma 0.1 >> dimension.log
done
