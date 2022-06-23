#! /bin/sh
set -eu

for run in {1..50}; do
    rm -rf dimension_results
    for k in 80 100 120 140 160 180 200; do
        python svm_flatten.py --train_amount 5000 --test_amount 2000 --projection_dimension ${k} --nonlinear_mult True --sigma 0.1 --test_name dimension >> dimension.log
    done
    cp dimension_results ${run}.dimension_results
    rm -rf dimension_results
done
