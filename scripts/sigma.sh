#! /bin/sh
set -eu

for run in {1..50}; do
    rm -rf sigma_results
    for sigma in 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2; do
        python svm_flatten.py --train_amount 5000 --test_amount 2000 --projection_dimension 100 --nonlinear_mult True --sigma ${sigma} --test_name sigma >> sigma.log
    done
    cp sigma_results ${run}.sigma_results
    rm -rf sigma_results
done
