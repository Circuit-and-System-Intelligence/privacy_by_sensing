#! /bin/sh
set -eu

for sigma in 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2; do
    python svm_flatten.py --train_amount 5000 --test_amount 2000 --projection_dimension 100 --nonlinear_mult True --sigma ${sigma} >> sigma.log
done
