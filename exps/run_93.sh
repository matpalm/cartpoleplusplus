#!/usr/bin/env bash
set -ex
unset CUDA_VISIBLE_DEVICES  # ensure running on gpu

export R=runs/93
mkdir $R

./naf_cartpole.py \
--use-raw-pixels \
--num-cameras=2 --action-repeats=3 --steps-per-repeat=4 \
--replay-memory-size=12000 \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.01, "momentum": 0.9}' \
--use-batch-norm \
--ckpt-dir=$R/ckpts --event-log-out=$R/events >$R/out 2>$R/err &
