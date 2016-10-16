#!/usr/bin/env bash
set -ex
unset CUDA_VISIBLE_DEVICES  # ensure running on gpu

R=runs/85

mkdir -p $R/{naf,ddpg}

export ARGS="--use-raw-pixels --max-run-time=14400 --use-batch-norm --action-force=100"

export RR=$R/naf
./naf_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.01, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err

export RR=$R/ddpg
./ddpg_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.01, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err
