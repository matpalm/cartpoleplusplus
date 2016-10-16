#!/usr/bin/env bash
set -ex
unset CUDA_VISIBLE_DEVICES  # ensure running on gpu

R=runs/84

mkdir -p $R/{naf,ddpg}_mom_bn

export ARGS="--use-raw-pixels --max-run-time=14400 --dont-do-rollouts --event-log-in=runs/80/events"

export RR=$R/naf_mom_bn
./naf_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.01, "momentum": 0.9}' \
--use-batch-norm \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err

export RR=$R/ddpg_mom_bn
./ddpg_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.01, "momentum": 0.9}' \
--use-batch-norm \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err
