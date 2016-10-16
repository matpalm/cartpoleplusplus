#!/usr/bin/env bash
set -ex

R=runs/82

mkdir -p $R/{sgd,mom}{,_bn}

export ARGS="--use-raw-pixels --max-run-time=3600 --dont-do-rollouts --event-log-in=runs/80/events"

export RR=$R/sgd
./naf_cartpole.py $ARGS \
--optimiser=GradientDescent --optimiser-args='{"learning_rate": 0.01}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err

export RR=$R/sgd_bn
./naf_cartpole.py $ARGS \
--optimiser=GradientDescent --optimiser-args='{"learning_rate": 0.01}' \
--use-batch-norm \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err

export RR=$R/mom
./naf_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.01, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err

export RR=$R/mom_bn
./naf_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.01, "momentum": 0.9}' \
--use-batch-norm \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err
