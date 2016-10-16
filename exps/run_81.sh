#!/usr/bin/env bash
set -ex

mkdir -p runs/81/{sgd,mom,adam}{,_bn}

export ARGS="--use-raw-pixels --max-run-time=3600 --dont-do-rollouts --event-log-in=runs/80/events"

export R=81/sgd
./naf_cartpole.py $ARGS \
--optimiser=GradientDescent --optimiser-args='{"learning_rate": 0.001}' \
--ckpt-dir=runs/$R/ckpts --event-log-out=runs/$R/events >runs/$R/out 2>runs/$R/err

export R=81/sgd_bn
./naf_cartpole.py $ARGS \
--optimiser=GradientDescent --optimiser-args='{"learning_rate": 0.001}' \
--use-batch-norm \
--ckpt-dir=runs/$R/ckpts --event-log-out=runs/$R/events >runs/$R/out 2>runs/$R/err

export R=81/mom
./naf_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.001, "momentum": 0.9}' \
--ckpt-dir=runs/$R/ckpts --event-log-out=runs/$R/events >runs/$R/out 2>runs/$R/err

export R=81/mom_bn
./naf_cartpole.py $ARGS \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.001, "momentum": 0.9}' \
--use-batch-norm \
--ckpt-dir=runs/$R/ckpts --event-log-out=runs/$R/events >runs/$R/out 2>runs/$R/err

export R=81/adam
./naf_cartpole.py $ARGS \
--optimiser=Adam --optimiser-args='{"learning_rate": 0.001}' \
--ckpt-dir=runs/$R/ckpts --event-log-out=runs/$R/events >runs/$R/out 2>runs/$R/err

export R=81/adam_bn
./naf_cartpole.py $ARGS \
--optimiser=Adam --optimiser-args='{"learning_rate": 0.001}' \
--use-batch-norm \
--ckpt-dir=runs/$R/ckpts --event-log-out=runs/$R/events >runs/$R/out 2>runs/$R/err
