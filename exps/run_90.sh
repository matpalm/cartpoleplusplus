#!/usr/bin/env bash
set -ex
export CUDA_VISIBLE_DEVICES=""  # ensure running on cpu

R=runs/90

ARGS="--action-force=100 --action-repeats=3 --steps-per-repeat=4"

export RR=$R/fixed
mkdir -p $RR
nice ./naf_cartpole.py $ARGS \
--reward-calc="fixed" \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &
sleep 1

export RR=$R/angle
mkdir -p $RR
nice ./naf_cartpole.py $ARGS \
--reward-calc="angle" \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &
sleep 1

export RR=$R/action
mkdir -p $RR
nice ./naf_cartpole.py $ARGS \
--reward-calc="action" \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &

export RR=$R/angle_action
mkdir -p $RR
nice ./naf_cartpole.py $ARGS \
--reward-calc="angle_action" \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$RR/ckpts --event-log-out=$RR/events >$RR/out 2>$RR/err &

