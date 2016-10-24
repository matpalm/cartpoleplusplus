use_cpu
export R=runs/97
mkdir -p $R
nice ./naf_cartpole.py \
--action-force=100 --action-repeats=3 --steps-per-repeat=4 \
--reward-calc="angle" \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.0001, "momentum": 0.9}' \
--ckpt-dir=$R/ckpts --event-log-out=$R/events >$R/out 2>$R/err
