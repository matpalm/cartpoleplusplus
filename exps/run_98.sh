use_gpu
R=runs/98/
mkdir $R
nice ./naf_cartpole.py \
--use-raw-pixels --use-batch-norm \
--action-force=100 --action-repeats=3 --steps-per-repeat=4 --num-cameras=2 \
--replay-memory-size=200000 --replay-memory-burn-in=10000 \
--optimiser=Momentum --optimiser-args='{"learning_rate": 0.01, "momentum": 0.9}' \
--ckpt-dir=$R/ckpts --event-log-out=$R/events >$R/out 2>$R/err
