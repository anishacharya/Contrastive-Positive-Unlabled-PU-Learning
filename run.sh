CKPT='/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/cifar10.dog_cat/pt-puCL-nP=1000/0/epoch=999.ckpt'

python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name lp-puCL-nnPU-puPL=1000

