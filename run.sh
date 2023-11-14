#CKPT='/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/cifar10.dog_cat/pt-puCL-nP=2500/0/epoch=999.ckpt'
# python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name lp-puCL-nnPU-puPL=5000


python3 run_representation.py --exp_name nP=10k/dcl-pi=0.4 --dataset cifar10.1
