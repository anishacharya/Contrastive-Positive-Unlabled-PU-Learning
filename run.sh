#CKPT='/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/cifar10.dog_cat/pt-puCL-nP=2500/0/epoch=999.ckpt'
# python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name lp-puCL-nnPU-puPL=5000


python3 run_representation.py --exp_name case-control/nP=5k/bs=4096/lr=2.8/pi=0.6/pt-puNCE --dataset cifar10.dog_cat
