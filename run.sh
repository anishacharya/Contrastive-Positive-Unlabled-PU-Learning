CKPT='/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/cifar10.dog_cat/pt-puCL-nP=2500/0/epoch=999.ckpt'

python3 run_linear_eval.py --mode ft --checkpoint $CKPT --exp_name lp-puCL-nnPU-puPL=5000

