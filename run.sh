# LP / FT evaluate a pretrained model
#CKPT="/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
#python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name nP=5k/lp-puCL-pupL --dataset cifar10.dog_cat

#CKPT="/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
#python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name nP=5k/lp-puCL-pupL --dataset cifar10.dog_cat --puPL True

# train from scratch
#python3 run_linear_eval.py --mode lp --exp_name gamma=0.5/nnPU --dataset imagenet


# Train contrastive encoder from scratch
# python3 run_representation.py --exp_name gamma=0.05/sCL --dataset imagenet
python3 run_representation.py --exp_name single_dataset/ssCL --dataset cifar10.dog_cat