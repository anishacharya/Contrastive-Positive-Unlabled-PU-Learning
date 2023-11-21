export CUDA_LAUNCH_BLOCKING=1
# LP / FT evaluate a pretrained model
CKPT="logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name nP=5k/puCL-nnPU --dataset cifar10.dog_cat

#CKPT="/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
#python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name nP=5k/lp-puCL-pupL --dataset cifar10.dog_cat --puPL True

# train from scratch
# python3 run_lp.py --mode ft --exp_name gamma=0.5/nnPU --dataset cifar10.dog_cat


# Train contrastive encoder from scratch
# python3 run_representation.py --exp_name gamma=0.05/sCL --dataset imagenet
#python3 run_representation.py --exp_name single_dataset/nP=500/sCL --dataset cifar10.dog_cat