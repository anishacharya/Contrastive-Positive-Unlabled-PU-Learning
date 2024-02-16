# export CUDA_LAUNCH_BLOCKING=1
# LP / FT evaluate a pretrained model
#CKPT="logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
#python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name nP=5k/lp-puCL-ce_sup --dataset cifar10.dog_cat


#CKPT="/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
#python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name nP=5k/lp-puCL-pupL --dataset cifar10.dog_cat --puPL True

# ----- Embedding Visualization
#CKPT="logs/checkpoints/imagenet/single-dataset/sup-sCL/0/epoch=99.ckpt"
#python3 run_tsne.py --checkpoint $CKPT --dataset imagenet --fig_name tsne_plots/imagenet-sup-sCL


# train from scratch
#python3 run_linear_eval.py --mode ft --exp_name sup-ce  --dataset cifar10.hard

# Train contrastive encoder from scratch
# python3 run_representation.py --exp_name case-control/nP=9k/puNCE-pi=0.9 --dataset imagenet
python3 run_representation.py --exp_name case-control/nP=1k/puNCE-pi=0.3 --dataset fmnist.2



##### Linear Probing
#CKPT="logs/checkpoints/imagenet/gamma=0.05/puCL/0/epoch=99.ckpt"
#CKPT="logs/checkpoints/cifar10.hard/nP=10k-puCL/0/epoch=299.ckpt"
#python3 run_lp.py --checkpoint $CKPT --dataset 'cifar10.hard' --mixup True #--puPL True --algo 'PUkMeans'