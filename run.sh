# export CUDA_LAUNCH_BLOCKING=1
# LP / FT evaluate a pretrained model
#CKPT="logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
#python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name nP=5k/lp-puCL-ce_sup --dataset cifar10.dog_cat


#CKPT="/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
#python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name nP=5k/lp-puCL-pupL --dataset cifar10.dog_cat --puPL True

# ----- Embedding Visualization
#CKPT="logs/checkpoints/imagenet/single-dataset/gamma=0.5/sCL/0/epoch=99.ckpt"
#python3 run_tsne.py --checkpoint $CKPT --dataset imagenet --fig_name tsne_plots/imagenet-sCL-gamma=5


# train from scratch
#python3 run_linear_eval.py --mode ft --exp_name sup-ce  --dataset cifar10.hard

# Train contrastive encoder from scratch
#python3 run_representation.py --exp_name sup-sCL --dataset imagenet
#python3 run_representation.py --exp_name single_dataset/nP=500/sCL --dataset cifar10.dog_cat



##### Linear Probing
#CKPT="logs/checkpoints/imagenet/gamma=0.05/puCL/0/epoch=99.ckpt"
CKPT="logs/checkpoints/cifar10.hard/nP=1k-puCL/0/epoch=299.ckpt"
python3 run_lp.py --checkpoint $CKPT --dataset 'cifar10.hard' #--puPL True --algo 'PUkMeans'