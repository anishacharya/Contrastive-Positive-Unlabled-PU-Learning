# export CUDA_LAUNCH_BLOCKING=1
# LP / FT evaluate a pretrained model
#CKPT="logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
#python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name nP=5k/lp-puCL-ce_sup --dataset cifar10.dog_cat


#CKPT="/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
#python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name nP=5k/lp-puCL-pupL --dataset cifar10.dog_cat --puPL True

# ----- Embedding Visualization
#CKPT="logs/checkpoints/cifar10.dog_cat/nP=5k/puCL/0/epoch=299.ckpt"
#python3 run_tsne.py --checkpoint $CKPT --dataset cifar10.dog_cat --fig_name tsne_plots/nP=5k-puCL-ckpt=299


# train from scratch
# python3 run_lp.py --mode ft --exp_name gamma=0.5/nnPU --dataset cifar10.dog_cat


# Train contrastive encoder from scratch
python3 run_representation.py --exp_name nP=10k-puCL --dataset cifar10.hard
#python3 run_representation.py --exp_name single_dataset/nP=500/sCL --dataset cifar10.dog_cat