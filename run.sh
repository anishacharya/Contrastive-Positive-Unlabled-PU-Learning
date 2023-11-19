# LP / FT evaluate a pretrained model
CKPT="/home/aa56927/CODE/ContrastivePULearning/logs/checkpoints/imagenet/gamma=0.5/puCL/0/epoch=99.ckpt"
python3 run_linear_eval.py --mode lp --checkpoint $CKPT --exp_name gamma=0.5/lp-ssCL-sup_ce --dataset imagenet


# Train contrastive encoder from scratch
# python3 run_representation.py --exp_name gamma=0.05/sCL --dataset imagenet
