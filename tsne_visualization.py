"""
Perform tsne eval of encoder
"""
import os
import time
from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
import torch
import yaml
from lightly.utils.dist import rank
from pytorch_lightning.callbacks import (
	LearningRateMonitor,
	ModelCheckpoint
)
from pytorch_lightning.loggers import TensorBoardLogger
from dataloader import DataManager
from training_framework import SimCLR


def _parse_args(verbose=True):
	parser = ArgumentParser(description="PreTraining Arguments")
	# ----- logging related ------ #
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=None,
		help="path of the ckpt e.g. os.path.join(os.getcwd(), pl_logs/checkpoints/cifar10/pt-SimCLR-0-v1.ckpt)"
	)
	parser.add_argument(
		'--dataset',
		type=str,
		default='cifar10.dog_cat',
		help='Pass Dataset'
	)
	args = parser.parse_args()
	verbose and print(args)
	return args


if __name__ == '__main__':
	args = _parse_args()
	config = yaml.load(
		open('config_pretrain.yaml'),
		Loader=yaml.FullLoader
	)
	torch.set_float32_matmul_precision("high")
	
	seed = 0
	# ---- parse config ----
	config = config[args.dataset]
	framework_config = config["framework_config"]
	data_config = config["data_config"]
	training_config = config["training_config"]
	pl.seed_everything(seed)
	# --- Data -----
	data_manager = DataManager(
		dataset=args.dataset,
		data_config=data_config
	)
	_, _, _, dataloader_test = data_manager.get_data()
	print('Loading PreTrained Model from Checkpoint {}'.format(args.checkpoint))
	model = SimCLR.load_from_checkpoint(
		args.checkpoint,
		framework_config=framework_config,
		training_config=training_config,
		data_config=data_config,
		val_dataloader=None,
		num_classes=data_manager.num_classes
	)