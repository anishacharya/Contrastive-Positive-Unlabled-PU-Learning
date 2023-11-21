"""
Perform Linear Evaluation : Supports FineTuning, Linear Probing etc
"""
import os
from typing import Dict
from argparse import (
	ArgumentParser,
	Namespace
)
from pathlib import Path
import yaml
import numpy as np
import torch
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import (
	print_rank_zero,
	rank
)
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from dataloader import DataManager, get_pseudo_labels, PseudoLabeledData
from linear_head import LinearClassificationHead
from training_framework import SimCLR
from torchvision.transforms import v2
import lightly.data as data
from torch.utils.data import DataLoader


def _parse_args(verbose=True):
	parser = ArgumentParser(description="Linear Evaluation Arguments")
	parser.add_argument(
		'--dataset',
		type=str,
		default='cifar10.dog_cat',
		help='Pass Dataset'
	)
	parser.add_argument(
		"--mode",
		type=str,
		default="lp",
		help="lp / ft"
	)
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=None,
	)
	parser.add_argument(
		"--puPL",
		type=bool,
		default=True,
		help="if enabled, pseudo-labeling will happen before training"
	)
	parser.add_argument(
		"--algo",
		type=str,
		default='kMeans',
		help=" kMeans | kMeans++ | PUkMeans++ | DBSCAN "
	)
	parser.add_argument(
		'--n_repeat',
		type=int,
		default=1,
		help='Specify number of repeat runs'
	)
	# ----- logging related ------ #
	parser.add_argument(
		"--log_dir",
		type=Path,
		default=os.path.join(os.getcwd(), "logs")
	)
	parser.add_argument(
		"--exp_name",
		type=str,
		default="lp-testbed",
		help="logs saved inside exp sub-folder in the logs folder"
	)
	args = parser.parse_args()
	verbose and print(args)
	
	return args


def run_linear_eval(args: Namespace, config: Dict, freeze_encoder: bool = True) -> None:
	"""
	Runs a linear evaluation on the given model. If no model is given trains one from scratch
	"""
	# --- Device Setup ---
	n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
	# TODO: **** remove this condition when sCL with ddp support is implemented *****
	if n_gpus >= 2:
		raise NotImplementedError('Not all methods support ddp - run using single gpu')
	
	# --- parse configs ---
	config = config[args.dataset]
	framework_config = config["framework_config"]
	data_config = config["data_config"]
	training_config = config["training_config"]
	
	# --- Run Experiment for different seeds # -----
	val_acc = []
	for seed in range(args.n_repeat):
		torch.set_float32_matmul_precision("high")
		pl.seed_everything(seed)
		# --- Data -----
		data_manager = DataManager(
			dataset=args.dataset,
			data_config=data_config,
			gpu_strategy="ddp" if n_gpus >= 2 else "auto"
		)
		_, dataloader_train_sv, _, dataloader_test = data_manager.get_data()
		# --- Model -------
		if args.checkpoint is not None:
			print('Loading PreTrained Model from Checkpoint {}'.format(args.checkpoint))
			model = SimCLR.load_from_checkpoint(
				args.checkpoint,
				framework_config=framework_config,
				training_config=training_config,
				data_config=data_config,
				val_dataloader=dataloader_test,
				num_classes=data_manager.num_classes,
				gather_distributed=True if n_gpus >= 2 else False
			)
			model.max_accuracy = 0.0
		else:
			print("You need to pass model chkpt to perform evaluation -- "
			      "since None provided training a model from scratch")
			model = SimCLR(
				framework_config=framework_config,
				training_config=training_config,
				data_config=data_config,
				val_dataloader=dataloader_test,
				num_classes=data_manager.num_classes,
				gather_distributed=True if n_gpus >= 2 else False
			)
		lin_classifier = LinearClassificationHead(
			model=model,
			training_config=training_config,
			feature_dim=model.feat_dim,
			num_classes=data_manager.num_classes,
			freeze_model=freeze_encoder,
			topk=(1, 2),
		)
		# --- Logging -----
		logger = TensorBoardLogger(
			save_dir=os.path.join(args.log_dir, f"tf_logging/{args.dataset}"),
			name=f"{args.exp_name}",
			version=f"seed={seed}",
		),
		metric_callback = MetricCallback()
		# ----- Train linear classifier.
		trainer = Trainer(
			max_epochs=training_config.get('epochs'),
			# --- devices ----
			devices=n_gpus,
			accelerator="gpu" if torch.cuda.is_available() else "mps",
			strategy="ddp" if n_gpus >= 2 else "auto",
			# --- reproducibility
			sync_batchnorm=True if n_gpus >= 2 else False,
			deterministic=True,
			use_distributed_sampler=True if n_gpus >= 2 else False,
			# --- logging ---
			logger=logger,
			default_root_dir=args.log_dir,
			check_val_every_n_epoch=training_config.get('eval_freq', 1),
			log_every_n_steps=1,
			callbacks=[
				LearningRateMonitor(logging_interval='step'),
				metric_callback,
			],
			precision="16-mixed",
		)
		
		# Pseudo-label before fitting.
		# -----------------------------
		if args.puPL:
			data, pseudo_labels = get_pseudo_labels(
				original_dataloader=dataloader_train_sv,
				model=model,
				algo=args.algo,
				n_cluster=2
			)
			dataloader_train_sv.dataset.dataset.data = data
			dataloader_train_sv.dataset.dataset.targets = pseudo_labels
		
		# ---- Kick off Training
		trainer.fit(
			model=lin_classifier,
			train_dataloaders=dataloader_train_sv,
			val_dataloaders=dataloader_test,
		)
		if rank() == 0:
			val_acc.append(lin_classifier.max_accuracy.cpu())
		
		# ----- delete model and trainer + free up cuda memory ---
		del model
		del trainer
		torch.cuda.reset_peak_memory_stats()
		torch.cuda.empty_cache()
		print('val acc = {} +- {}'.format(np.mean(val_acc), np.var(val_acc)))


if __name__ == '__main__':
	# ----- Parse Arguments ----- #
	arguments = _parse_args()
	if arguments.mode == 'lp':
		freeze = True
		print_rank_zero("Running linear probing")
	
	elif arguments.mode == 'ft':
		freeze = False
		print_rank_zero("Running Finetuning")
	
	else:
		raise ValueError(
			'Need to Linear Probe or Finetune'
		)
	
	run_linear_eval(
		args=arguments,
		config=yaml.load(
			open('config_linear_eval.yaml'),
			Loader=yaml.FullLoader
		),
		freeze_encoder=freeze
	)
