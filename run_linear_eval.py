import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import print_rank_zero, rank
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader import DataManager
from linear_head import LinearClassificationHead
from training_framework import SimCLR


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
		default="lp-SimCLR",
		help="logs saved inside exp sub-folder in the logs folder"
	)
	args = parser.parse_args()
	verbose and print(args)
	
	return args


def run_linear_eval(args, config, freeze_encoder: bool = True) -> None:
	"""Runs a linear evaluation on the given model.

	Parameters follow SimCLR [0] settings.

	The most important settings are:
		- Backbone: Frozen
		- Epochs: 90
		- Optimizer: SGD
		- Base Learning Rate: 0.1
		- Momentum: 0.9
		- Weight Decay: 0.0
		- LR Schedule: Cosine without warmup

	References:
		- [0]: SimCLR, 2020, https://arxiv.org/abs/2002.05709
	"""
	print_rank_zero("Running linear probing")
	config = config[args.dataset]
	framework_config = config["framework_config"]
	data_config = config["data_config"]
	training_config = config["training_config"]
	# --- Device Setup ---
	n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
	accelerator = "gpu" if torch.cuda.is_available() else "mps"
	strategy = "ddp" if n_gpus >= 2 else "auto"
	
	runs = []
	for seed in range(args.n_repeat):
		torch.set_float32_matmul_precision("high")
		pl.seed_everything(seed)
		# --- Data -----
		data_manager = DataManager(
			data_set=args.dataset,
			data_config=data_config,
			gpu_strategy=strategy
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
				val_dataloader=dataloader_train_sv,
				num_classes=data_manager.num_classes,
				gather_distributed=True if n_gpus >= 2 else False
			)
		else:
			print("You need to pass model chkpt to perform evaluation -- "
			      "since none provided training from scratch")
			model = SimCLR(
				framework_config=framework_config,
				training_config=training_config,
				data_config=data_config,
				val_dataloader=dataloader_train_sv,
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
		tf_logger = TensorBoardLogger(
			save_dir=os.path.join(args.log_dir, f"tf_logging/{args.dataset}"),
			name=f"{args.exp_name}",
			version=f"seed={seed}",
		),
		metric_callback = MetricCallback()
		# ----- Train linear classifier.
		trainer = Trainer(
			max_epochs=training_config.get('epochs'),
			accelerator=accelerator,
			devices=n_gpus,
			callbacks=[
				LearningRateMonitor(logging_interval='step'),
				metric_callback,
			],
			logger=tf_logger,
			# --- reproducibility
			sync_batchnorm=True if n_gpus >= 2 else False,
			deterministic=True,
			use_distributed_sampler=True if n_gpus >= 2 else False,
			check_val_every_n_epoch=training_config.get('eval_freq', 1),
			log_every_n_steps=1,
			precision="16-mixed",
			# strategy="ddp_find_unused_parameters_true",
		)
		trainer.fit(
			model=lin_classifier,
			train_dataloaders=dataloader_train_sv,
			val_dataloaders=dataloader_test,
		)
		run = {
			"batch_size": data_manager.train_batch_size,
			"epochs": training_config.get('epochs'),
			"max_accuracy": model.max_accuracy,
			"seed": seed,
		}
		for metric in ["val_top1", "val_top5"]:
			print_rank_zero(
				f"max linear {metric}: {max(metric_callback.val_metrics[metric])}"
			)
		if rank() == 0:
			runs.append(run)
			tf_logger.log_metrics(metrics=run)
			tf_logger.log_hyperparams(config)
			print(run)


if __name__ == '__main__':
	# ----- Parse Arguments ----- #
	arguments = _parse_args()
	if arguments.mode == 'lp':
		freeze = True
	elif arguments.mode == 'ft':
		freeze = False
	else:
		raise ValueError(
			'Need to Linear Probe or Finetune'
		)
	run_linear_eval(
		args=arguments,
		config=yaml.load(
			open('lin_eval_config.yaml'),
			Loader=yaml.FullLoader
		),
		freeze_encoder=freeze
	)
